//! Async support for GGUF file parsing
//!
//! This module provides async versions of the parsing functions,
//! useful for non-blocking I/O in async applications.
//!
//! # Example
//!
//! ```rust,no_run
//! use gguf_rs::async_io::AsyncGGUF;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut container = AsyncGGUF::open("model.gguf").await?;
//!     let model = container.decode().await?;
//!
//!     println!("Architecture: {}", model.model_family());
//!     println!("Tensors: {}", model.num_tensor());
//!
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! - Non-blocking file I/O
//! - Compatible with tokio runtime
//! - Same API as sync version

use anyhow::{anyhow, Result};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{ByteOrder, GGUFModel, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE};

/// Async GGUF file container
///
/// Provides async access to GGUF files using tokio.
pub struct AsyncGGUF {
    byte_order: ByteOrder,
    reader: Box<dyn tokio::io::AsyncRead + Unpin + Send>,
    max_array_size: u64,
}

impl AsyncGGUF {
    /// Open a GGUF file asynchronously
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist or has an invalid format.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::async_io::AsyncGGUF;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let container = AsyncGGUF::open("model.gguf").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(anyhow!("file not found: {}", path.display()));
        }

        let mut file = File::open(path).await?;

        // Read magic number
        let mut magic_bytes = [0u8; 4];
        file.read_exact(&mut magic_bytes).await?;
        let magic = i32::from_le_bytes(magic_bytes);

        let byte_order = match magic {
            FILE_MAGIC_GGUF_LE => ByteOrder::LE,
            FILE_MAGIC_GGUF_BE => ByteOrder::BE,
            _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
        };

        // Seek back to start (after magic)
        // We'll re-read from beginning in decode()
        let boxed_reader: Box<dyn tokio::io::AsyncRead + Unpin + Send> =
            Box::new(tokio::io::BufReader::new(file));

        Ok(Self {
            byte_order,
            reader: Box::new(MagicSkippedReader(boxed_reader)),
            max_array_size: 3,
        })
    }

    /// Create a new AsyncGGUF with custom max array size
    pub fn with_max_array_size(mut self, max_array_size: u64) -> Self {
        self.max_array_size = max_array_size;
        self
    }

    /// Decode the GGUF file asynchronously
    ///
    /// # Errors
    ///
    /// Returns an error if the file contains malformed data.
    pub async fn decode(&mut self) -> Result<GGUFModel> {
        // For now, we use a synchronous approach with the async reader
        // A fully async implementation would require more complex state machine
        let mut all_data = Vec::new();
        self.reader.read_to_end(&mut all_data).await?;

        let cursor = std::io::Cursor::new(all_data);
        let mut container =
            crate::GGUFContainer::new(self.byte_order.clone(), Box::new(cursor), self.max_array_size);
        container.decode()
    }
}

/// Wrapper to skip the magic bytes already read
struct MagicSkippedReader(Box<dyn tokio::io::AsyncRead + Unpin + Send>);

impl tokio::io::AsyncRead for MagicSkippedReader {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.0).poll_read(cx, buf)
    }
}

/// Open and decode a GGUF file in one async operation
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::async_io::read_gguf;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let model = read_gguf("model.gguf").await?;
/// println!("Architecture: {}", model.model_family());
/// # Ok(())
/// # }
/// ```
pub async fn read_gguf<P: AsRef<std::path::Path>>(path: P) -> Result<GGUFModel> {
    let mut container = AsyncGGUF::open(path).await?;
    container.decode().await
}

/// Open and decode a GGUF file with custom array size
pub async fn read_gguf_with_array_size<P: AsRef<std::path::Path>>(
    path: P,
    max_array_size: u64,
) -> Result<GGUFModel> {
    let mut container = AsyncGGUF::open(path).await?.with_max_array_size(max_array_size);
    container.decode().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_async_open() {
        let container = AsyncGGUF::open("tests/test-le-v3.gguf").await;
        assert!(container.is_ok());
    }

    #[tokio::test]
    async fn test_async_decode() {
        let mut container = AsyncGGUF::open("tests/test-le-v3.gguf").await.unwrap();
        let model = container.decode().await.unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
    }

    #[tokio::test]
    async fn test_async_read_gguf() {
        let model = read_gguf("tests/test-le-v3.gguf").await.unwrap();
        assert_eq!(model.model_family(), "llama");
    }

    #[tokio::test]
    async fn test_async_file_not_found() {
        let result = AsyncGGUF::open("nonexistent.gguf").await;
        assert!(result.is_err());
    }
}
