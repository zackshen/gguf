//! Memory-mapped GGUF file support
//!
//! This module provides memory-mapped file access for GGUF files,
//! which is more efficient for large files as it avoids loading
//! the entire file into memory.
//!
//! # Example
//!
//! ```rust,no_run
//! use gguf_rs::mmap::MmapGGUF;
//!
//! let mmap = MmapGGUF::open("model.gguf")?;
//! let model = mmap.decode()?;
//!
//! println!("Architecture: {}", model.model_family());
//! println!("Tensors: {}", model.num_tensor());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! - Lazy loading: only accessed pages are loaded into memory
//! - Efficient random access to tensor data
//! - OS-managed memory paging

use anyhow::{anyhow, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use crate::{ByteOrder, GGUFContainer, GGUFModel, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE};

/// Memory-mapped GGUF file
///
/// Provides efficient access to GGUF files using memory mapping.
/// The file is not loaded into memory until specific regions are accessed.
pub struct MmapGGUF {
    mmap: Mmap,
    byte_order: ByteOrder,
}

impl MmapGGUF {
    /// Open a GGUF file with memory mapping
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist
    /// - The file cannot be memory mapped
    /// - The file has an invalid magic number
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::mmap::MmapGGUF;
    ///
    /// let mmap = MmapGGUF::open("model.gguf")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(anyhow!("file not found: {}", path.display()));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Check magic number to determine byte order
        if mmap.len() < 4 {
            return Err(anyhow!("file too small to be a valid GGUF file"));
        }

        let magic = i32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);

        let byte_order = match magic {
            FILE_MAGIC_GGUF_LE => ByteOrder::LE,
            FILE_MAGIC_GGUF_BE => ByteOrder::BE,
            _ => return Err(anyhow!("invalid file magic: not a GGUF file")),
        };

        Ok(Self { mmap, byte_order })
    }

    /// Decode the GGUF file and return a model
    ///
    /// This parses the metadata and tensor information from the memory-mapped file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file contains malformed data.
    pub fn decode(&self) -> Result<GGUFModel> {
        // Create a cursor over the mmap data (skip magic, already consumed)
        let cursor = std::io::Cursor::new(&self.mmap[4..]);
        let mut container = GGUFContainer::new(self.byte_order.clone(), Box::new(cursor), u64::MAX);
        container.decode()
    }

    /// Get a reference to the raw memory-mapped data
    ///
    /// # Safety
    ///
    /// The returned slice is valid as long as this `MmapGGUF` is alive.
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Get the byte order of the file
    pub fn byte_order(&self) -> &ByteOrder {
        &self.byte_order
    }

    /// Get the size of the memory-mapped file
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if the file is empty
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_open() {
        let mmap = MmapGGUF::open("tests/test-le-v3.gguf").unwrap();
        assert!(!mmap.is_empty());
        assert!(matches!(mmap.byte_order(), ByteOrder::LE));
    }

    #[test]
    fn test_mmap_decode() {
        let mmap = MmapGGUF::open("tests/test-le-v3.gguf").unwrap();
        let model = mmap.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
    }

    #[test]
    fn test_mmap_file_not_found() {
        let result = MmapGGUF::open("nonexistent.gguf");
        assert!(result.is_err());
    }
}
