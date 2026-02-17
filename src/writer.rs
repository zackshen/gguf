//! GGUF file writing support
//!
//! This module provides functionality to create and write GGUF files.
//!
//! # Example
//!
//! ```rust,no_run
//! use gguf_rs::writer::{GGUFWriter, TensorInfo};
//! use std::collections::HashMap;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut writer = GGUFWriter::new("output.gguf", 3)?;
//!
//!     // Write metadata
//!     writer.write_metadata("general.architecture", "llama")?;
//!     writer.write_metadata_u32("llama.block_count", 12)?;
//!
//!     // Write tensor info
//!     let tensor = TensorInfo {
//!         name: "token_embd.weight".to_string(),
//!         shape: vec![4096, 32000],
//!         dtype: 0, // F32
//!     };
//!     writer.write_tensor_info(&tensor)?;
//!
//!     // Finalize the file
//!     writer.finalize()?;
//!
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::{FILE_MAGIC_GGUF_LE, GGUF_VERSION_V1, GGUF_VERSION_V2, GGUF_VERSION_V3};

/// Metadata value types for writing
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Tensor information for writing
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Name of the tensor
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<u64>,
    /// Data type (GGML type)
    pub dtype: u32,
}

/// GGUF file writer
///
/// Provides functionality to write GGUF files.
pub struct GGUFWriter {
    writer: BufWriter<File>,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    version: i32,
    data_start_offset: u64,
    alignment: u64,
}

impl GGUFWriter {
    /// Create a new GGUF writer
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `version` - GGUF version (1, 2, or 3)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn new<P: AsRef<Path>>(path: P, version: i32) -> Result<Self> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);

        let gguf_version = match version {
            1 => GGUF_VERSION_V1,
            2 => GGUF_VERSION_V2,
            3 => GGUF_VERSION_V3,
            _ => return Err(anyhow!("invalid GGUF version, must be 1, 2, or 3")),
        };

        Ok(Self {
            writer,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
            version: gguf_version,
            data_start_offset: 0,
            alignment: 32,
        })
    }

    /// Set alignment for tensor data
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    /// Add metadata (string value)
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata
            .insert(key.to_string(), MetadataValue::String(value.to_string()));
    }

    /// Add metadata (u32 value)
    pub fn add_metadata_u32(&mut self, key: &str, value: u32) {
        self.metadata.insert(key.to_string(), MetadataValue::Uint32(value));
    }

    /// Add metadata (i32 value)
    pub fn add_metadata_i32(&mut self, key: &str, value: i32) {
        self.metadata.insert(key.to_string(), MetadataValue::Int32(value));
    }

    /// Add metadata (u64 value)
    pub fn add_metadata_u64(&mut self, key: &str, value: u64) {
        self.metadata.insert(key.to_string(), MetadataValue::Uint64(value));
    }

    /// Add metadata (f32 value)
    pub fn add_metadata_f32(&mut self, key: &str, value: f32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Float32(value));
    }

    /// Add metadata (bool value)
    pub fn add_metadata_bool(&mut self, key: &str, value: bool) {
        self.metadata.insert(key.to_string(), MetadataValue::Bool(value));
    }

    /// Add metadata (array value)
    pub fn add_metadata_array(&mut self, key: &str, value: Vec<MetadataValue>) {
        self.metadata.insert(key.to_string(), MetadataValue::Array(value));
    }

    /// Add tensor info
    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        self.tensors.push(tensor);
    }

    /// Write the complete GGUF file
    ///
    /// This writes the header, metadata, tensor info, and tensor data.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write(&mut self) -> Result<()> {
        // Write header
        self.write_header()?;

        // Write metadata
        self.write_metadata_section()?;

        // Write tensor info
        self.write_tensor_info_section()?;

        // Calculate data offset (aligned)
        let current_pos = self.writer.stream_position()?;
        let padding = (self.alignment - (current_pos % self.alignment)) % self.alignment;
        for _ in 0..padding {
            self.writer.write_u8(0)?;
        }

        self.data_start_offset = self.writer.stream_position()?;

        Ok(())
    }

    /// Write tensor data at the appropriate offset
    ///
    /// # Arguments
    ///
    /// * `tensor_index` - Index of the tensor (as added)
    /// * `data` - Raw tensor data bytes
    pub fn write_tensor_data(&mut self, tensor_index: usize, data: &[u8]) -> Result<()> {
        if tensor_index >= self.tensors.len() {
            return Err(anyhow!("invalid tensor index"));
        }

        // Calculate offset for this tensor
        let mut offset = self.data_start_offset;
        for i in 0..tensor_index {
            offset += self.calculate_tensor_size(&self.tensors[i])?;
        }

        // Seek and write
        self.writer.seek(SeekFrom::Start(offset))?;
        self.writer.write_all(data)?;

        Ok(())
    }

    /// Finalize the file (flush and sync)
    pub fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    // Private methods

    fn write_header(&mut self) -> Result<()> {
        // Magic
        self.writer.write_i32::<LittleEndian>(FILE_MAGIC_GGUF_LE)?;
        // Version
        self.writer.write_i32::<LittleEndian>(self.version)?;
        // Tensor count
        self.write_size(self.tensors.len() as u64)?;
        // Metadata count
        self.write_size(self.metadata.len() as u64)?;

        Ok(())
    }

    fn write_metadata_section(&mut self) -> Result<()> {
        // Clone all items to avoid borrow conflicts
        let items: Vec<(String, MetadataValue)> = self
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (key, value) in items {
            // Write key
            self.write_string(&key)?;
            // Write value type and value
            self.write_metadata_value(&value)?;
        }
        Ok(())
    }

    fn write_metadata_value(&mut self, value: &MetadataValue) -> Result<()> {
        match value {
            MetadataValue::Uint8(v) => {
                self.writer.write_u32::<LittleEndian>(0)?; // type
                self.writer.write_u8(*v)?;
            }
            MetadataValue::Int8(v) => {
                self.writer.write_u32::<LittleEndian>(1)?;
                self.writer.write_i8(*v)?;
            }
            MetadataValue::Uint16(v) => {
                self.writer.write_u32::<LittleEndian>(2)?;
                self.writer.write_u16::<LittleEndian>(*v)?;
            }
            MetadataValue::Int16(v) => {
                self.writer.write_u32::<LittleEndian>(3)?;
                self.writer.write_i16::<LittleEndian>(*v)?;
            }
            MetadataValue::Uint32(v) => {
                self.writer.write_u32::<LittleEndian>(4)?;
                self.writer.write_u32::<LittleEndian>(*v)?;
            }
            MetadataValue::Int32(v) => {
                self.writer.write_u32::<LittleEndian>(5)?;
                self.writer.write_i32::<LittleEndian>(*v)?;
            }
            MetadataValue::Float32(v) => {
                self.writer.write_u32::<LittleEndian>(6)?;
                self.writer.write_f32::<LittleEndian>(*v)?;
            }
            MetadataValue::Bool(v) => {
                self.writer.write_u32::<LittleEndian>(7)?;
                self.writer.write_u8(if *v { 1 } else { 0 })?;
            }
            MetadataValue::String(v) => {
                self.writer.write_u32::<LittleEndian>(8)?;
                self.write_string(v)?;
            }
            MetadataValue::Array(arr) => {
                self.writer.write_u32::<LittleEndian>(9)?;
                self.write_size(arr.len() as u64)?;
                // All elements must be same type
                if !arr.is_empty() {
                    for elem in arr {
                        self.write_metadata_value(elem)?;
                    }
                }
            }
            MetadataValue::Uint64(v) => {
                self.writer.write_u32::<LittleEndian>(10)?;
                self.writer.write_u64::<LittleEndian>(*v)?;
            }
            MetadataValue::Int64(v) => {
                self.writer.write_u32::<LittleEndian>(11)?;
                self.writer.write_i64::<LittleEndian>(*v)?;
            }
            MetadataValue::Float64(v) => {
                self.writer.write_u32::<LittleEndian>(12)?;
                self.writer.write_f64::<LittleEndian>(*v)?;
            }
        }
        Ok(())
    }

    fn write_tensor_info_section(&mut self) -> Result<()> {
        // Clone to avoid borrow conflicts
        let tensors = self.tensors.clone();
        for tensor in &tensors {
            // Name
            self.write_string(&tensor.name)?;
            // Number of dimensions
            self.writer.write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
            // Shape
            for dim in &tensor.shape {
                self.writer.write_u64::<LittleEndian>(*dim)?;
            }
            // Data type
            self.writer.write_u32::<LittleEndian>(tensor.dtype)?;
            // Offset (will be filled later)
            self.writer.write_u64::<LittleEndian>(0)?;
        }
        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        self.write_size(bytes.len() as u64)?;
        self.writer.write_all(bytes)?;
        Ok(())
    }

    fn write_size(&mut self, size: u64) -> Result<()> {
        match self.version {
            GGUF_VERSION_V1 => {
                self.writer.write_u32::<LittleEndian>(size as u32)?;
            }
            _ => {
                self.writer.write_u64::<LittleEndian>(size)?;
            }
        }
        Ok(())
    }

    fn calculate_tensor_size(&self, tensor: &TensorInfo) -> Result<u64> {
        // Simplified calculation - actual size depends on dtype
        let elements: u64 = tensor.shape.iter().product();
        let bytes_per_element = match tensor.dtype {
            0 => 4, // F32
            1 => 2, // F16
            _ => 1, // Default to 1 byte for quantized
        };
        Ok(elements * bytes_per_element)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_writer_create() {
        let writer = GGUFWriter::new("/tmp/test_output.gguf", 3);
        assert!(writer.is_ok());
    }

    #[test]
    fn test_write_metadata() {
        let mut writer = GGUFWriter::new("/tmp/test_metadata.gguf", 3).unwrap();
        writer.add_metadata("general.architecture", "llama");
        writer.add_metadata_u32("llama.block_count", 12);
        writer.add_metadata_f32("test.value", 3.14);

        let result = writer.write();
        assert!(result.is_ok());
    }

    #[test]
    fn test_write_with_tensor() {
        let mut writer = GGUFWriter::new("/tmp/test_tensor.gguf", 3).unwrap();
        writer.add_metadata("general.architecture", "test");

        let tensor = TensorInfo {
            name: "test.weight".to_string(),
            shape: vec![10, 20],
            dtype: 0, // F32
        };
        writer.add_tensor(tensor);

        let result = writer.write();
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_version() {
        let result = GGUFWriter::new("/tmp/test_invalid.gguf", 5);
        assert!(result.is_err());
    }
}
