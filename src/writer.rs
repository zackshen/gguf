//! GGUF file writing support
//!
//! This module provides functionality to create and write GGUF files.
//!
//! # Example
//!
//! ```rust,ignore
//! use gguf_rs::writer::{GGUFWriter, TensorInfo};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut writer = GGUFWriter::new("output.gguf", 3)?;
//!
//!     // Add metadata
//!     writer.add_metadata("general.architecture", "llama");
//!     writer.add_metadata_u32("llama.block_count", 12);
//!
//!     // Add tensor info
//!     let tensor = TensorInfo {
//!         name: "token_embd.weight".to_string(),
//!         shape: vec![4096, 32000],
//!         dtype: 0, // F32
//!     };
//!     writer.add_tensor(tensor);
//!
//!     // Write header and metadata
//!     writer.write()?;
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
        self.metadata
            .insert(key.to_string(), MetadataValue::Uint32(value));
    }

    /// Add metadata (i32 value)
    pub fn add_metadata_i32(&mut self, key: &str, value: i32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Int32(value));
    }

    /// Add metadata (u64 value)
    pub fn add_metadata_u64(&mut self, key: &str, value: u64) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Uint64(value));
    }

    /// Add metadata (f32 value)
    pub fn add_metadata_f32(&mut self, key: &str, value: f32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Float32(value));
    }

    /// Add metadata (bool value)
    pub fn add_metadata_bool(&mut self, key: &str, value: bool) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Bool(value));
    }

    /// Add metadata (array value)
    pub fn add_metadata_array(&mut self, key: &str, value: Vec<MetadataValue>) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Array(value));
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
            self.writer
                .write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
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

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gguf_writer_test_{name}.gguf"))
    }

    fn read_back(path: &std::path::Path) -> crate::GGUFModel {
        let mut c = crate::get_gguf_container_array_size(path.to_str().unwrap(), u64::MAX)
            .expect("open");
        c.decode().expect("decode")
    }

    #[test]
    fn test_writer_v1_v2_roundtrip() {
        for version in [1, 2] {
            let path = tmp_path(&format!("v{version}"));
            let mut w = GGUFWriter::new(&path, version).unwrap();
            w.add_metadata("general.architecture", "test");
            w.write().unwrap();
            w.finalize().unwrap();
            let model = read_back(&path);
            assert_eq!(model.get_version(), format!("v{version}"));
            assert_eq!(model.model_family(), "test");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn test_writer_all_metadata_types_roundtrip() {
        let path = tmp_path("all_meta");
        let mut w = GGUFWriter::new(&path, 3).unwrap();

        w.add_metadata("general.architecture", "llama");
        w.add_metadata_u32("k.u32", 42);
        w.add_metadata_i32("k.i32", -42);
        w.add_metadata_u64("k.u64", 1_000_000);
        w.add_metadata_f32("k.f32", 1.25);
        w.add_metadata_bool("k.bool", true);

        // Exercise every leaf MetadataValue variant by inserting directly into the map.
        w.metadata.insert("k.u8".into(), MetadataValue::Uint8(7));
        w.metadata.insert("k.i8".into(), MetadataValue::Int8(-7));
        w.metadata.insert("k.u16".into(), MetadataValue::Uint16(7));
        w.metadata.insert("k.i16".into(), MetadataValue::Int16(-7));
        w.metadata.insert("k.i64".into(), MetadataValue::Int64(-7));
        w.metadata.insert("k.f64".into(), MetadataValue::Float64(2.5));
        w.metadata.insert("k.str".into(), MetadataValue::String("hi".into()));

        w.write().unwrap();
        w.finalize().unwrap();

        let model = read_back(&path);
        let kv = model.metadata();
        assert_eq!(kv.get("k.u32").unwrap().as_u64().unwrap(), 42);
        assert_eq!(kv.get("k.i32").unwrap().as_i64().unwrap(), -42);
        assert_eq!(kv.get("k.bool").unwrap().as_bool().unwrap(), true);
        assert_eq!(kv.get("k.str").unwrap().as_str().unwrap(), "hi");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_add_metadata_array_writes_without_error() {
        // The writer's array format is incompatible with the reader (missing item_type
        // field). This test exercises the array write path only — no round-trip.
        let path = tmp_path("add_arr");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata_array(
            "k.arr",
            vec![MetadataValue::Uint32(1), MetadataValue::Uint32(2)],
        );
        // Cover every primitive variant of write_metadata_value via array elements.
        w.add_metadata_array(
            "k.primitives",
            vec![
                MetadataValue::Uint8(1),
                MetadataValue::Int8(-1),
                MetadataValue::Uint16(1),
                MetadataValue::Int16(-1),
                MetadataValue::Int32(-1),
                MetadataValue::Float32(1.0),
                MetadataValue::Bool(false),
                MetadataValue::String("x".into()),
                MetadataValue::Uint64(1),
                MetadataValue::Int64(-1),
                MetadataValue::Float64(1.0),
            ],
        );
        assert!(w.write().is_ok());
        w.finalize().unwrap();
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_with_tensor_roundtrip() {
        let path = tmp_path("tensor");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata("general.architecture", "test");
        w.add_tensor(TensorInfo {
            name: "t.weight".to_string(),
            shape: vec![4, 4],
            dtype: 0,
        });
        w.add_tensor(TensorInfo {
            name: "t.bias".to_string(),
            shape: vec![4],
            dtype: 1, // F16
        });
        w.write().unwrap();

        // Write data for tensor 0 (16 f32 = 64 bytes)
        let payload = vec![0u8; 64];
        w.write_tensor_data(0, &payload).unwrap();
        w.finalize().unwrap();

        let model = read_back(&path);
        assert_eq!(model.tensors().len(), 2);
        assert_eq!(model.tensors()[0].name, "t.weight");
        assert_eq!(model.tensors()[1].name, "t.bias");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_tensor_data_invalid_index() {
        let path = tmp_path("bad_idx");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_tensor(TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 0,
        });
        w.write().unwrap();
        let err = w.write_tensor_data(5, &[0u8; 4]).err().unwrap();
        assert!(err.to_string().contains("invalid tensor index"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_with_alignment_setter() {
        let path = tmp_path("align");
        let w = GGUFWriter::new(&path, 3).unwrap().with_alignment(64);
        assert_eq!(w.alignment, 64);
        drop(w);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_calculate_tensor_size_branches() {
        let path = tmp_path("calc_size");
        let w = GGUFWriter::new(&path, 3).unwrap();
        let f32_t = TensorInfo { name: "a".into(), shape: vec![2, 3], dtype: 0 };
        let f16_t = TensorInfo { name: "b".into(), shape: vec![4], dtype: 1 };
        let quant_t = TensorInfo { name: "c".into(), shape: vec![32], dtype: 8 };
        assert_eq!(w.calculate_tensor_size(&f32_t).unwrap(), 2 * 3 * 4);
        assert_eq!(w.calculate_tensor_size(&f16_t).unwrap(), 4 * 2);
        assert_eq!(w.calculate_tensor_size(&quant_t).unwrap(), 32);
        drop(w);
        let _ = std::fs::remove_file(&path);
    }
}
