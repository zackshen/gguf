//! GGUF file writing support
//!
//! This module provides functionality to create and write GGUF files
//! that conform to the official GGUF specification:
//! <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>
//!
//! # Example
//!
//! ```rust,ignore
//! use gguf_rs::writer::{GGUFWriter, TensorInfo};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut writer = GGUFWriter::new("output.gguf", 3)?;
//!     writer.add_metadata("general.architecture", "llama");
//!     writer.add_metadata_u32("llama.block_count", 12);
//!
//!     let tensor = TensorInfo {
//!         name: "token_embd.weight".to_string(),
//!         shape: vec![4096, 32000],
//!         dtype: 0,
//!     };
//!     writer.add_tensor(tensor);
//!     writer.write()?;
//!     writer.finalize()?;
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

/// Default tensor-data alignment, per GGUF spec (must be a multiple of 8).
pub const DEFAULT_ALIGNMENT: u64 = 32;

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
/// Produces a spec-conformant GGUF file. Tensor offsets in the info section
/// are computed at `write()` time and properly aligned per `general.alignment`.
pub struct GGUFWriter {
    writer: BufWriter<File>,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: Vec<TensorInfo>,
    version: i32,
    data_start_offset: u64,
    alignment: u64,
    tensor_offsets: Vec<u64>,
}

impl GGUFWriter {
    /// Create a new GGUF writer
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    /// * `version` - GGUF version (1, 2, or 3)
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
            alignment: DEFAULT_ALIGNMENT,
            tensor_offsets: Vec::new(),
        })
    }

    /// Set alignment for tensor data. Must be a multiple of 8 per spec.
    /// The chosen alignment will be recorded in `general.alignment` metadata
    /// when it differs from [`DEFAULT_ALIGNMENT`].
    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata
            .insert(key.to_string(), MetadataValue::String(value.to_string()));
    }
    pub fn add_metadata_u32(&mut self, key: &str, value: u32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Uint32(value));
    }
    pub fn add_metadata_i32(&mut self, key: &str, value: i32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Int32(value));
    }
    pub fn add_metadata_u64(&mut self, key: &str, value: u64) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Uint64(value));
    }
    pub fn add_metadata_f32(&mut self, key: &str, value: f32) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Float32(value));
    }
    pub fn add_metadata_bool(&mut self, key: &str, value: bool) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Bool(value));
    }
    pub fn add_metadata_array(&mut self, key: &str, value: Vec<MetadataValue>) {
        self.metadata
            .insert(key.to_string(), MetadataValue::Array(value));
    }

    pub fn add_tensor(&mut self, tensor: TensorInfo) {
        self.tensors.push(tensor);
    }

    /// Write the complete GGUF file structure (header, metadata, tensor info,
    /// alignment padding). Reserves enough zeroed space for all tensor data so
    /// that subsequent [`write_tensor_data`](Self::write_tensor_data) calls
    /// seek into a pre-sized region.
    pub fn write(&mut self) -> Result<()> {
        if self.alignment == 0 || self.alignment % 8 != 0 {
            return Err(anyhow!("alignment must be a non-zero multiple of 8"));
        }

        if self.alignment != DEFAULT_ALIGNMENT && !self.metadata.contains_key("general.alignment") {
            let aligned_u32 = u32::try_from(self.alignment)
                .map_err(|_| anyhow!("alignment {} too large for u32", self.alignment))?;
            self.metadata
                .insert("general.alignment".to_string(), MetadataValue::Uint32(aligned_u32));
        }

        let mut tensor_offsets = Vec::with_capacity(self.tensors.len());
        let mut cur: u64 = 0;
        for t in &self.tensors {
            tensor_offsets.push(cur);
            let sz = calculate_tensor_size(t)?;
            cur = cur
                .checked_add(sz)
                .ok_or_else(|| anyhow!("tensor offset overflow"))?;
            let pad = (self.alignment - (cur % self.alignment)) % self.alignment;
            cur = cur
                .checked_add(pad)
                .ok_or_else(|| anyhow!("tensor offset overflow"))?;
        }
        self.tensor_offsets = tensor_offsets;
        let total_data_bytes = cur;

        self.write_header()?;
        self.write_metadata_section()?;
        self.write_tensor_info_section()?;

        let current_pos = self.writer.stream_position()?;
        let padding = (self.alignment - (current_pos % self.alignment)) % self.alignment;
        for _ in 0..padding {
            self.writer.write_u8(0)?;
        }
        self.data_start_offset = self.writer.stream_position()?;

        for _ in 0..total_data_bytes {
            self.writer.write_u8(0)?;
        }

        Ok(())
    }

    /// Write tensor data at the tensor's pre-computed offset.
    pub fn write_tensor_data(&mut self, tensor_index: usize, data: &[u8]) -> Result<()> {
        if tensor_index >= self.tensors.len() {
            return Err(anyhow!("invalid tensor index"));
        }
        if self.data_start_offset == 0 {
            return Err(anyhow!("write() must be called before write_tensor_data()"));
        }
        let expected = calculate_tensor_size(&self.tensors[tensor_index])?;
        if (data.len() as u64) > expected {
            return Err(anyhow!(
                "tensor data {} bytes exceeds tensor size {} bytes",
                data.len(),
                expected
            ));
        }
        let abs = self
            .data_start_offset
            .checked_add(self.tensor_offsets[tensor_index])
            .ok_or_else(|| anyhow!("absolute offset overflow"))?;
        self.writer.seek(SeekFrom::Start(abs))?;
        self.writer.write_all(data)?;
        Ok(())
    }

    /// Flush buffered writes to disk.
    pub fn finalize(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn write_header(&mut self) -> Result<()> {
        self.writer.write_i32::<LittleEndian>(FILE_MAGIC_GGUF_LE)?;
        self.writer.write_i32::<LittleEndian>(self.version)?;
        self.write_size(self.tensors.len() as u64)?;
        self.write_size(self.metadata.len() as u64)?;
        Ok(())
    }

    fn write_metadata_section(&mut self) -> Result<()> {
        let items: Vec<(String, MetadataValue)> = self
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (key, value) in items {
            self.write_string(&key)?;
            self.write_metadata_value(&value)?;
        }
        Ok(())
    }

    fn write_metadata_value(&mut self, value: &MetadataValue) -> Result<()> {
        let type_id = metadata_type_id(value)?;
        self.writer.write_u32::<LittleEndian>(type_id)?;
        match value {
            MetadataValue::Array(arr) => self.write_array_payload(arr)?,
            other => self.write_scalar_payload(other)?,
        }
        Ok(())
    }

    fn write_array_payload(&mut self, arr: &[MetadataValue]) -> Result<()> {
        let item_type: u32 = match arr.first() {
            Some(v) => metadata_type_id(v)?,
            None => 0,
        };
        if item_type == 9 {
            return Err(anyhow!("nested arrays are not supported by GGUF"));
        }
        self.writer.write_u32::<LittleEndian>(item_type)?;
        self.write_size(arr.len() as u64)?;
        for elem in arr {
            let elem_type = metadata_type_id(elem)?;
            if elem_type != item_type {
                return Err(anyhow!(
                    "array elements must share one type (expected {}, got {})",
                    item_type,
                    elem_type
                ));
            }
            self.write_scalar_payload(elem)?;
        }
        Ok(())
    }

    fn write_scalar_payload(&mut self, value: &MetadataValue) -> Result<()> {
        match value {
            MetadataValue::Uint8(v) => self.writer.write_u8(*v)?,
            MetadataValue::Int8(v) => self.writer.write_i8(*v)?,
            MetadataValue::Uint16(v) => self.writer.write_u16::<LittleEndian>(*v)?,
            MetadataValue::Int16(v) => self.writer.write_i16::<LittleEndian>(*v)?,
            MetadataValue::Uint32(v) => self.writer.write_u32::<LittleEndian>(*v)?,
            MetadataValue::Int32(v) => self.writer.write_i32::<LittleEndian>(*v)?,
            MetadataValue::Float32(v) => self.writer.write_f32::<LittleEndian>(*v)?,
            MetadataValue::Bool(v) => self.writer.write_u8(if *v { 1 } else { 0 })?,
            MetadataValue::String(v) => self.write_string(v)?,
            MetadataValue::Uint64(v) => self.writer.write_u64::<LittleEndian>(*v)?,
            MetadataValue::Int64(v) => self.writer.write_i64::<LittleEndian>(*v)?,
            MetadataValue::Float64(v) => self.writer.write_f64::<LittleEndian>(*v)?,
            MetadataValue::Array(_) => {
                return Err(anyhow!("nested arrays are not supported by GGUF"))
            }
        }
        Ok(())
    }

    fn write_tensor_info_section(&mut self) -> Result<()> {
        let tensors = self.tensors.clone();
        let offsets = self.tensor_offsets.clone();
        for (i, tensor) in tensors.iter().enumerate() {
            self.write_string(&tensor.name)?;
            self.writer
                .write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
            for dim in &tensor.shape {
                self.writer.write_u64::<LittleEndian>(*dim)?;
            }
            self.writer.write_u32::<LittleEndian>(tensor.dtype)?;
            self.writer.write_u64::<LittleEndian>(offsets[i])?;
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
                let v32 = u32::try_from(size)
                    .map_err(|_| anyhow!("size {} exceeds u32 for v1 GGUF", size))?;
                self.writer.write_u32::<LittleEndian>(v32)?;
            }
            _ => {
                self.writer.write_u64::<LittleEndian>(size)?;
            }
        }
        Ok(())
    }
}

fn metadata_type_id(v: &MetadataValue) -> Result<u32> {
    Ok(match v {
        MetadataValue::Uint8(_) => 0,
        MetadataValue::Int8(_) => 1,
        MetadataValue::Uint16(_) => 2,
        MetadataValue::Int16(_) => 3,
        MetadataValue::Uint32(_) => 4,
        MetadataValue::Int32(_) => 5,
        MetadataValue::Float32(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::Array(_) => 9,
        MetadataValue::Uint64(_) => 10,
        MetadataValue::Int64(_) => 11,
        MetadataValue::Float64(_) => 12,
    })
}

/// Block size + per-block byte size for each GGML type, matching the reader.
fn block_and_type_size(kind: u32) -> Result<(u64, u64)> {
    let block_size: u64 = if kind < 2 {
        1
    } else if kind < 10 {
        32
    } else {
        256
    };
    let type_size: u64 = match kind {
        0 => 4,
        1 => 2,
        2 => 2 + block_size / 2,
        3 => 2 + 2 + block_size / 2,
        4 | 5 => return Err(anyhow!("GGML kind {} (Q4_2/Q4_3) is unsupported", kind)),
        6 => 2 + 4 + block_size / 2,
        7 => 2 + 2 + 4 + block_size / 2,
        8 => 2 + block_size,
        9 => 4 + 4 + block_size,
        10 => block_size / 16 + block_size / 4 + 2 + 2,
        11 => block_size / 8 + block_size / 4 + 12 + 2,
        12 => 2 + 2 + 12 + block_size / 2,
        13 => 2 + 2 + 12 + block_size / 8 + block_size / 2,
        14 => block_size / 2 + block_size / 4 + block_size / 16 + 2,
        15 => 4 + block_size + block_size / 16 * 2,
        16 => 2 + block_size / 8 * 2,
        17 => 2 + block_size / 8 * 2 + block_size / 32,
        18 => 2 + 3 * (block_size / 8),
        19 => 2 + block_size / 8 + block_size / 16,
        20 => 2 + 16,
        21 => 2 + 13 * (block_size / 32) + block_size / 64,
        22 => 2 + block_size / 4 + block_size / 16,
        23 => 2 + 2 + block_size / 64 + block_size / 2,
        24 => 1,
        25 => 2,
        26 => 4,
        27 => 8,
        28 => 8,
        29 => block_size / 8 + block_size / 16 + block_size / 32,
        30 => 2,
        31 | 32 | 33 => return Err(anyhow!("GGML kind {} (Q4_0_4_*) is unsupported", kind)),
        34 => 2 + block_size / 64 + (block_size - 4 * block_size / 64) / 5,
        35 => 2 + block_size / 4,
        36 | 37 | 38 => return Err(anyhow!("GGML kind {} (IQ4_NL_*) is unsupported", kind)),
        39 => block_size + 1 + 16,
        _ => return Err(anyhow!("invalid GGML kind {}", kind)),
    };
    Ok((block_size, type_size))
}

/// Byte size of a tensor's raw data, matching the reader's formula.
pub fn calculate_tensor_size(tensor: &TensorInfo) -> Result<u64> {
    let (block_size, type_size) = block_and_type_size(tensor.dtype)?;
    if block_size == 0 {
        return Err(anyhow!("zero block_size for GGML kind {}", tensor.dtype));
    }
    let elements: u64 = tensor.shape.iter().try_fold(1u64, |acc, d| {
        acc.checked_mul(*d)
            .ok_or_else(|| anyhow!("element count overflow"))
    })?;
    let raw = elements
        .checked_mul(type_size)
        .ok_or_else(|| anyhow!("tensor byte size overflow"))?;
    Ok(raw / block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("gguf_writer_test_{name}.gguf"))
    }

    fn read_back(path: &std::path::Path) -> crate::GGUFModel {
        let mut c =
            crate::get_gguf_container_array_size(path.to_str().unwrap(), u64::MAX).expect("open");
        c.decode().expect("decode")
    }

    #[test]
    fn test_writer_create() {
        let writer = GGUFWriter::new(tmp_path("create"), 3);
        assert!(writer.is_ok());
    }

    #[test]
    fn test_invalid_version() {
        let result = GGUFWriter::new(tmp_path("badver"), 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_writer_v1_v2_v3_roundtrip() {
        for version in [1, 2, 3] {
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
        w.metadata.insert("k.u8".into(), MetadataValue::Uint8(7));
        w.metadata.insert("k.i8".into(), MetadataValue::Int8(-7));
        w.metadata.insert("k.u16".into(), MetadataValue::Uint16(7));
        w.metadata.insert("k.i16".into(), MetadataValue::Int16(-7));
        w.metadata.insert("k.i64".into(), MetadataValue::Int64(-7));
        w.metadata
            .insert("k.f64".into(), MetadataValue::Float64(2.5));
        w.metadata
            .insert("k.str".into(), MetadataValue::String("hi".into()));

        w.write().unwrap();
        w.finalize().unwrap();

        let model = read_back(&path);
        let kv = model.metadata();
        assert_eq!(kv.get("k.u8").unwrap().as_u64().unwrap(), 7);
        assert_eq!(kv.get("k.i8").unwrap().as_i64().unwrap(), -7);
        assert_eq!(kv.get("k.u16").unwrap().as_u64().unwrap(), 7);
        assert_eq!(kv.get("k.i16").unwrap().as_i64().unwrap(), -7);
        assert_eq!(kv.get("k.u32").unwrap().as_u64().unwrap(), 42);
        assert_eq!(kv.get("k.i32").unwrap().as_i64().unwrap(), -42);
        assert_eq!(kv.get("k.u64").unwrap().as_u64().unwrap(), 1_000_000);
        assert_eq!(kv.get("k.i64").unwrap().as_i64().unwrap(), -7);
        assert!((kv.get("k.f32").unwrap().as_f64().unwrap() - 1.25).abs() < 1e-6);
        assert!((kv.get("k.f64").unwrap().as_f64().unwrap() - 2.5).abs() < 1e-9);
        assert_eq!(kv.get("k.bool").unwrap().as_bool().unwrap(), true);
        assert_eq!(kv.get("k.str").unwrap().as_str().unwrap(), "hi");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_array_roundtrip() {
        let path = tmp_path("arr_rt");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata("general.architecture", "x");
        w.add_metadata_array(
            "arr.u32",
            vec![
                MetadataValue::Uint32(10),
                MetadataValue::Uint32(20),
                MetadataValue::Uint32(30),
            ],
        );
        w.add_metadata_array(
            "arr.str",
            vec![
                MetadataValue::String("a".into()),
                MetadataValue::String("bb".into()),
                MetadataValue::String("ccc".into()),
            ],
        );
        w.add_metadata_array("arr.empty", vec![]);
        w.write().unwrap();
        w.finalize().unwrap();

        let model = read_back(&path);
        let kv = model.metadata();
        let u32_arr = kv.get("arr.u32").unwrap().as_array().unwrap();
        assert_eq!(u32_arr.len(), 3);
        assert_eq!(u32_arr[0].as_u64().unwrap(), 10);
        assert_eq!(u32_arr[2].as_u64().unwrap(), 30);
        let str_arr = kv.get("arr.str").unwrap().as_array().unwrap();
        assert_eq!(str_arr[1].as_str().unwrap(), "bb");
        assert_eq!(kv.get("arr.empty").unwrap().as_array().unwrap().len(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_array_mixed_types_error() {
        let path = tmp_path("arr_mixed");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata_array(
            "mixed",
            vec![MetadataValue::Uint32(1), MetadataValue::String("x".into())],
        );
        let err = w.write().err().unwrap();
        assert!(err.to_string().contains("must share one type"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_nested_array_error() {
        let path = tmp_path("nested");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata_array("nested", vec![MetadataValue::Array(vec![MetadataValue::Uint8(1)])]);
        let err = w.write().err().unwrap();
        assert!(err.to_string().contains("nested arrays"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_with_tensors_roundtrip() {
        let path = tmp_path("tensors_rt");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata("general.architecture", "test");
        w.add_tensor(TensorInfo {
            name: "t.weight".to_string(),
            shape: vec![4, 4],
            dtype: 0, // F32
        });
        w.add_tensor(TensorInfo {
            name: "t.bias".to_string(),
            shape: vec![4],
            dtype: 1, // F16
        });
        w.write().unwrap();

        let payload0 = (0..16u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect::<Vec<u8>>();
        let payload1 = vec![0xABu8; 4 * 2];
        w.write_tensor_data(0, &payload0).unwrap();
        w.write_tensor_data(1, &payload1).unwrap();
        w.finalize().unwrap();

        let model = read_back(&path);
        let ts = model.tensors();
        assert_eq!(ts.len(), 2);
        assert_eq!(ts[0].name, "t.weight");
        // offsets must be strictly increasing and aligned to default 32
        assert_eq!(ts[0].offset % DEFAULT_ALIGNMENT, 0);
        assert_eq!(ts[1].offset % DEFAULT_ALIGNMENT, 0);
        assert!(ts[1].offset >= ts[0].offset + ts[0].size);

        // Verify written tensor bytes are recoverable via the recorded offset.
        let raw = std::fs::read(&path).unwrap();
        let data_start = model.data_offset() as usize;
        let off0 = data_start + ts[0].offset as usize;
        assert_eq!(&raw[off0..off0 + payload0.len()], payload0.as_slice());
        let off1 = data_start + ts[1].offset as usize;
        assert_eq!(&raw[off1..off1 + payload1.len()], payload1.as_slice());

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
    fn test_write_tensor_data_before_write_errors() {
        let path = tmp_path("before_write");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_tensor(TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 0,
        });
        let err = w.write_tensor_data(0, &[0u8; 4]).err().unwrap();
        assert!(err.to_string().contains("write() must be called"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_write_tensor_data_too_large_errors() {
        let path = tmp_path("too_large");
        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_tensor(TensorInfo {
            name: "x".into(),
            shape: vec![2],
            dtype: 0, // F32 -> 8 bytes
        });
        w.write().unwrap();
        let err = w.write_tensor_data(0, &[0u8; 16]).err().unwrap();
        assert!(err.to_string().contains("exceeds tensor size"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_alignment_recorded_when_custom() {
        let path = tmp_path("align64");
        let mut w = GGUFWriter::new(&path, 3).unwrap().with_alignment(64);
        w.add_metadata("general.architecture", "t");
        w.add_tensor(TensorInfo {
            name: "t".into(),
            shape: vec![2],
            dtype: 0,
        });
        w.write().unwrap();
        w.finalize().unwrap();
        let model = read_back(&path);
        // general.alignment auto-inserted at the requested value
        assert_eq!(
            model
                .metadata()
                .get("general.alignment")
                .unwrap()
                .as_u64()
                .unwrap(),
            64
        );
        assert_eq!(model.tensors()[0].offset % 64, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_alignment_invalid_rejected() {
        let path = tmp_path("align_bad");
        let mut w = GGUFWriter::new(&path, 3).unwrap().with_alignment(7);
        let err = w.write().err().unwrap();
        assert!(err.to_string().contains("multiple of 8"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_writer_v1_size_overflow_rejected() {
        let path = tmp_path("v1_big");
        let mut w = GGUFWriter::new(&path, 1).unwrap();
        // Build a string longer than u32::MAX bytes is impractical; instead
        // simulate via metadata count exceeding u32::MAX by mocking the field directly.
        // We exercise this by triggering the explicit error path with a too-big shape.
        w.add_tensor(TensorInfo {
            name: "t".into(),
            shape: vec![u32::MAX as u64 + 1],
            dtype: 0,
        });
        // calculate_tensor_size will not overflow at this point; the v1 write_size
        // check fires on the tensor count which is fine — so we hand-test write_size.
        let res = w.write_size(u32::MAX as u64 + 1);
        assert!(res.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_calculate_tensor_size_branches() {
        // F32: 2*3 elements * 4 bytes / 1 block = 24
        let f32_t = TensorInfo {
            name: "a".into(),
            shape: vec![2, 3],
            dtype: 0,
        };
        assert_eq!(calculate_tensor_size(&f32_t).unwrap(), 24);
        // F16: 4 * 2 / 1 = 8
        let f16_t = TensorInfo {
            name: "b".into(),
            shape: vec![4],
            dtype: 1,
        };
        assert_eq!(calculate_tensor_size(&f16_t).unwrap(), 8);
        // Q8_0: block 32, type 34, 32 elems -> 32*34/32 = 34
        let q8 = TensorInfo {
            name: "c".into(),
            shape: vec![32],
            dtype: 8,
        };
        assert_eq!(calculate_tensor_size(&q8).unwrap(), 34);
        // Q4_K: block 256, type 2+2+12+128=144, 256 elems -> 256*144/256 = 144
        let q4k = TensorInfo {
            name: "d".into(),
            shape: vec![256],
            dtype: 12,
        };
        assert_eq!(calculate_tensor_size(&q4k).unwrap(), 144);
    }

    #[test]
    fn test_calculate_tensor_size_unsupported_kind() {
        let bad = TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 4,
        }; // Q4_2
        assert!(calculate_tensor_size(&bad).is_err());
        let bad2 = TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 31,
        }; // Q4_0_4_4
        assert!(calculate_tensor_size(&bad2).is_err());
        let bad3 = TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 36,
        }; // IQ4_NL_4_4
        assert!(calculate_tensor_size(&bad3).is_err());
        let bad4 = TensorInfo {
            name: "x".into(),
            shape: vec![1],
            dtype: 999,
        };
        assert!(calculate_tensor_size(&bad4).is_err());
    }

    #[test]
    fn test_calculate_tensor_size_all_supported_kinds() {
        let supported = [
            0u32, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 34, 35, 39,
        ];
        for k in supported {
            let t = TensorInfo {
                name: "t".into(),
                shape: vec![256],
                dtype: k,
            };
            assert!(calculate_tensor_size(&t).is_ok(), "kind {} should compute a size", k);
        }
    }
}
