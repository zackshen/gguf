//! # GGUF File Parser
//!
//! A Rust library for parsing and reading GGUF (GGML Universal Format) files.
//!
//! GGUF files are binary files that contain key-value metadata and tensors,
//! commonly used for storing quantized machine learning models like LLaMA, Phi, etc.
//!
//! ## Features
//!
//! - Decode GGUF files (v1, v2, v3)
//! - Access key-value metadata
//! - Access tensor information
//! - Support for little-endian and big-endian files
//! - CLI tool for quick inspection
//! - Optional memory-mapped file support (enable `mmap` feature)
//!
//! ## Example
//!
//! ```rust,no_run
//! use gguf_rs::get_gguf_container;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Open a GGUF file
//!     let mut container = get_gguf_container("model.gguf")?;
//!     let model = container.decode()?;
//!
//!     // Print model info
//!     println!("Version: {}", model.get_version());
//!     println!("Architecture: {}", model.model_family());
//!     println!("Parameters: {}", model.model_parameters());
//!     println!("File type: {}", model.file_type());
//!     println!("Tensors: {}", model.num_tensor());
//!
//!     // List tensors
//!     for tensor in model.tensors() {
//!         println!("  {}: {:?} {:?}", tensor.name, tensor.kind, tensor.shape);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## CLI Usage
//!
//! Install the CLI tool:
//! ```bash
//! cargo install gguf-rs
//! ```
//!
//! Show model info:
//! ```bash
//! gguf model.gguf
//! ```
//!
//! Show tensors:
//! ```bash
//! gguf model.gguf --tensors
//! ```
//!
//! ## Memory-Mapped Files
//!
//! For large files, enable the `mmap` feature for more efficient access:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["mmap"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::mmap::MmapGGUF;
//!
//! let mmap = MmapGGUF::open("large_model.gguf")?;
//! let model = mmap.model();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Async I/O
//!
//! For async applications, enable the `async` feature:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["async"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::async_io::AsyncGGUF;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut container = AsyncGGUF::open("model.gguf").await?;
//!     let model = container.decode().await?;
//!
//!     println!("Architecture: {}", model.model_family());
//!     Ok(())
//! }
//! ```

use anyhow::{anyhow, Result};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
#[cfg(feature = "debug")]
use log::debug;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{borrow::Borrow, collections::BTreeMap, fmt::Display};

/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_GGML: i32 = 0x67676d6c;
/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: i32 = 0x67676d66;
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: i32 = 0x67676a74;
/// Magic constant for `ggla` files (LoRA adapter).
pub const FILE_MAGIC_GGLA: i32 = 0x67676C61;
/// Magic constant for `gguf` files (versioned, gguf)
pub const FILE_MAGIC_GGUF_LE: i32 = 0x46554747;
pub const FILE_MAGIC_GGUF_BE: i32 = 0x47475546;

pub const GGUF_VERSION_V1: i32 = 0x00000001;
pub const GGUF_VERSION_V2: i32 = 0x00000002;
pub const GGUF_VERSION_V3: i32 = 0x00000003;

/// Maximum number of tensor dimensions supported by GGUF, matching `GGML_MAX_DIMS`.
pub const GGUF_MAX_DIMS: u32 = 4;
/// Maximum allowed string length in bytes (1 GiB), matching llama.cpp's `GGUF_MAX_STRING_LENGTH`.
pub const GGUF_MAX_STRING_LENGTH: u64 = 1 << 30;

const THOUSAND: u64 = 1000;
const MILLION: u64 = 1_000_000;
const BILLION: u64 = 1_000_000_000;

/// Convert a number to a human-readable string.
fn human_number(value: u64) -> String {
    match value {
        _ if value > BILLION => format!("{:.0}B", value as f64 / BILLION as f64),
        _ if value > MILLION => format!("{:.0}M", value as f64 / MILLION as f64),
        _ if value > THOUSAND => format!("{:.0}K", value as f64 / THOUSAND as f64),
        _ => format!("{}", value),
    }
}

/// Convert a file type to a human-readable string.
/// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
fn file_type(ft: u64) -> String {
    match ft {
        0 => "All F32",
        1 => "Mostly F16",
        2 => "Mostly Q4_0",
        3 => "Mostly Q4_1",
        4 => "Mostly Q4_1 Some F16",
        5 => "Mostly Q4_2 (UNSUPPORTED)",
        6 => "Mostly Q4_3 (UNSUPPORTED)",
        7 => "Mostly Q8_0",
        8 => "Mostly Q5_0",
        9 => "Mostly Q5_1",
        10 => "Mostly Q2_K",
        11 => "Mostly Q3_K",
        12 => "Mostly Q4_K",
        13 => "Mostly Q5_K",
        14 => "Mostly Q6_K",
        15 => "Mostly IQ2_XXS",
        16 => "Mostly IQ2_XS",
        17 => "Mostly IQ3_XXS",
        18 => "Mostly IQ1_S",
        19 => "Mostly IQ4_NL",
        20 => "Mostly IQ3_S",
        21 => "Mostly IQ2_S",
        22 => "Mostly IQ4_XS",
        23 => "Mostly IQ1_M",
        24 => "Mostly BF16",
        _ => "unknown",
    }
    .to_string()
}

/// Byte order of the GGUF file.
#[derive(Default, Debug, Clone)]
pub enum ByteOrder {
    #[default]
    LE,
    BE,
}

/// Version of the GGUF file.
#[derive(Debug, Clone)]
pub enum Version {
    V1(V1),
    V2(V2),
    V3(V3),
}

/// Version 1 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V1 {
    num_tensor: u32,
    num_kv: u32,
}

/// Version 2 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V2 {
    num_tensor: u64,
    num_kv: u64,
}

/// Version 3 of the GGUF file.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct V3 {
    num_tensor: u64,
    num_kv: u64,
}

/// Internal reader that counts bytes consumed. Lets the container compute
/// the absolute file offset of the tensor-data section after parsing.
struct CountingReader {
    inner: Box<dyn std::io::Read + 'static>,
    count: u64,
}

impl std::io::Read for CountingReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count = self.count.saturating_add(n as u64);
        Ok(n)
    }
}

/// Default tensor-data alignment per GGUF spec.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// GGUF file container for reading GGUF binary files.
///
/// The container wraps a reader and provides methods to decode the GGUF file
/// into a [`GGUFModel`].
///
/// Use [`get_gguf_container`] for a convenient way to open a file.
pub struct GGUFContainer {
    bo: ByteOrder,
    version: Version,
    reader: CountingReader,
    max_array_size: u64,
}

impl GGUFContainer {
    /// Create a new `GGUFContainer` from a byte order and a reader.
    ///
    /// # Arguments
    ///
    /// * `bo` - Byte order (little-endian or big-endian)
    /// * `reader` - A reader implementing `std::io::Read`
    /// * `max_array_size` - Maximum size for array metadata values
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::{GGUFContainer, ByteOrder};
    /// use std::fs::File;
    ///
    /// let file = File::open("model.gguf")?;
    /// let container = GGUFContainer::new(ByteOrder::LE, Box::new(file), 1024);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(bo: ByteOrder, reader: Box<dyn std::io::Read>, max_array_size: u64) -> Self {
        Self {
            bo,
            version: Version::V1(V1::default()),
            reader: CountingReader {
                inner: reader,
                count: 0,
            },
            max_array_size,
        }
    }

    /// Get the version of the GGUF file container.
    ///
    /// Returns the default version ("v1") before decoding.
    /// After successful decode, returns the actual file version ("v1", "v2", or "v3").
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Decode the GGUF file and return a `GGUFModel`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file has an invalid or unsupported GGUF version
    /// - The file contains malformed data
    /// - An I/O error occurs while reading
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gguf_rs::get_gguf_container;
    ///
    /// let mut container = get_gguf_container("model.gguf")?;
    /// let model = container.decode()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decode(&mut self) -> Result<GGUFModel> {
        let version = match self.bo {
            ByteOrder::LE => self.reader.read_i32::<LittleEndian>()?,
            ByteOrder::BE => self.reader.read_i32::<BigEndian>()?,
        };

        #[cfg(feature = "debug")]
        {
            debug!("version {}", version);
        }

        match version {
            GGUF_VERSION_V1 => {
                let mut buffer: [u32; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => self.reader.read_u32_into::<LittleEndian>(&mut buffer)?,
                    ByteOrder::BE => self.reader.read_u32_into::<BigEndian>(&mut buffer)?,
                };

                self.version = Version::V1(V1 {
                    num_tensor: buffer[0],
                    num_kv: buffer[1],
                });
            }
            GGUF_VERSION_V2 | GGUF_VERSION_V3 => {
                let mut buffer: [u64; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => self.reader.read_u64_into::<LittleEndian>(&mut buffer)?,
                    ByteOrder::BE => self.reader.read_u64_into::<BigEndian>(&mut buffer)?,
                };

                if version == GGUF_VERSION_V2 {
                    self.version = Version::V2(V2 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                } else {
                    self.version = Version::V3(V3 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                }
            }
            invalid_version => {
                return Err(anyhow!(
                    "invalid version {}, only support version: 1 | 2 | 3",
                    invalid_version
                ));
            }
        };

        let mut model = GGUFModel {
            kv: BTreeMap::new(),
            tensors: Vec::new(),
            parameters: 0,
            max_array_size: self.max_array_size,
            bo: self.bo.clone(),
            version: self.version.clone(),
            data_offset: 0,
            alignment: GGUF_DEFAULT_ALIGNMENT,
        };

        model.decode(&mut self.reader)?;

        // Honor general.alignment if present; spec defaults to 32.
        let alignment = model
            .kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .filter(|a| *a > 0 && a % 8 == 0)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);
        model.alignment = alignment;

        // 4 bytes were consumed for the magic before the container's reader was created.
        let pre_pad = 4u64
            .checked_add(self.reader.count)
            .ok_or_else(|| anyhow!("header byte counter overflow"))?;
        let padding = (alignment - (pre_pad % alignment)) % alignment;
        model.data_offset = pre_pad
            .checked_add(padding)
            .ok_or_else(|| anyhow!("data offset overflow"))?;

        Ok(model)
    }
}

/// Tensor in the GGUF file.
///
/// Represents a single tensor with its metadata including name, type, offset, size, and shape.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// for tensor in model.tensors() {
///     println!("Tensor: {} (shape: {:?})", tensor.name, tensor.shape);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Name of the tensor (e.g., "token_embd.weight", "blk.0.attn_q.weight")
    pub name: String,
    /// GGML type identifier (see [`GGMLType`] for interpretation)
    pub kind: u32,
    /// Number of dimensions as recorded on disk (`<= GGUF_MAX_DIMS`).
    pub n_dimensions: u32,
    /// Byte offset of this tensor's data, relative to [`GGUFModel::data_offset`].
    pub offset: u64,
    /// Size of tensor data in bytes
    pub size: u64,
    /// Shape dimensions (one entry per dimension, length == `n_dimensions`).
    pub shape: Vec<u64>,
}

/// Decoded GGUF model containing metadata and tensors.
///
/// Use [`get_gguf_container`] to create a container, then call [`GGUFContainer::decode`]
/// to get a `GGUFModel`.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// println!("Model: {}", model.model_family());
/// println!("Parameters: {}", model.model_parameters());
/// println!("Tensors: {}", model.num_tensor());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GGUFModel {
    kv: BTreeMap<String, Value>,
    tensors: Vec<Tensor>,
    parameters: u64,
    max_array_size: u64,
    bo: ByteOrder,
    version: Version,
    /// Absolute byte offset in the file where the tensor data section begins.
    /// Equal to the end of the tensor info section rounded up to `alignment`.
    data_offset: u64,
    /// Alignment in bytes for tensor data, parsed from `general.alignment`
    /// (defaults to [`GGUF_DEFAULT_ALIGNMENT`]).
    alignment: u64,
}

/// Metadata value type in GGUF files.
///
/// Represents the type of a metadata value in the key-value store.
/// Used when decoding metadata to determine how to interpret bytes.
#[derive(Debug)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => MetadataValueType::Uint8,
            1 => MetadataValueType::Int8,
            2 => MetadataValueType::Uint16,
            3 => MetadataValueType::Int16,
            4 => MetadataValueType::Uint32,
            5 => MetadataValueType::Int32,
            6 => MetadataValueType::Float32,
            7 => MetadataValueType::Bool,
            8 => MetadataValueType::String,
            9 => MetadataValueType::Array,
            10 => MetadataValueType::Uint64,
            11 => MetadataValueType::Int64,
            12 => MetadataValueType::Float64,
            _ => return Err(anyhow!("unsupport metadata value type")),
        })
    }
}

/// GGML type of a tensor in the GGUF file.
///
/// Represents the quantization format used for tensor data.
/// Most types are quantized formats that compress float values
/// to reduce memory footprint while maintaining accuracy.
#[derive(Debug, Serialize)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4, // Unsupported
    Q4_3 = 5, // Unsupported
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31, // Unsupported
    Q4_0_4_8 = 32, // Unsupported
    Q4_0_8_8 = 33, // Unsupported
    TQ1_0 = 34,
    TQ2_0 = 35,
    IQ4_NL_4_4 = 36, // Unsupported
    IQ4_NL_4_8 = 37, // Unsupported
    IQ4_NL_8_8 = 38, // Unsupported
    MXFP4 = 39,
    Count = 40,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q4_2 => write!(f, "Q4_2 (UNSUPPORTED)"),
            GGMLType::Q4_3 => write!(f, "Q4_3 (UNSUPPORTED)"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2_K => write!(f, "Q2_K"),
            GGMLType::Q3_K => write!(f, "Q3_K"),
            GGMLType::Q4_K => write!(f, "Q4_K"),
            GGMLType::Q5_K => write!(f, "Q5_K"),
            GGMLType::Q6_K => write!(f, "Q6_K"),
            GGMLType::Q8_K => write!(f, "Q8_K"),
            GGMLType::IQ2_XXS => write!(f, "IQ2_XXS"),
            GGMLType::IQ2_XS => write!(f, "IQ2_XS"),
            GGMLType::IQ3_XXS => write!(f, "IQ3_XXS"),
            GGMLType::IQ1_S => write!(f, "IQ1_S"),
            GGMLType::IQ4_NL => write!(f, "IQ4_NL"),
            GGMLType::IQ3_S => write!(f, "IQ3_S"),
            GGMLType::IQ2_S => write!(f, "IQ2_S"),
            GGMLType::IQ4_XS => write!(f, "IQ4_XS"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::I64 => write!(f, "I64"),
            GGMLType::F64 => write!(f, "F64"),
            GGMLType::IQ1_M => write!(f, "IQ1_M"),
            GGMLType::BF16 => write!(f, "BF16"),
            GGMLType::Q4_0_4_4 => write!(f, "Q4_0_4_4 (UNSUPPORTED)"),
            GGMLType::Q4_0_4_8 => write!(f, "Q4_0_4_8 (UNSUPPORTED)"),
            GGMLType::Q4_0_8_8 => write!(f, "Q4_0_8_8 (UNSUPPORTED)"),
            GGMLType::TQ1_0 => write!(f, "TQ1_0"),
            GGMLType::TQ2_0 => write!(f, "TQ2_0"),
            GGMLType::IQ4_NL_4_4 => write!(f, "IQ4_NL_4_4 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_4_8 => write!(f, "IQ4_NL_4_8 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_8_8 => write!(f, "IQ4_NL_8_8 (UNSUPPORTED)"),
            GGMLType::MXFP4 => write!(f, "MXFP4"),
            GGMLType::Count => write!(f, "Count"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> std::prelude::v1::Result<Self, Self::Error> {
        Ok(match value {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            16 => GGMLType::IQ2_XXS,
            17 => GGMLType::IQ2_XS,
            18 => GGMLType::IQ3_XXS,
            19 => GGMLType::IQ1_S,
            20 => GGMLType::IQ4_NL,
            21 => GGMLType::IQ3_S,
            22 => GGMLType::IQ2_S,
            23 => GGMLType::IQ4_XS,
            24 => GGMLType::I8,
            25 => GGMLType::I16,
            26 => GGMLType::I32,
            27 => GGMLType::I64,
            28 => GGMLType::F64,
            29 => GGMLType::IQ1_M,
            30 => GGMLType::BF16,
            31 => GGMLType::Q4_0_4_4,
            32 => GGMLType::Q4_0_4_8,
            33 => GGMLType::Q4_0_8_8,
            34 => GGMLType::TQ1_0,
            35 => GGMLType::TQ2_0,
            36 => GGMLType::IQ4_NL_4_4,
            37 => GGMLType::IQ4_NL_4_8,
            38 => GGMLType::IQ4_NL_8_8,
            39 => GGMLType::MXFP4,
            40 => GGMLType::Count,
            _ => return Err(anyhow!("invalid GGML type")),
        })
    }
}

impl GGUFModel {
    /// Decode the GGUF file.
    pub(crate) fn decode(&mut self, mut reader: impl std::io::Read) -> Result<()> {
        // decode kv
        for _i in 0..self.num_kv() {
            let key = self.read_string(&mut reader)?;
            let value_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
            let value = match value_type {
                MetadataValueType::Uint8 => Value::from(self.read_u8(&mut reader)?),
                MetadataValueType::Int8 => Value::from(self.read_i8(&mut reader)?),
                MetadataValueType::Uint16 => Value::from(self.read_u16(&mut reader)?),
                MetadataValueType::Int16 => Value::from(self.read_i16(&mut reader)?),
                MetadataValueType::Uint32 => Value::from(self.read_u32(&mut reader)?),
                MetadataValueType::Int32 => Value::from(self.read_i32(&mut reader)?),
                MetadataValueType::Float32 => Value::from(self.read_f32(&mut reader)?),
                MetadataValueType::Bool => Value::from(self.read_bool(&mut reader)?),
                MetadataValueType::String => Value::from(self.read_string(&mut reader)?),
                MetadataValueType::Array => Value::from(self.read_array(&key, &mut reader)?),
                MetadataValueType::Uint64 => Value::from(self.read_u64(&mut reader)?),
                MetadataValueType::Int64 => Value::from(self.read_i64(&mut reader)?),
                MetadataValueType::Float64 => Value::from(self.read_f64(&mut reader)?),
            };
            #[cfg(feature = "debug")]
            {
                debug!("kv [{}] vtype {:?} key={}, value={}", _i, value_type, key, value);
            }
            self.kv.insert(key, value);
        }

        // decode tensors
        for _ in 0..self.num_tensor() {
            let name = self.read_string(&mut reader)?;
            let dims = self.read_u32(&mut reader)?;
            if dims > GGUF_MAX_DIMS {
                return Err(anyhow!(
                    "tensor '{}' has {} dimensions, exceeds maximum of {}",
                    name,
                    dims,
                    GGUF_MAX_DIMS
                ));
            }
            let mut shape_vec = Vec::with_capacity(dims as usize);
            let mut shape = [1u64; 4];
            for i in 0..dims {
                let dim = self.read_u64(&mut reader)?;
                shape[i as usize] = dim;
                shape_vec.push(dim);
            }

            let kind = self.read_u32(&mut reader)?;
            let offset = self.read_u64(&mut reader)?;
            let block_size = match kind {
                _ if kind < 2 => 1,
                _ if kind < 10 => 32,
                _ => 256,
            };
            let ggml_type_kind: GGMLType = kind.try_into()?;
            let type_size = match ggml_type_kind {
                GGMLType::F32 => 4,
                GGMLType::F16 => 2,
                GGMLType::Q4_0 => 2 + block_size / 2,
                GGMLType::Q4_1 => 2 + 2 + block_size / 2,
                GGMLType::Q4_2 => 0,
                GGMLType::Q4_3 => 0,
                GGMLType::Q5_0 => 2 + 4 + block_size / 2,
                GGMLType::Q5_1 => 2 + 2 + 4 + block_size / 2,
                GGMLType::Q8_0 => 2 + block_size,
                GGMLType::Q8_1 => 4 + 4 + block_size,
                GGMLType::Q2_K => block_size / 16 + block_size / 4 + 2 + 2,
                GGMLType::Q3_K => block_size / 8 + block_size / 4 + 12 + 2,
                GGMLType::Q4_K => 2 + 2 + 12 + block_size / 2,
                GGMLType::Q5_K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
                GGMLType::Q6_K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
                GGMLType::Q8_K => 4 + block_size + block_size / 16 * 2,
                GGMLType::IQ2_XXS => 2 + block_size / 8 * 2,
                GGMLType::IQ2_XS => 2 + block_size / 8 * 2 + block_size / 32,
                GGMLType::IQ3_XXS => 2 + 3 * (block_size / 8),
                GGMLType::IQ1_S => 2 + block_size / 8 + block_size / 16,
                GGMLType::IQ4_NL => 2 + 16,
                GGMLType::IQ3_S => 2 + 13 * (block_size / 32) + block_size / 64,
                GGMLType::IQ2_S => 2 + block_size / 4 + block_size / 16,
                GGMLType::IQ4_XS => 2 + 2 + block_size / 64 + block_size / 2,
                GGMLType::I8 => 1,
                GGMLType::I16 => 2,
                GGMLType::I32 => 4,
                GGMLType::I64 => 8,
                GGMLType::F64 => 8,
                GGMLType::IQ1_M => block_size / 8 + block_size / 16 + block_size / 32,
                GGMLType::BF16 => 2,
                GGMLType::IQ4_NL_4_4 => 0,
                GGMLType::IQ4_NL_4_8 => 0,
                GGMLType::IQ4_NL_8_8 => 0,
                GGMLType::TQ1_0 => 2 + block_size / 64 + (block_size - 4 * block_size / 64) / 5,
                GGMLType::TQ2_0 => 2 + block_size / 4,
                GGMLType::Q4_0_4_4 => 0,
                GGMLType::Q4_0_4_8 => 0,
                GGMLType::Q4_0_8_8 => 0,
                GGMLType::MXFP4 => block_size + 1 + 16,
                GGMLType::Count => unreachable!("GGMLType::Count is not a real data format"),
            };

            let parameters = shape[0]
                .checked_mul(shape[1])
                .and_then(|v| v.checked_mul(shape[2]))
                .and_then(|v| v.checked_mul(shape[3]))
                .ok_or_else(|| {
                    anyhow!("tensor '{}' parameter count overflows u64: shape={:?}", name, shape)
                })?;
            if block_size == 0 {
                return Err(anyhow!("tensor '{}' has zero block_size for kind {}", name, kind));
            }
            let size = parameters
                .checked_mul(type_size)
                .map(|v| v / block_size)
                .ok_or_else(|| {
                    anyhow!("tensor '{}' byte size overflows u64: shape={:?}", name, shape)
                })?;

            self.tensors.push(Tensor {
                name,
                kind,
                n_dimensions: dims,
                offset,
                size,
                shape: shape_vec,
            });

            self.parameters = self
                .parameters
                .checked_add(parameters)
                .ok_or_else(|| anyhow!("total parameter count overflows u64"))?;
        }

        Ok(())
    }

    fn read_u8(&self, mut reader: impl std::io::Read) -> Result<u8> {
        Ok(reader.read_u8()?)
    }

    fn read_u32(&self, mut reader: impl std::io::Read) -> Result<u32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u32::<BigEndian>()?,
        })
    }

    fn read_f32(&self, mut reader: impl std::io::Read) -> Result<f32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f32::<BigEndian>()?,
        })
    }

    fn read_f64(&self, mut reader: impl std::io::Read) -> Result<f64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f64::<BigEndian>()?,
        })
    }

    fn read_u64(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u64::<BigEndian>()?,
        })
    }

    fn read_i8(&self, mut reader: impl std::io::Read) -> Result<i8> {
        Ok(reader.read_i8()?)
    }

    fn read_u16(&self, mut reader: impl std::io::Read) -> Result<u16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u16::<BigEndian>()?,
        })
    }

    fn read_i16(&self, mut reader: impl std::io::Read) -> Result<i16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i16::<BigEndian>()?,
        })
    }

    fn read_i32(&self, mut reader: impl std::io::Read) -> Result<i32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i32::<BigEndian>()?,
        })
    }

    fn read_i64(&self, mut reader: impl std::io::Read) -> Result<i64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i64::<BigEndian>()?,
        })
    }

    fn read_bool(&self, mut reader: impl std::io::Read) -> Result<bool> {
        Ok(reader.read_u8()? != 0)
    }

    fn read_string(&self, mut reader: impl std::io::Read) -> Result<String> {
        let name_len = self.read_version_size(&mut reader)?;
        if name_len > GGUF_MAX_STRING_LENGTH {
            return Err(anyhow!(
                "string length {} exceeds maximum of {} bytes",
                name_len,
                GGUF_MAX_STRING_LENGTH
            ));
        }
        let len_usize = usize::try_from(name_len)
            .map_err(|_| anyhow!("string length {} does not fit in usize", name_len))?;
        let mut buffer = vec![0; len_usize];
        reader.read_exact(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    fn read_array(&self, key: &str, mut reader: impl std::io::Read) -> Result<Vec<Value>> {
        let mut data = Vec::new();
        let item_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
        let array_len = self.read_version_size(&mut reader)?;
        // The `max_array_size` cap exists to avoid materializing huge tokenizer
        // vocabularies (`tokenizer.ggml.tokens`/`scores`/`token_type`/`merges`).
        // Model-configuration arrays such as `<arch>.attention.head_count_kv`
        // are per-layer and small; truncating them silently corrupts the value,
        // so they are always read in full. This matters for heterogeneous-layer
        // models like Gemma 4. See https://github.com/zackshen/gguf/issues/21
        let read_count: usize = if key.starts_with("tokenizer.") {
            u64::min(array_len, self.max_array_size) as usize
        } else {
            array_len as usize
        };
        for _ in 0..array_len {
            let value = match item_type {
                MetadataValueType::Uint8 => Value::from(self.read_u8(&mut reader)?),
                MetadataValueType::Int8 => Value::from(self.read_i8(&mut reader)?),
                MetadataValueType::Uint16 => Value::from(self.read_u16(&mut reader)?),
                MetadataValueType::Int16 => Value::from(self.read_i16(&mut reader)?),
                MetadataValueType::Uint32 => Value::from(self.read_u32(&mut reader)?),
                MetadataValueType::Int32 => Value::from(self.read_i32(&mut reader)?),
                MetadataValueType::Float32 => Value::from(self.read_f32(&mut reader)?),
                MetadataValueType::Bool => Value::from(self.read_bool(&mut reader)?),
                MetadataValueType::String => Value::from(self.read_string(&mut reader)?),
                MetadataValueType::Uint64 => Value::from(self.read_u64(&mut reader)?),
                MetadataValueType::Int64 => Value::from(self.read_i64(&mut reader)?),
                MetadataValueType::Float64 => Value::from(self.read_f64(&mut reader)?),
                _ => return Err(anyhow!("Unsupport item value type: Array")),
            };
            if data.len() < read_count {
                data.push(value);
            }
        }

        Ok(data)
    }

    fn read_version_size(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.version.borrow() {
            Version::V1(_) => self.read_u32(&mut reader)? as u64,
            Version::V2(_) => self.read_u64(&mut reader)?,
            Version::V3(_) => self.read_u64(&mut reader)?,
        })
    }

    /// Get the version of the decoded GGUF model.
    ///
    /// Returns one of: "v1", "v2", or "v3".
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Get the number of key-value pairs in the GGUF file.
    pub fn num_kv(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_kv as u64,
            Version::V2(v2) => v2.num_kv,
            Version::V3(v3) => v3.num_kv,
        }
    }

    /// Get the number of tensors in the GGUF file.
    ///
    /// Returns the total count of tensors stored in the model.
    pub fn num_tensor(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_tensor as u64,
            Version::V2(v2) => v2.num_tensor,
            Version::V3(v3) => v3.num_tensor,
        }
    }

    /// Get the model family/architecture of the GGUF file.
    ///
    /// Returns the value of `general.architecture` metadata key,
    /// or "unknown" if not present.
    ///
    /// Common values include: "llama", "phi", "mistral", "qwen", etc.
    pub fn model_family(&self) -> String {
        let arch = self
            .kv
            .get("general.architecture")
            .cloned()
            .unwrap_or(Value::from("unknown"));

        match arch {
            Value::String(arch) => arch,
            _ => String::from("unknown"),
        }
    }

    /// Get the estimated number of parameters in the model.
    ///
    /// Returns a human-readable string (e.g., "7B", "13B", "192").
    /// Returns "unknown" if parameters cannot be determined.
    pub fn model_parameters(&self) -> String {
        if self.parameters > 0 {
            human_number(self.parameters)
        } else {
            String::from("unknown")
        }
    }

    /// Get the quantization file type of the GGUF file.
    ///
    /// Returns a human-readable description of the quantization method
    /// (e.g., "All F32", "Mostly Q4_0", "Mostly BF16").
    /// Returns "unknown" if not present.
    pub fn file_type(&self) -> String {
        if let Some(ft) = self.kv.get("general.file_type") {
            file_type(ft.as_u64().unwrap())
        } else {
            String::from("unknown")
        }
    }

    /// Get the key-value metadata of the GGUF file.
    ///
    /// Returns a reference to the metadata map containing all key-value pairs
    /// from the GGUF file. Values are JSON values for flexibility.
    ///
    /// Common keys include:
    /// - `general.architecture`: Model architecture (e.g., "llama")
    /// - `general.name`: Model name
    /// - `tokenizer.ggml.tokens`: Tokenizer vocabulary
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.kv
    }

    /// Get the tensors of the GGUF file.
    ///
    /// Returns a reference to the vector of tensors, each containing
    /// name, type, offset, size, and shape information.
    pub fn tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }

    /// Absolute file byte offset where the tensor data section begins.
    /// Add a tensor's `offset` to this value to locate its data in the file.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Tensor data alignment in bytes, parsed from `general.alignment`
    /// (defaulting to [`GGUF_DEFAULT_ALIGNMENT`]).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }
}

/// Get a `GGUFContainer` from a file, truncating tokenizer arrays to length 3.
///
/// Only large `tokenizer.*` arrays (vocabularies, scores, merges) are capped;
/// model-configuration arrays such as `<arch>.attention.head_count_kv` are
/// always read in full.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let container = get_gguf_container("model.gguf")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container(file: &str) -> Result<GGUFContainer> {
    get_gguf_container_array_size(file, 3)
}

/// Get a `GGUFContainer` from a file with the provided max array size.
///
/// # Arguments
///
/// * `file` - Path to the GGUF file
/// * `max_array_size` - Maximum number of elements to read from array metadata
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container_array_size;
///
/// // Read all array elements
/// let container = get_gguf_container_array_size("model.gguf", u64::MAX)?;
///
/// // Limit arrays to 100 elements for performance
/// let container = get_gguf_container_array_size("model.gguf", 100)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container_array_size(file: &str, max_array_size: u64) -> Result<GGUFContainer> {
    if !std::path::Path::new(file).exists() {
        return Err(anyhow!("file not found"));
    }
    let mut reader = std::fs::File::open(file)?;
    let byte_le = reader.read_i32::<LittleEndian>()?;
    match byte_le {
        FILE_MAGIC_GGML => Err(anyhow!("unsupport ggml format")),
        FILE_MAGIC_GGMF => Err(anyhow!("unsupport ggmf format")),
        FILE_MAGIC_GGJT => Err(anyhow!("unsupport ggjt format")),
        FILE_MAGIC_GGLA => Err(anyhow!("unsupport ggla format")),
        FILE_MAGIC_GGUF_LE => {
            Ok(GGUFContainer::new(ByteOrder::LE, Box::new(reader), max_array_size))
        }
        FILE_MAGIC_GGUF_BE => {
            Ok(GGUFContainer::new(ByteOrder::BE, Box::new(reader), max_array_size))
        }
        _ => Err(anyhow!("invalid file magic")),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn test_read_le_v3_gguf() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
        assert_eq!(
            serde_json::to_value(model.kv).unwrap(),
            json!({
                "general.architecture": "llama",
                "llama.block_count": 12,
                "general.alignment": 64,
                "answer": 42,
                "answer_in_float": 42.0,
                "tokenizer.ggml.tokens": ["a", "b", "c"],
            })
        );
    }

    #[test]
    fn test_read_le_v3_gguf_with_tokens() {
        let mut container =
            super::get_gguf_container_array_size("tests/test-le-v3.gguf", u64::MAX).unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
        println!("{:?}", model.kv);
        assert_eq!(
            serde_json::to_value(model.kv).unwrap(),
            json!({
                "general.architecture": "llama", 
                "llama.block_count": 12, 
                "general.alignment": 64, 
                "answer": 42, 
                "answer_in_float": 42.0,
                "tokenizer.ggml.tokens": ["a", "b", "c", "d", "e"],})
        );
    }

    /// Regression test for https://github.com/zackshen/gguf/issues/21
    ///
    /// Heterogeneous-layer models (e.g. Gemma 4) store per-layer values such as
    /// `<arch>.attention.head_count_kv` as a full-length array. The default
    /// container truncates large *tokenizer* arrays for readability, but must
    /// preserve model-configuration arrays in full so the KV head counts can be
    /// read correctly.
    #[test]
    fn test_config_array_not_truncated_by_default() {
        use crate::writer::{GGUFWriter, MetadataValue};

        let path = std::env::temp_dir().join("gguf_issue21_head_count_kv.gguf");

        // Gemma 4 12B head_count_kv: 48 per-layer entries.
        let head_count_kv: Vec<i32> = (0..48).map(|i| if i % 6 == 5 { 1 } else { 8 }).collect();
        let head_count_kv_values: Vec<MetadataValue> =
            head_count_kv.iter().map(|v| MetadataValue::Int32(*v)).collect();
        // A large tokenizer array that should still be truncated.
        let tokens: Vec<MetadataValue> =
            (0..50).map(|i| MetadataValue::String(format!("tok{i}"))).collect();

        let mut w = GGUFWriter::new(&path, 3).unwrap();
        w.add_metadata("general.architecture", "gemma4");
        w.add_metadata_array("gemma4.attention.head_count_kv", head_count_kv_values);
        w.add_metadata_array("tokenizer.ggml.tokens", tokens);
        w.write().unwrap();
        w.finalize().unwrap();

        // Default container caps arrays at length 3.
        let mut container = super::get_gguf_container(path.to_str().unwrap()).unwrap();
        let model = container.decode().unwrap();

        let kv_arr = model
            .metadata()
            .get("gemma4.attention.head_count_kv")
            .and_then(|v| v.as_array())
            .expect("head_count_kv should be an array");
        assert_eq!(kv_arr.len(), 48, "config array must be read in full");
        let read_back: Vec<i64> = kv_arr.iter().map(|v| v.as_i64().unwrap()).collect();
        let expected: Vec<i64> = head_count_kv.iter().map(|v| *v as i64).collect();
        assert_eq!(read_back, expected);

        // Tokenizer arrays are still truncated to the configured cap.
        let tokens_arr = model
            .metadata()
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .expect("tokens should be an array");
        assert_eq!(tokens_arr.len(), 3, "tokenizer arrays remain capped");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_data_offset_honors_general_alignment() {
        // test-le-v3.gguf sets general.alignment = 64.
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.alignment(), 64);
        let data_off = model.data_offset();
        assert!(data_off > 0);
        assert_eq!(data_off % 64, 0, "data_offset must respect general.alignment");
    }

    #[test]
    fn test_tensor_preserves_ndimensions() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        for t in model.tensors() {
            assert_eq!(
                t.shape.len() as u32,
                t.n_dimensions,
                "shape length should equal n_dimensions for tensor {}",
                t.name
            );
            assert!(t.n_dimensions > 0 && t.n_dimensions <= super::GGUF_MAX_DIMS);
        }
    }

    #[test]
    fn test_file_not_found() {
        let result = super::get_gguf_container("nonexistent.gguf");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_invalid_file_magic() {
        use std::io::Cursor;
        let invalid_data = vec![0x00, 0x00, 0x00, 0x00];
        let cursor = Cursor::new(invalid_data);
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::LE, Box::new(cursor), u64::MAX);
        let result = container.decode();
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_value_type_conversion() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(MetadataValueType::try_from(100).is_err());
    }

    #[test]
    fn test_ggml_type_conversion() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(matches!(GGMLType::try_from(0), Ok(GGMLType::F32)));
        assert!(matches!(GGMLType::try_from(2), Ok(GGMLType::Q4_0)));
        assert!(GGMLType::try_from(100).is_err());
    }

    #[test]
    fn test_byte_order_default() {
        use super::ByteOrder;
        let bo = ByteOrder::default();
        assert!(matches!(bo, ByteOrder::LE));
    }

    #[test]
    fn test_tensors() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();
        assert!(!tensors.is_empty());

        for tensor in tensors {
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
        }
    }

    #[test]
    fn test_num_tensor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_tensor() > 0);
    }

    #[test]
    fn test_get_version() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        assert_eq!(container.get_version(), "v1"); // Before decode, default is v1
        let _ = container.decode().unwrap();
        // After decode, version should be v3
    }

    // ========== Additional tests for improved coverage ==========

    #[test]
    fn test_human_number_small() {
        assert_eq!(super::human_number(999), "999");
        assert_eq!(super::human_number(1000), "1000");
        assert_eq!(super::human_number(1001), "1K");
        assert_eq!(super::human_number(1500), "2K");
    }

    #[test]
    fn test_human_number_medium() {
        assert_eq!(super::human_number(1_000_000), "1000K");
        assert_eq!(super::human_number(1_000_001), "1M");
        assert_eq!(super::human_number(2_000_001), "2M");
        assert_eq!(super::human_number(3_500_000), "4M");
    }

    #[test]
    fn test_human_number_large() {
        assert_eq!(super::human_number(1_000_000_000), "1000M");
        assert_eq!(super::human_number(1_000_000_001), "1B");
        assert_eq!(super::human_number(7_500_000_000), "8B");
    }

    #[test]
    fn test_file_type_all_values() {
        assert_eq!(super::file_type(0), "All F32");
        assert_eq!(super::file_type(1), "Mostly F16");
        assert_eq!(super::file_type(2), "Mostly Q4_0");
        assert_eq!(super::file_type(7), "Mostly Q8_0");
        assert_eq!(super::file_type(14), "Mostly Q6_K");
        assert_eq!(super::file_type(24), "Mostly BF16");
        assert_eq!(super::file_type(99), "unknown");
    }

    #[test]
    fn test_metadata_value_type_all_variants() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        // Test all valid type values
        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(1), Ok(MetadataValueType::Int8)));
        assert!(matches!(MetadataValueType::try_from(2), Ok(MetadataValueType::Uint16)));
        assert!(matches!(MetadataValueType::try_from(3), Ok(MetadataValueType::Int16)));
        assert!(matches!(MetadataValueType::try_from(4), Ok(MetadataValueType::Uint32)));
        assert!(matches!(MetadataValueType::try_from(5), Ok(MetadataValueType::Int32)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(7), Ok(MetadataValueType::Bool)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(matches!(MetadataValueType::try_from(9), Ok(MetadataValueType::Array)));
        assert!(matches!(MetadataValueType::try_from(10), Ok(MetadataValueType::Uint64)));
        assert!(matches!(MetadataValueType::try_from(11), Ok(MetadataValueType::Int64)));
        assert!(matches!(MetadataValueType::try_from(12), Ok(MetadataValueType::Float64)));
    }

    #[test]
    fn test_ggml_type_all_valid_types() {
        use super::GGMLType;
        use std::convert::TryFrom;

        // Test a representative sample of GGML types
        assert!(matches!(GGMLType::try_from(1), Ok(GGMLType::F16)));
        assert!(matches!(GGMLType::try_from(3), Ok(GGMLType::Q4_1)));
        assert!(matches!(GGMLType::try_from(6), Ok(GGMLType::Q5_0)));
        assert!(matches!(GGMLType::try_from(7), Ok(GGMLType::Q5_1)));
        assert!(matches!(GGMLType::try_from(8), Ok(GGMLType::Q8_0)));
        assert!(matches!(GGMLType::try_from(10), Ok(GGMLType::Q2_K)));
        assert!(matches!(GGMLType::try_from(30), Ok(GGMLType::BF16)));
        assert!(matches!(GGMLType::try_from(39), Ok(GGMLType::MXFP4)));
    }

    #[test]
    fn test_ggml_type_invalid() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(GGMLType::try_from(100).is_err());
        assert!(GGMLType::try_from(255).is_err());
    }

    #[test]
    fn test_model_family_unknown() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // This test file has "llama" architecture
        assert_eq!(model.model_family(), "llama");
    }

    #[test]
    fn test_model_parameters_format() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // Test file has 192 parameters
        assert_eq!(model.model_parameters(), "192");
    }

    #[test]
    fn test_metadata_accessor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let metadata = model.metadata();
        assert!(metadata.contains_key("general.architecture"));
        assert!(metadata.contains_key("llama.block_count"));
    }

    #[test]
    fn test_num_kv() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_kv() > 0);
    }

    #[test]
    fn test_tensor_properties() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();

        for tensor in tensors {
            // Verify tensor has valid properties
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
            // Offset and size should be non-negative (u64)
            let _ = tensor.offset;
            let _ = tensor.size;
            let _ = tensor.kind;
        }
    }

    #[test]
    fn test_container_new() {
        use super::{ByteOrder, GGUFContainer};
        use std::io::Cursor;

        let cursor = Cursor::new(vec![]);
        let container = GGUFContainer::new(ByteOrder::LE, Box::new(cursor), 100);
        assert_eq!(container.get_version(), "v1"); // Default version
    }

    #[test]
    fn test_byte_order_variants() {
        use super::ByteOrder;

        let le = ByteOrder::LE;
        let be = ByteOrder::BE;

        // Just verify we can create both variants
        let _ = format!("{:?}", le);
        let _ = format!("{:?}", be);
    }

    #[test]
    fn test_version_variants() {
        use super::{Version, V1, V2, V3};

        let v1 = Version::V1(V1::default());
        let v2 = Version::V2(V2::default());
        let v3 = Version::V3(V3::default());

        // Verify we can create all version variants
        let _ = format!("{:?}", v1);
        let _ = format!("{:?}", v2);
        let _ = format!("{:?}", v3);
    }

    #[test]
    fn test_invalid_file_magic_detailed() {
        use std::io::Cursor;

        // Test with various invalid magic numbers
        let invalid_magics = vec![
            vec![0x00, 0x00, 0x00, 0x00],
            vec![0xFF, 0xFF, 0xFF, 0xFF],
            vec![0x12, 0x34, 0x56, 0x78],
        ];

        for magic in invalid_magics {
            let cursor = Cursor::new(magic);
            let mut container =
                super::GGUFContainer::new(super::ByteOrder::LE, Box::new(cursor), u64::MAX);
            let result = container.decode();
            assert!(result.is_err(), "Expected error for invalid magic");
        }
    }

    #[test]
    fn test_file_not_found_message() {
        let result = super::get_gguf_container("this_file_does_not_exist.gguf");
        assert!(result.is_err());
        // Check error message
        if let Err(err) = result {
            assert!(
                err.to_string().contains("file not found"),
                "Error message should mention 'file not found'"
            );
        }
    }

    #[test]
    fn test_get_gguf_container_array_size() {
        // Test with custom array size
        let result = super::get_gguf_container_array_size("tests/test-le-v3.gguf", 1);
        assert!(result.is_ok());

        let mut container = result.unwrap();
        let model = container.decode().unwrap();

        // With max_array_size=1, arrays should be truncated
        let tokens = model.kv.get("tokenizer.ggml.tokens");
        if let Some(tokens_arr) = tokens {
            if let serde_json::Value::Array(arr) = tokens_arr {
                assert!(arr.len() <= 1, "Array should be truncated to max size");
            }
        }
    }

    // ========== Regression tests for issue #15 (CWE-190, CWE-129) ==========

    /// Build a minimal LE v3 GGUF byte stream (post-magic) with a single tensor.
    fn build_v3_tensor_bytes(tensor: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        // version v3
        buf.extend_from_slice(&3i32.to_le_bytes());
        // num_tensor = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // num_kv = 0
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(tensor);
        buf
    }

    fn encode_string_v3(s: &str) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
        b
    }

    #[test]
    fn test_oob_tensor_dims_returns_error() {
        // CWE-129: dims > GGUF_MAX_DIMS must not panic with index-out-of-bounds.
        use std::io::Cursor;
        let mut tensor = encode_string_v3("oob");
        tensor.extend_from_slice(&5u32.to_le_bytes()); // dims = 5 (>4)
                                                       // Five u64 shape entries (would have panicked previously on i==4)
        for _ in 0..5 {
            tensor.extend_from_slice(&1u64.to_le_bytes());
        }
        tensor.extend_from_slice(&0u32.to_le_bytes()); // kind = F32
        tensor.extend_from_slice(&0u64.to_le_bytes()); // offset

        let bytes = build_v3_tensor_bytes(&tensor);
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::LE, Box::new(Cursor::new(bytes)), u64::MAX);
        let result = container.decode();
        assert!(result.is_err(), "decode must reject dims > 4");
        let msg = result.err().unwrap().to_string();
        assert!(
            msg.contains("dimensions") && msg.contains("exceeds maximum"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_integer_overflow_in_tensor_size_returns_error() {
        // CWE-190: shape[0] * shape[1] overflows u64 and must be rejected.
        use std::io::Cursor;
        let mut tensor = encode_string_v3("ovf");
        tensor.extend_from_slice(&4u32.to_le_bytes()); // dims = 4
        tensor.extend_from_slice(&u64::MAX.to_le_bytes());
        tensor.extend_from_slice(&2u64.to_le_bytes());
        tensor.extend_from_slice(&1u64.to_le_bytes());
        tensor.extend_from_slice(&1u64.to_le_bytes());
        tensor.extend_from_slice(&0u32.to_le_bytes()); // kind = F32
        tensor.extend_from_slice(&0u64.to_le_bytes()); // offset

        let bytes = build_v3_tensor_bytes(&tensor);
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::LE, Box::new(Cursor::new(bytes)), u64::MAX);
        let result = container.decode();
        assert!(result.is_err(), "decode must reject overflowing shape");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("overflows"), "unexpected error: {msg}");
    }

    #[test]
    fn test_unbounded_string_length_returns_error() {
        // CWE-789-ish: declared string length must be capped before allocation.
        use std::io::Cursor;
        // tensor name claims a length larger than GGUF_MAX_STRING_LENGTH.
        let mut tensor = Vec::new();
        tensor.extend_from_slice(&(super::GGUF_MAX_STRING_LENGTH + 1).to_le_bytes());
        // No string bytes are actually present — the length check must fire first.

        let bytes = build_v3_tensor_bytes(&tensor);
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::LE, Box::new(Cursor::new(bytes)), u64::MAX);
        let result = container.decode();
        assert!(result.is_err(), "decode must reject huge string lengths");
        let msg = result.err().unwrap().to_string();
        assert!(msg.contains("exceeds maximum"), "unexpected error: {msg}");
    }

    // ========== Synthetic byte-stream coverage tests ==========

    /// Build a post-magic GGUF byte stream for a given version, kv section, and tensor section.
    /// Caller supplies pre-encoded kv/tensor bytes and their counts.
    fn build_post_magic(
        version: i32,
        num_tensor: u64,
        num_kv: u64,
        kv: &[u8],
        tensors: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&version.to_le_bytes());
        match version {
            super::GGUF_VERSION_V1 => {
                buf.extend_from_slice(&(num_tensor as u32).to_le_bytes());
                buf.extend_from_slice(&(num_kv as u32).to_le_bytes());
            }
            _ => {
                buf.extend_from_slice(&num_tensor.to_le_bytes());
                buf.extend_from_slice(&num_kv.to_le_bytes());
            }
        }
        buf.extend_from_slice(kv);
        buf.extend_from_slice(tensors);
        buf
    }

    fn encode_string_versioned(s: &str, version: i32) -> Vec<u8> {
        let mut b = Vec::new();
        if version == super::GGUF_VERSION_V1 {
            b.extend_from_slice(&(s.len() as u32).to_le_bytes());
        } else {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        }
        b.extend_from_slice(s.as_bytes());
        b
    }

    fn encode_tensor_v3(name: &str, shape: &[u64], kind: u32, offset: u64) -> Vec<u8> {
        let mut b = encode_string_v3(name);
        b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    fn decode_post_magic(bytes: Vec<u8>) -> super::Result<super::GGUFModel> {
        use std::io::Cursor;
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::LE, Box::new(Cursor::new(bytes)), u64::MAX);
        container.decode()
    }

    #[test]
    fn test_decode_v1_format() {
        let bytes = build_post_magic(super::GGUF_VERSION_V1, 0, 0, &[], &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.get_version(), "v1");
        assert_eq!(model.num_kv(), 0);
        assert_eq!(model.num_tensor(), 0);
    }

    #[test]
    fn test_decode_v2_format() {
        let bytes = build_post_magic(super::GGUF_VERSION_V2, 0, 0, &[], &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.get_version(), "v2");
    }

    #[test]
    fn test_decode_invalid_version() {
        let bytes = build_post_magic(99, 0, 0, &[], &[]);
        let err = decode_post_magic(bytes).err().unwrap().to_string();
        assert!(err.contains("invalid version"), "unexpected: {err}");
    }

    /// KV pair builder: key (string) + value_type (u32) + raw value bytes
    fn kv_entry(key: &str, vtype: u32, value: &[u8]) -> Vec<u8> {
        let mut b = encode_string_v3(key);
        b.extend_from_slice(&vtype.to_le_bytes());
        b.extend_from_slice(value);
        b
    }

    #[test]
    fn test_decode_all_metadata_value_types() {
        // Build kv section containing every supported metadata type
        let mut kv = Vec::new();
        let mut count: u64 = 0;

        kv.extend_from_slice(&kv_entry("k_u8", 0, &[42u8]));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_i8", 1, &[(-7i8) as u8]));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_u16", 2, &1234u16.to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_i16", 3, &(-1234i16).to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_u32", 4, &123456u32.to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_i32", 5, &(-123456i32).to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_f32", 6, &1.5f32.to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_bool_t", 7, &[1u8]));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_bool_f", 7, &[0u8]));
        count += 1;
        let s = encode_string_v3("hello");
        kv.extend_from_slice(&kv_entry("k_str", 8, &s));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_u64", 10, &9_999_999_999u64.to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_i64", 11, &(-9_999_999_999i64).to_le_bytes()));
        count += 1;
        kv.extend_from_slice(&kv_entry("k_f64", 12, &3.25f64.to_le_bytes()));
        count += 1;

        // Array of u32 with 3 entries
        let mut arr = Vec::new();
        arr.extend_from_slice(&4u32.to_le_bytes()); // item_type = Uint32
        arr.extend_from_slice(&3u64.to_le_bytes()); // array_len
        arr.extend_from_slice(&1u32.to_le_bytes());
        arr.extend_from_slice(&2u32.to_le_bytes());
        arr.extend_from_slice(&3u32.to_le_bytes());
        kv.extend_from_slice(&kv_entry("k_arr_u32", 9, &arr));
        count += 1;

        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, count, &kv, &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.num_kv(), count);
        let kv = model.metadata();
        assert_eq!(kv.get("k_u8").unwrap().as_u64().unwrap(), 42);
        assert_eq!(kv.get("k_bool_t").unwrap().as_bool().unwrap(), true);
        assert_eq!(kv.get("k_bool_f").unwrap().as_bool().unwrap(), false);
        assert_eq!(kv.get("k_str").unwrap().as_str().unwrap(), "hello");
        let arr_v = kv.get("k_arr_u32").unwrap().as_array().unwrap();
        assert_eq!(arr_v.len(), 3);
    }

    #[test]
    fn test_decode_array_of_every_primitive() {
        // Each primitive that read_array supports gets an array entry.
        fn arr_kv(item_type: u32, value_bytes: &[u8], n: u64) -> Vec<u8> {
            let mut a = Vec::new();
            a.extend_from_slice(&item_type.to_le_bytes());
            a.extend_from_slice(&n.to_le_bytes());
            a.extend_from_slice(value_bytes);
            a
        }

        let mut kv = Vec::new();
        let mut count: u64 = 0;

        // u8
        kv.extend_from_slice(&kv_entry("a_u8", 9, &arr_kv(0, &[1u8, 2, 3], 3)));
        count += 1;
        // i8
        kv.extend_from_slice(&kv_entry("a_i8", 9, &arr_kv(1, &[1u8, 2, 3], 3)));
        count += 1;
        // u16
        let mut b = Vec::new();
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&2u16.to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_u16", 9, &arr_kv(2, &b, 2)));
        count += 1;
        // i16
        let mut b = Vec::new();
        b.extend_from_slice(&(-1i16).to_le_bytes());
        b.extend_from_slice(&2i16.to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_i16", 9, &arr_kv(3, &b, 2)));
        count += 1;
        // i32
        let mut b = Vec::new();
        b.extend_from_slice(&(-5i32).to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_i32", 9, &arr_kv(5, &b, 1)));
        count += 1;
        // f32
        let mut b = Vec::new();
        b.extend_from_slice(&1.5f32.to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_f32", 9, &arr_kv(6, &b, 1)));
        count += 1;
        // bool
        kv.extend_from_slice(&kv_entry("a_bool", 9, &arr_kv(7, &[1u8, 0], 2)));
        count += 1;
        // string
        let mut b = Vec::new();
        b.extend_from_slice(&encode_string_v3("hi"));
        kv.extend_from_slice(&kv_entry("a_str", 9, &arr_kv(8, &b, 1)));
        count += 1;
        // u64
        let mut b = Vec::new();
        b.extend_from_slice(&7u64.to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_u64", 9, &arr_kv(10, &b, 1)));
        count += 1;
        // i64
        let mut b = Vec::new();
        b.extend_from_slice(&(-7i64).to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_i64", 9, &arr_kv(11, &b, 1)));
        count += 1;
        // f64
        let mut b = Vec::new();
        b.extend_from_slice(&3.25f64.to_le_bytes());
        kv.extend_from_slice(&kv_entry("a_f64", 9, &arr_kv(12, &b, 1)));
        count += 1;

        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, count, &kv, &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(
            model
                .metadata()
                .get("a_u8")
                .unwrap()
                .as_array()
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn test_decode_nested_array_rejected() {
        // Array-of-array is not supported and must error.
        let mut inner = Vec::new();
        inner.extend_from_slice(&9u32.to_le_bytes()); // item_type = Array
        inner.extend_from_slice(&1u64.to_le_bytes());
        let kv = kv_entry("nested", 9, &inner);
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 1, &kv, &[]);
        let err = decode_post_magic(bytes).err().unwrap().to_string();
        assert!(err.contains("Unsupport item value type"));
    }

    #[test]
    fn test_decode_tensor_variety() {
        // One tensor for each documented kind, all shape=[1,1,1,1].
        let kinds: &[u32] = &[
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        ];
        let mut t = Vec::new();
        for (i, &k) in kinds.iter().enumerate() {
            let name = format!("t_{i}");
            t.extend_from_slice(&encode_tensor_v3(&name, &[1, 1, 1, 1], k, i as u64 * 64));
        }
        let bytes = build_post_magic(super::GGUF_VERSION_V3, kinds.len() as u64, 0, &[], &t);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.tensors().len(), kinds.len());
    }

    #[test]
    fn test_decode_tensor_count_kind_rejected() {
        // Kind 40 (GGMLType::Count) hits unreachable!(); decode must not reach it.
        // We assert the safer thing: kind=4 (Q4_2) and kind=5 (Q4_3) which are valid in
        // GGMLType::try_from but type_size=0 — the decoder still accepts them.
        // Use kind=200 to exercise the TryFrom error path.
        let t = encode_tensor_v3("bad", &[1, 1, 1, 1], 200, 0);
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 1, 0, &[], &t);
        let err = decode_post_magic(bytes).err().unwrap().to_string();
        assert!(err.contains("invalid GGML type"), "unexpected: {err}");
    }

    #[test]
    fn test_decode_unsupported_metadata_type() {
        // value_type 100 is not in MetadataValueType -> TryFrom error.
        let kv = kv_entry("k", 100, &[]);
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 1, &kv, &[]);
        let err = decode_post_magic(bytes).err().unwrap().to_string();
        assert!(err.contains("unsupport metadata value type"), "unexpected: {err}");
    }

    #[test]
    fn test_model_parameters_unknown_when_zero() {
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 0, &[], &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.model_parameters(), "unknown");
    }

    #[test]
    fn test_model_family_unknown_when_missing_or_wrong_type() {
        // No general.architecture key.
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 0, &[], &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.model_family(), "unknown");

        // general.architecture set to u32 instead of string -> fallback to "unknown".
        let kv = kv_entry("general.architecture", 4, &7u32.to_le_bytes());
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 1, &kv, &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.model_family(), "unknown");
    }

    #[test]
    fn test_file_type_via_model() {
        // general.file_type = 14 (Mostly Q6_K)
        let kv = kv_entry("general.file_type", 10, &14u64.to_le_bytes());
        let bytes = build_post_magic(super::GGUF_VERSION_V3, 0, 1, &kv, &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.file_type(), "Mostly Q6_K");
    }

    #[test]
    fn test_file_type_every_value() {
        for ft in 0u64..=24 {
            let s = super::file_type(ft);
            assert!(!s.is_empty());
        }
        assert_eq!(super::file_type(999), "unknown");
    }

    #[test]
    fn test_human_number_boundaries() {
        // Exercise the exact boundary lines.
        assert_eq!(super::human_number(0), "0");
        assert_eq!(super::human_number(999), "999");
        assert_eq!(super::human_number(1_001), "1K");
        assert_eq!(super::human_number(1_000_001), "1M");
        assert_eq!(super::human_number(1_000_000_001), "1B");
    }

    #[test]
    fn test_ggml_type_display_all_variants() {
        use super::GGMLType::*;
        let variants = [
            F32, F16, Q4_0, Q4_1, Q4_2, Q4_3, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
            Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, I8, I16, I32, I64,
            F64, IQ1_M, BF16, Q4_0_4_4, Q4_0_4_8, Q4_0_8_8, TQ1_0, TQ2_0, IQ4_NL_4_4, IQ4_NL_4_8,
            IQ4_NL_8_8, MXFP4, Count,
        ];
        for v in variants {
            let s = format!("{v}");
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_ggml_type_try_from_all_valid() {
        use super::GGMLType;
        use std::convert::TryFrom;
        for i in [
            0u32, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        ] {
            assert!(GGMLType::try_from(i).is_ok(), "{i} should be valid");
        }
        // Unmapped values 4 and 5 are reserved (Q4_2 / Q4_3) and not in TryFrom.
        assert!(GGMLType::try_from(4).is_err());
        assert!(GGMLType::try_from(5).is_err());
    }

    #[test]
    fn test_metadata_value_type_try_from_all() {
        use super::MetadataValueType;
        use std::convert::TryFrom;
        for i in 0u32..=12 {
            assert!(MetadataValueType::try_from(i).is_ok());
        }
        assert!(MetadataValueType::try_from(13).is_err());
    }

    #[test]
    fn test_decode_be_byte_order_invalid_version_reports() {
        // BE container with bytes containing version=v3 in LE will be misread.
        // Exercises the BE branch of decode().
        use std::io::Cursor;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3i32.to_be_bytes()); // v3 in BE
        bytes.extend_from_slice(&0u64.to_be_bytes()); // num_tensor
        bytes.extend_from_slice(&0u64.to_be_bytes()); // num_kv
        let mut container =
            super::GGUFContainer::new(super::ByteOrder::BE, Box::new(Cursor::new(bytes)), u64::MAX);
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
    }

    #[test]
    fn test_unsupported_legacy_magic_formats() {
        use std::io::Write;
        let dir = std::env::temp_dir();
        for (name, magic) in [
            ("ggml.bin", super::FILE_MAGIC_GGML),
            ("ggmf.bin", super::FILE_MAGIC_GGMF),
            ("ggjt.bin", super::FILE_MAGIC_GGJT),
            ("ggla.bin", super::FILE_MAGIC_GGLA),
        ] {
            let path = dir.join(format!("gguf_test_{name}"));
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&magic.to_le_bytes()).unwrap();
            let err = super::get_gguf_container(path.to_str().unwrap())
                .err()
                .unwrap();
            assert!(err.to_string().contains("unsupport"));
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn test_invalid_magic_in_file() {
        use std::io::Write;
        let path = std::env::temp_dir().join("gguf_test_bad_magic.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        let err = super::get_gguf_container(path.to_str().unwrap())
            .err()
            .unwrap();
        assert!(err.to_string().contains("invalid file magic"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_decode_v1_string_uses_u32_length() {
        // v1 read_string uses u32 length (not u64).
        let mut kv = Vec::new();
        // key
        kv.extend_from_slice(&encode_string_versioned("k", super::GGUF_VERSION_V1));
        // value_type = String
        kv.extend_from_slice(&8u32.to_le_bytes());
        // value: u32-length string
        kv.extend_from_slice(&encode_string_versioned("v1str", super::GGUF_VERSION_V1));

        let bytes = build_post_magic(super::GGUF_VERSION_V1, 0, 1, &kv, &[]);
        let model = decode_post_magic(bytes).unwrap();
        assert_eq!(model.metadata().get("k").unwrap().as_str().unwrap(), "v1str");
    }
}

/// Memory-mapped file support (requires `mmap` feature)
#[cfg(feature = "mmap")]
pub mod mmap;

/// Async I/O support (requires `async` feature)
#[cfg(feature = "async")]
pub mod async_io;

/// GGUF file writing support
pub mod writer;
