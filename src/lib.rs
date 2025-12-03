/// This module provides functionality for decoding and working with GGUF files.
///
/// GGUF files are binary files that contain key-value metadata and tensors.
/// The `GGUFContainer` struct represents a GGUF file container and provides methods for decoding and accessing the data.
/// The `GGUFModel` struct represents the decoded GGUF data, including the key-value metadata and tensors.
/// The `Tensor` struct represents a tensor in the GGUF file, including its name, kind, offset, size, and shape.
/// The `ByteOrder` enum represents the byte order of the GGUF file (little endian or big endian).
/// The `Version` enum represents the version of the GGUF file (v1, v2, or v3).
/// The `MetadataValueType` enum represents the value type of the metadata in the GGUF file.
/// The `GGMLType` enum represents the GGML type of a tensor in the GGUF file.
///
/// Example usage:
/// ```
/// use gguf_rs::{get_gguf_container};
/// use std::fs::File;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let mut container = get_gguf_container("tests/test-le-v3.gguf")?;
///     let model = container.decode()?;
///
///     println!("GGUF version: {}", model.get_version());
///
///     for tensor in model.tensors() {
///         println!("Tensor name: {}", tensor.name);
///         println!("Tensor kind: {}", tensor.kind);
///         println!("Tensor shape: {:?}", tensor.shape);
///     }
///
///     Ok(())
/// }
/// ```
use anyhow::{Result, anyhow};
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

const GGUF_VERSION_V1: i32 = 0x00000001;
const GGUF_VERSION_V2: i32 = 0x00000002;
const GGUF_VERSION_V3: i32 = 0x00000003;

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

/// GGUF file container.
pub struct GGUFContainer {
    bo: ByteOrder,
    version: Version,
    reader: Box<dyn std::io::Read + 'static>,
    max_array_size: u64,
}

impl GGUFContainer {
    /// Create a new `GGUFContainer` from a byte order and a reader.
    /// The reader must implement the `std::io::Read` trait.
    /// ```
    /// use gguf_rs::{get_gguf_container};
    /// use std::fs::File;
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let mut container = get_gguf_container("tests/test-le-v3.gguf")?;
    ///     let model = container.decode()?;
    ///
    ///     println!("GGUF version: {}", model.get_version());
    ///
    ///     for tensor in model.tensors() {
    ///         println!("Tensor name: {}", tensor.name);
    ///         println!("Tensor kind: {}", tensor.kind);
    ///         println!("Tensor shape: {:?}", tensor.shape);
    ///     }
    ///
    ///     Ok(())
    /// }
    pub fn new(bo: ByteOrder, reader: Box<dyn std::io::Read>, max_array_size: u64) -> Self {
        Self {
            bo,
            version: Version::V1(V1::default()),
            reader,
            max_array_size,
        }
    }

    /// Get the version of the GGUF file.
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Decode the GGUF file and return a `GGUFModel`.
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
        };

        model.decode(&mut self.reader)?;
        Ok(model)
    }
}

/// Tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub kind: u32,
    pub offset: u64,
    pub size: u64,
    // shape is the number of elements in each dimension
    pub shape: Vec<u64>,
}

/// GGUF model.
pub struct GGUFModel {
    kv: BTreeMap<String, Value>,
    tensors: Vec<Tensor>,
    parameters: u64,
    max_array_size: u64,
    bo: ByteOrder,
    version: Version,
}

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
#[derive(Debug, Serialize)]
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
                MetadataValueType::Array => Value::from(self.read_array(&mut reader)?),
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
            let mut shape = [1; 4];
            for i in 0..dims {
                shape[i as usize] = self.read_u64(&mut reader)?;
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

            let parameters = shape[0] * shape[1] * shape[2] * shape[3];
            let size = parameters * type_size / block_size;

            self.tensors.push(Tensor {
                name,
                kind,
                offset,
                size,
                shape: shape.to_vec(),
            });

            self.parameters += parameters;
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
        let mut buffer = vec![0; name_len as usize];
        reader.read_exact(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    fn read_array(&self, mut reader: impl std::io::Read) -> Result<Vec<Value>> {
        let mut data = Vec::new();
        let item_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
        let array_len = self.read_version_size(&mut reader)?;
        let read_count: usize = u64::min(array_len, self.max_array_size) as usize;
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
            if read_count > 0 && data.len() < read_count {
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

    /// Get the version of the GGUF file.
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
    pub fn num_tensor(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_tensor as u64,
            Version::V2(v2) => v2.num_tensor,
            Version::V3(v3) => v3.num_tensor,
        }
    }

    /// Get the model family of the GGUF file.
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

    /// Get the number of parameters in the GGUF file.
    pub fn model_parameters(&self) -> String {
        if self.parameters > 0 {
            human_number(self.parameters)
        } else {
            String::from("unknown")
        }
    }

    /// Get the file type of the GGUF file.
    pub fn file_type(&self) -> String {
        if let Some(ft) = self.kv.get("general.file_type") {
            file_type(ft.as_u64().unwrap())
        } else {
            String::from("unknown")
        }
    }

    /// Get the key-value metadata of the GGUF file.
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.kv
    }

    /// Get the tensors of the GGUF file.
    pub fn tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }
}

/// Get a `GGUFContainer` from a file, truncating all arrays to length 3.
pub fn get_gguf_container(file: &str) -> Result<GGUFContainer> {
    get_gguf_container_array_size(file, 3)
}

/// Get a `GGUFContainer` from a file with the provided max array size.
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
        FILE_MAGIC_GGUF_LE => Ok(GGUFContainer::new(ByteOrder::LE, Box::new(reader), max_array_size)),
        FILE_MAGIC_GGUF_BE => Ok(GGUFContainer::new(ByteOrder::BE, Box::new(reader), max_array_size)),
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
        let mut container = super::get_gguf_container_array_size("tests/test-le-v3.gguf", u64::MAX).unwrap();
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
}
