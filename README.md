# GGUF

This is a Rust project that provides functionality for decoding and working with GGUF files. GGUF files are binary files that contain key-value metadata and tensors.

![Unit test](https://github.com/zackshen/gguf/actions/workflows/test.yml/badge.svg)
![Publish](https://github.com/zackshen/gguf/actions/workflows/publish.yml/badge.svg)

## Features

- Decoding GGUF files
- Accessing key-value metadata and tensors from GGUF files
- Support for different versions of GGUF files: v1, v2, v3

## Usage

```rust
use gguf_rs::get_gguf_container;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut container = get_gguf_container("path_to_your_file")?;
    let model = container.decode()?;
    println!("Model Family: {}", model.model_family());
    println!("Number of Parameters: {}", model.model_parameters());
    println!("File Type: {}", model.file_type());
    println!("Number of Tensors: {}", model.num_tensor());
    Ok(())
}
```

also you can install `gguf-rs` as a command line tool.

```bash
cargo install gguf-rs
```

show model file info

```bash
gguf path_to_your_file

Metadata:
+----+-----------------------------------+---------------+
| #  | Key                               | Value         |
+========================================================+
| 1  | general.architecture              | phi2          |
|----+-----------------------------------+---------------|
| 2  | general.file_type                 | 2             |
|----+-----------------------------------+---------------|
| 3  | general.name                      | Phi2          |
|----+-----------------------------------+---------------|
| 4  | general.quantization_version      | 2             |
|----+-----------------------------------+---------------|
| 5  | phi2.attention.head_count         | 32            |
|----+-----------------------------------+---------------|
| 6  | phi2.attention.head_count_kv      | 32            |
|----+-----------------------------------+---------------|
| 7  | phi2.attention.layer_norm_epsilon | 0.00001       |
|----+-----------------------------------+---------------|
| 8  | phi2.block_count                  | 32            |
|----+-----------------------------------+---------------|
| 9  | phi2.context_length               | 2048          |
|----+-----------------------------------+---------------|
| 10 | phi2.embedding_length             | 2560          |
|----+-----------------------------------+---------------|
| 11 | phi2.feed_forward_length          | 10240         |
|----+-----------------------------------+---------------|
| 12 | phi2.rope.dimension_count         | 32            |
|----+-----------------------------------+---------------|
| 13 | tokenizer.ggml.add_bos_token      | false         |
|----+-----------------------------------+---------------|
| 14 | tokenizer.ggml.bos_token_id       | 50256         |
|----+-----------------------------------+---------------|
| 15 | tokenizer.ggml.eos_token_id       | 50256         |
|----+-----------------------------------+---------------|
| 16 | tokenizer.ggml.merges             | [Ġ t,Ġ a,h e] |
|----+-----------------------------------+---------------|
| 17 | tokenizer.ggml.model              | gpt2          |
|----+-----------------------------------+---------------|
| 18 | tokenizer.ggml.token_type         | [1,1,1]       |
|----+-----------------------------------+---------------|
| 19 | tokenizer.ggml.tokens             | [!,",#]       |
|----+-----------------------------------+---------------|
| 20 | tokenizer.ggml.unknown_token_id   | 50256         |
+----+-----------------------------------+---------------+
```

show tensors

```bash
gguf path_to_your_file --tensors

Metadata:
+----+-----------------------------------+---------------+
| #  | Key                               | Value         |
+========================================================+
| 1  | general.architecture              | phi2          |
|----+-----------------------------------+---------------|
| 2  | general.file_type                 | 2             |
|----+-----------------------------------+---------------|
| 3  | general.name                      | Phi2          |
|----+-----------------------------------+---------------|
| 4  | general.quantization_version      | 2             |
|----+-----------------------------------+---------------|
| 5  | phi2.attention.head_count         | 32            |
|----+-----------------------------------+---------------|
| 6  | phi2.attention.head_count_kv      | 32            |
|----+-----------------------------------+---------------|
| 7  | phi2.attention.layer_norm_epsilon | 0.00001       |
|----+-----------------------------------+---------------|
| 8  | phi2.block_count                  | 32            |
|----+-----------------------------------+---------------|
| 9  | phi2.context_length               | 2048          |
|----+-----------------------------------+---------------|
| 10 | phi2.embedding_length             | 2560          |
|----+-----------------------------------+---------------|
| 11 | phi2.feed_forward_length          | 10240         |
|----+-----------------------------------+---------------|
| 12 | phi2.rope.dimension_count         | 32            |
|----+-----------------------------------+---------------|
| 13 | tokenizer.ggml.add_bos_token      | false         |
|----+-----------------------------------+---------------|
| 14 | tokenizer.ggml.bos_token_id       | 50256         |
|----+-----------------------------------+---------------|
| 15 | tokenizer.ggml.eos_token_id       | 50256         |
|----+-----------------------------------+---------------|
| 16 | tokenizer.ggml.merges             | [Ġ t,Ġ a,h e] |
|----+-----------------------------------+---------------|
| 17 | tokenizer.ggml.model              | gpt2          |
|----+-----------------------------------+---------------|
| 18 | tokenizer.ggml.token_type         | [1,1,1]       |
|----+-----------------------------------+---------------|
| 19 | tokenizer.ggml.tokens             | [!,",#]       |
|----+-----------------------------------+---------------|
| 20 | tokenizer.ggml.unknown_token_id   | 50256         |
+----+-----------------------------------+---------------+
Tensors:
+-----+---------------------------+------+----------------+------------+
| #   | Name                      | Type | Dimension      | Offset     |
+======================================================================+
| 1   | token_embd.weight         | Q4_0 | 2560,51200,1,1 | 0          |
|-----+---------------------------+------+----------------+------------|
| 2   | blk.0.attn_norm.bias      | F32  | 2560,1,1,1     | 73728000   |
|-----+---------------------------+------+----------------+------------|
| 3   | blk.0.attn_norm.weight    | F32  | 2560,1,1,1     | 73738240   |
|-----+---------------------------+------+----------------+------------|
| 4   | blk.0.attn_qkv.bias       | F32  | 7680,1,1,1     | 73748480   |
|-----+---------------------------+------+----------------+------------|
| 5   | blk.0.attn_qkv.weight     | Q4_0 | 2560,7680,1,1  | 73779200   |
|-----+---------------------------+------+----------------+------------|
| 6   | blk.0.attn_output.bias    | F32  | 2560,1,1,1     | 84838400   |
```

## Testing

This project includes unit tests. Run them with `cargo test`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## GGUF Specification

[GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)
