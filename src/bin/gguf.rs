use anyhow::{anyhow, Result};
use comfy_table::Table;
use gguf::{get_gguf_container, GGMLType, GGUFModel};
use log::{debug, LevelFilter};
use simple_logger::SimpleLogger;

fn print_metadata(model: &GGUFModel) {
    let mut table = Table::new();
    table.set_header(vec!["#", "Key", "Value"]);

    model
        .metadata()
        .iter()
        .enumerate()
        .for_each(|(i, (key, value))| {
            let unwrap_value = match value {
                serde_json::Value::Null => String::from("null"),
                serde_json::Value::Bool(v) => v.to_string(),
                serde_json::Value::Number(v) => v.to_string(),
                serde_json::Value::String(v) => v.to_owned(),
                serde_json::Value::Array(v) => {
                    let concat_values = v
                        .iter()
                        .map(|v| match v {
                            serde_json::Value::Null => String::from("null"),
                            serde_json::Value::Bool(v) => v.to_string(),
                            serde_json::Value::Number(v) => v.to_string(),
                            serde_json::Value::String(v) => v.to_string(),
                            serde_json::Value::Object(v) => serde_json::to_string(v).unwrap(),
                            _ => String::from("unsupport array type"),
                        })
                        .collect::<Vec<String>>()
                        .join(",");
                    format!("[{}]", concat_values)
                }
                serde_json::Value::Object(_) => todo!(),
            };
            table.add_row(vec![(i + 1).to_string(), key.clone(), unwrap_value]);
        });

    println!("Metadata:\n{table}");
}

fn print_tensors(model: &GGUFModel) {
    let mut table = Table::new();
    table.set_header(vec!["#", "Name", "Type", "Dimension", "Offset"]);

    model.tensors().iter().enumerate().for_each(|(i, tensor)| {
        table.add_row(vec![
            (i + 1).to_string(),
            tensor.name.clone(),
            GGMLType::from(tensor.kind).to_string(),
            tensor
                .shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
            tensor.offset.to_string(),
        ]);
    });

    println!("Tensors:\n{table}");
}

fn main() -> Result<()> {
    SimpleLogger::default()
        .with_level(LevelFilter::Debug)
        .init()
        .unwrap();

    let args = std::env::args().skip(1).collect::<Vec<String>>();
    let file = args.first().unwrap();

    let gguf_model = match get_gguf_container(file) {
        Ok(mut container) => container.decode(),
        Err(err) => return Err(anyhow!(err.to_string())),
    };

    debug!("gguf version {}", gguf_model.get_version());
    debug!("model family {}", gguf_model.model_family());
    debug!("model parameter {}", gguf_model.model_parameters());
    debug!("file type {}", gguf_model.file_type());

    print_metadata(&gguf_model);
    print_tensors(&gguf_model);
    Ok(())
}
