use ndarray::Array2;
use rsmf_core::RsmfFile;
use rsmf_runtime::Engine;

fn main() -> anyhow::Result<()> {
    // 1. Open the full artifact
    let file = RsmfFile::open("minilm_full.rsmf")?;

    // 2. Initialize inference engine
    println!("Initializing engine...");
    let engine = Engine::new(file)?;
    let mut session = engine.session(0)?;

    // 3. Prepare inputs (batch_size=1, sequence_length=8)
    println!("Preparing inputs...");
    let shape = (1, 8);
    let input_ids = Array2::<i64>::zeros(shape);
    let attention_mask = Array2::<i64>::ones(shape);
    let token_type_ids = Array2::<i64>::zeros(shape);

    let v_input_ids = ort::value::Value::from_array(input_ids)?;
    let v_attention_mask = ort::value::Value::from_array(attention_mask)?;
    let v_token_type_ids = ort::value::Value::from_array(token_type_ids)?;

    let inputs = vec![
        ("input_ids", v_input_ids),
        ("attention_mask", v_attention_mask),
        ("token_type_ids", v_token_type_ids),
    ];

    // 4. Run inference
    println!("Running inference on graph 0...");
    let outputs = session.run(inputs)?;

    println!("Inference successful!");
    for (name, value) in outputs.iter() {
        if let Ok((shape, _)) = value.try_extract_tensor::<f32>() {
            println!("Output '{}' shape: {:?}", name, shape);
        } else {
            println!("Output '{}' present", name);
        }
    }

    Ok(())
}
