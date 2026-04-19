use rsmf_core::LogicalDtype;
use rsmf_core::RsmfFile;
use rsmf_core::writer::{GraphInput, RsmfWriter, TensorInput, VariantInput};
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("--- RSMF Classification Model Comparison ---\n");

    let st_path = "mobilenet_v2.safetensors";
    let rsmf_path = "mobilenet_v2_optimized.rsmf";
    let graph_path = "mobilenet_v2.onnx";

    // 1. Create a synthetic "MobileNet" model in Safetensors format
    // We'll create 10MB of weights to make the comparison meaningful.
    let weight_data = vec![0.5f32; 2_500_000]; // 10MB of FP32 data
    let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();

    // Write out the "Original" Safetensors (simulated by writing raw bytes)
    fs::write(st_path, &weight_bytes)?;
    let st_size = fs::metadata(st_path)?.len();

    // 2. Create a "Graph" file
    fs::write(graph_path, b"FAKE-ONNX-GRAPH-PAYLOAD")?;

    // 3. Convert to RSMF with INT8 Quantization and Compression
    println!("Step 1: Packing model to RSMF with INT8 + Compression...");
    let rsmf_writer = RsmfWriter::new()
        .with_metadata("framework", "rsmf-demo")
        .with_tensor_auto_q8_0(
            TensorInput {
                shard_id: 0,
                name: "backbone.weight".into(),
                dtype: LogicalDtype::F32,
                shape: vec![1, 2_500_000],
                metadata: vec![],
                canonical: VariantInput::canonical_raw(weight_bytes),
                packed: vec![],
            },
            rsmf_core::TargetTag::Wgpu,
        )
        .with_graph(GraphInput::onnx(b"FAKE-ONNX-GRAPH-PAYLOAD".to_vec()))
        .with_canonical_compression(3) // Zstd level 3
        .with_packed_compression(3);

    rsmf_writer.write_to_path(rsmf_path)?;
    let rsmf_size = fs::metadata(rsmf_path)?.len();

    // 4. Analysis
    println!("Step 2: Analyzing results...");
    let file = RsmfFile::open(rsmf_path)?;
    let summary = file.inspect();

    println!("\n| Metric             | Original (Safetensors) | Optimized (RSMF)      |");
    println!("|--------------------|-----------------------|-----------------------|");
    println!(
        "| File Size          | {:<21} | {:<21} |",
        format!("{} bytes", st_size),
        format!("{} bytes", rsmf_size)
    );
    println!("| Backend Support    | CPU Only              | CPU + WGPU (INT8)     |");
    println!("| Self-Contained?    | No (Weights Only)     | Yes (Weights+Graph)   |");
    println!("| Variants           | 1 (FP32)              | 2 (FP32 + INT8)       |");
    println!(
        "| Graph Type         | External              | {}                  |",
        summary
            .graph_kinds
            .first()
            .map(|k| k.name())
            .unwrap_or("None")
    );

    let compression_ratio = (st_size as f64) / (rsmf_size as f64);
    println!(
        "\nConclusion: RSMF artifact is {:.2}x more efficient for distribution,",
        compression_ratio
    );
    println!("while providing a hardware-optimized INT8 path and a bundled execution graph.");

    // Cleanup
    let _ = fs::remove_file(st_path);
    let _ = fs::remove_file(rsmf_path);
    let _ = fs::remove_file(graph_path);

    Ok(())
}
