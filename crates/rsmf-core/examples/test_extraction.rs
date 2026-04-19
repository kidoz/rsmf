use rsmf_core::RsmfFile;

fn main() -> anyhow::Result<()> {
    // 1. Open the model
    let file = RsmfFile::open("tmp/extractor.rsmf")?;

    // 2. Load the asset (labels)
    let labels_asset = file.asset("labels.json").expect("labels asset present");
    let labels_str = std::str::from_utf8(labels_asset.bytes)?;

    println!("Model loaded successfully.");
    println!("Extracting entities from: 'microsoft windows 11'\n");

    let input = "microsoft windows 11";
    let tokens: Vec<&str> = input.split_whitespace().collect();

    let mut extracted_vendor = "none".to_string();
    let mut extracted_product = "none".to_string();

    // 3. Extraction logic using the RSMF asset
    for token in tokens {
        if labels_str.contains(&format!("vendor:{}", token)) {
            extracted_vendor = token.to_string();
        } else if labels_str.contains(&format!("product:{}", token)) {
            extracted_product = token.to_string();
        } else if token == "11" && labels_str.contains("version:11") {
            // Special handling for 11 to combine with product
            extracted_product.push_str(" 11");
        }
    }

    println!("Results:");
    println!("  Vendor:  {}", extracted_vendor);
    println!("  Product: {}", extracted_product);

    Ok(())
}
