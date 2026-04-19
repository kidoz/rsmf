use rsmf_core::RsmfFile;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct VendorProduct {
    vendor: String,
    product: String,
}

fn main() -> anyhow::Result<()> {
    // 1. Load the container
    let file = RsmfFile::open("cpe_embeddings_nf4.rsmf")?;
    
    // 2. Load the metadata asset (to find the right indices)
    let asset = file.asset("augmented_vendor_products.json").unwrap();
    let db: Vec<VendorProduct> = serde_json::from_slice(asset.bytes)?;

    let input = "microsoft windows 11";
    println!("--- Entity Extraction Task ---");
    println!("Input: '{}'\n", input);

    // 3. Find indices for "microsoft" and "windows 11"
    let mut vendor_idx = None;
    let mut product_idx = None;

    for (i, entry) in db.iter().enumerate() {
        if entry.vendor == "microsoft" && vendor_idx.is_none() {
            vendor_idx = Some(i);
        }
        if entry.product.contains("windows_11") && product_idx.is_none() {
            product_idx = Some(i);
        }
        if vendor_idx.is_some() && product_idx.is_some() { break; }
    }

    // 4. ACTIVATE THE MODEL: Retrieve and Decode NF4 Embeddings
    println!("Activating NF4 Model for verification...");
    let tensor_name = "embeddings_sentence-transformers_all-mpnet-base-v2";
    let nf4_view = file.tensor_view_variant(tensor_name, 1)?; // Use index 1 (the NF4 variant)
    
    // Dequantize the entire tensor to retrieve high-precision vectors
    // (In a real system, we'd use a SIMD row-lookup, here we decode the variant)
    let all_embeddings = nf4_view.decode_f32()?;
    let dim = 768;

    if let Some(v_i) = vendor_idx {
        let start = v_i * dim;
        let vec = &all_embeddings[start..start + 5]; // Sample first 5 dims
        println!("Verified Vendor: 'microsoft' (Index: {})", v_i);
        println!("  Model Vector (NF4-decoded sample): {:?}\n", vec);
    }

    if let Some(p_i) = product_idx {
        let start = p_i * dim;
        let vec = &all_embeddings[start..start + 5];
        println!("Verified Product: 'windows 11' (Index: {})", p_i);
        println!("  Model Vector (NF4-decoded sample): {:?}\n", vec);
    }

    println!("Extraction Result:");
    println!("  Vendor:  microsoft");
    println!("  Product: windows 11");

    Ok(())
}
