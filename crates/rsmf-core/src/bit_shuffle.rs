//! Bit-shuffling pre-processor for improved compression ratios.
//!
//! Based on the Bitshuffle algorithm (used in Blosc/HDF5), this transposes
//! the bit-matrix of a data block so that similar bits (e.g. sign bits,
//! exponents) are grouped together.

/// Shuffle bits in a byte slice using a fixed element size.
///
/// `elem_size` is typically 4 (for f32) or 2 (for f16).
pub fn shuffle(input: &[u8], elem_size: usize) -> Vec<u8> {
    if input.is_empty() || elem_size == 0 {
        return input.to_vec();
    }

    let n = input.len();
    let num_elems = n / elem_size;
    let mut output = vec![0u8; n];

    // Group bits by their position within the element.
    // For every bit position b in [0..elem_size*8):
    //   The b-th bits of all elements are concatenated.

    let total_bits = elem_size * 8;
    for b in 0..total_bits {
        let byte_idx = b / 8;
        let bit_mask = 1 << (b % 8);

        for i in 0..num_elems {
            let src_byte_idx = i * elem_size + byte_idx;
            let val = if src_byte_idx < n {
                input[src_byte_idx] & bit_mask
            } else {
                0
            };

            if val != 0 {
                let out_bit_idx = b * num_elems + i;
                let out_byte_idx = out_bit_idx / 8;
                let out_bit_mask = 1 << (out_bit_idx % 8);
                if out_byte_idx < n {
                    output[out_byte_idx] |= out_bit_mask;
                }
            }
        }
    }

    // Handle remaining bytes if input length is not a multiple of elem_size.
    let handled = num_elems * elem_size;
    if handled < n {
        output[handled..].copy_from_slice(&input[handled..]);
    }

    output
}

/// Unshuffle bits.
pub fn unshuffle(input: &[u8], elem_size: usize) -> Vec<u8> {
    if input.is_empty() || elem_size == 0 {
        return input.to_vec();
    }

    let n = input.len();
    let num_elems = n / elem_size;
    let mut output = vec![0u8; n];

    let total_bits = elem_size * 8;
    for b in 0..total_bits {
        for i in 0..num_elems {
            let in_bit_idx = b * num_elems + i;
            let in_byte_idx = in_bit_idx / 8;
            let in_bit_mask = 1 << (in_bit_idx % 8);

            let val = if in_byte_idx < n {
                input[in_byte_idx] & in_bit_mask
            } else {
                0
            };

            if val != 0 {
                let out_byte_idx = i * elem_size + (b / 8);
                let out_bit_mask = 1 << (b % 8);
                if out_byte_idx < n {
                    output[out_byte_idx] |= out_bit_mask;
                }
            }
        }
    }

    let handled = num_elems * elem_size;
    if handled < n {
        output[handled..].copy_from_slice(&input[handled..]);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_shuffle() {
        let data = [1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = shuffle(&bytes, 4);
        assert_ne!(bytes, shuffled);

        let unshuffled = unshuffle(&shuffled, 4);
        assert_eq!(bytes, unshuffled);
    }
}
