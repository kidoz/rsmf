//! Routing helpers for host-side MoE dispatch.

/// Tokens grouped by destination expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingBatch {
    /// Expert id selected by the router.
    pub expert_id: u32,
    /// Input token row indices routed to this expert.
    pub token_indices: Vec<usize>,
}

/// Group token assignments by destination expert.
///
/// The output order is deterministic: first occurrence of each expert in the
/// assignment stream.
#[must_use]
pub fn batch_by_destination(assignments: &[u32]) -> Vec<RoutingBatch> {
    let mut out: Vec<RoutingBatch> = Vec::new();
    for (token_idx, &expert_id) in assignments.iter().enumerate() {
        if let Some(batch) = out.iter_mut().find(|batch| batch.expert_id == expert_id) {
            batch.token_indices.push(token_idx);
        } else {
            out.push(RoutingBatch {
                expert_id,
                token_indices: vec![token_idx],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_by_first_seen_destination() {
        let batches = batch_by_destination(&[1, 0, 1, 2, 0]);
        assert_eq!(
            batches,
            vec![
                RoutingBatch {
                    expert_id: 1,
                    token_indices: vec![0, 2],
                },
                RoutingBatch {
                    expert_id: 0,
                    token_indices: vec![1, 4],
                },
                RoutingBatch {
                    expert_id: 2,
                    token_indices: vec![3],
                },
            ]
        );
    }
}
