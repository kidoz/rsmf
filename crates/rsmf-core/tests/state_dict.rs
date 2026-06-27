mod common;

use std::io::Write;

use rsmf_core::{LogicalDtype, RsmfFile, StateDictSchema, StateDictValidationIssue, TensorSpec};
use tempfile::NamedTempFile;

#[test]
fn state_dict_lists_tensor_contracts_in_name_order() {
    let file = open_basic_file();
    let state = file.state_dict();

    assert_eq!(state.len(), 2);
    assert_eq!(state.names().collect::<Vec<_>>(), vec!["bias", "weight"]);
    assert!(state.contains_key("weight"));

    let weight = state.get("weight").unwrap();
    assert_eq!(weight.name, "weight");
    assert_eq!(weight.dtype, LogicalDtype::F32);
    assert_eq!(weight.shape, vec![4, 4]);
    assert_eq!(weight.packed_variants.len(), 1);
    assert_eq!(weight.shard_id, 0);
    assert_eq!(
        weight.metadata,
        vec![("layer".to_string(), "0".to_string())]
    );
}

#[test]
fn state_dict_strict_schema_reports_unexpected_keys() {
    let file = open_basic_file();
    let state = file.state_dict();
    let schema = StateDictSchema::strict(vec![TensorSpec::new(
        "weight",
        LogicalDtype::F32,
        vec![4, 4],
    )]);

    let report = state.validate(&schema);

    assert!(!report.is_valid());
    assert_eq!(report.unexpected_keys().collect::<Vec<_>>(), vec!["bias"]);
    assert_eq!(
        report.issues,
        vec![StateDictValidationIssue::UnexpectedKey {
            name: "bias".to_string()
        }]
    );
}

#[test]
fn state_dict_non_strict_schema_allows_unexpected_keys() {
    let file = open_basic_file();
    let state = file.state_dict();
    let schema = StateDictSchema::non_strict(vec![TensorSpec::new(
        "weight",
        LogicalDtype::F32,
        vec![4, 4],
    )]);

    let report = state.validate(&schema);

    assert!(report.is_valid());
}

#[test]
fn state_dict_schema_reports_missing_dtype_and_shape_issues() {
    let file = open_basic_file();
    let state = file.state_dict();
    let schema = StateDictSchema::non_strict(vec![
        TensorSpec::new("missing", LogicalDtype::F32, vec![1]),
        TensorSpec::new("weight", LogicalDtype::I32, vec![2, 8]),
    ]);

    let report = state.validate(&schema);

    assert_eq!(report.missing_keys().collect::<Vec<_>>(), vec!["missing"]);
    assert_eq!(
        report.issues,
        vec![
            StateDictValidationIssue::MissingKey {
                name: "missing".to_string()
            },
            StateDictValidationIssue::DtypeMismatch {
                name: "weight".to_string(),
                expected: LogicalDtype::I32,
                actual: LogicalDtype::F32,
            },
            StateDictValidationIssue::ShapeMismatch {
                name: "weight".to_string(),
                expected: vec![2, 8],
                actual: vec![4, 4],
            },
        ]
    );
}

#[test]
fn state_dict_schema_accepts_absent_optional_specs() {
    let file = open_basic_file();
    let state = file.state_dict();
    let schema = StateDictSchema::non_strict(vec![TensorSpec::optional(
        "optional.bias",
        LogicalDtype::F32,
        vec![4],
    )]);

    let report = state.validate(&schema);

    assert!(report.is_valid());
}

fn open_basic_file() -> RsmfFile {
    let bytes = common::build_basic_file_bytes();
    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    tmp.flush().unwrap();
    RsmfFile::open(tmp.path()).unwrap()
}
