//! Crate-level error type.
//!
//! Public APIs return [`Result<T>`], which is a shorthand for
//! `std::result::Result<T, RsmfError>`. Internal modules may use
//! [`anyhow::Result`] for ergonomic aggregation, but surface typed errors at
//! crate boundaries.

use std::path::PathBuf;

/// Alias for the crate-wide `Result` type.
pub type Result<T> = std::result::Result<T, RsmfError>;

/// Every error returned from `rsmf-core` public APIs.
#[derive(Debug, thiserror::Error)]
pub enum RsmfError {
    /// An I/O failure from `std::io`.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The file does not exist, or a component inside it (tensor, asset) does
    /// not exist.
    #[error("not found: {what}")]
    NotFound {
        /// What was missing (for example a tensor or asset name, or a path).
        what: String,
    },

    /// The file's 8-byte magic does not match RSMF.
    #[error("invalid magic: expected {expected:02x?}, got {got:02x?}")]
    InvalidMagic {
        /// Magic that was expected.
        expected: [u8; 8],
        /// Magic that was read from the file.
        got: [u8; 8],
    },

    /// The format major version does not match what this crate supports.
    #[error("unsupported format version {major}.{minor}")]
    UnsupportedVersion {
        /// Major version as read from the preamble.
        major: u16,
        /// Minor version as read from the preamble.
        minor: u16,
    },

    /// Structural invariant violated: overlapping sections, bad offset, duplicate
    /// tensor names, reserved bits set, malformed manifest, etc.
    #[error("structural: {0}")]
    Structural(String),

    /// Section or variant checksum did not match the declared value.
    #[error("checksum mismatch: {kind}")]
    ChecksumMismatch {
        /// Human-readable description of what failed (e.g. `"section[2] CanonicalArena"`
        /// or `"variant tensor=weight target=Wgpu"`).
        kind: String,
    },

    /// The requested operation is not supported in this MVP (e.g. non-Raw
    /// decoding on a `tensor_view` call).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A safetensors source file could not be converted.
    #[error("safetensors conversion failed: {0}")]
    SafetensorsConversion(String),

    /// A GGUF source file could not be converted.
    #[error("GGUF conversion failed: {0}")]
    GgufConversion(String),

    /// A path-specific wrapper around an I/O error.
    #[error("i/o error for {path}: {source}")]
    IoWithPath {
        /// The filesystem path the operation was targeting.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
}

impl RsmfError {
    /// Helper for producing a [`RsmfError::Structural`] without a dedicated variant.
    #[must_use]
    pub fn structural(msg: impl Into<String>) -> Self {
        Self::Structural(msg.into())
    }

    /// Helper for producing a [`RsmfError::NotFound`].
    #[must_use]
    pub fn not_found(what: impl Into<String>) -> Self {
        Self::NotFound { what: what.into() }
    }

    /// Helper for producing a [`RsmfError::Unsupported`].
    #[must_use]
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }
}
