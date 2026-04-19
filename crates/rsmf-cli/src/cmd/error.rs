//! CLI error type. Wraps [`anyhow::Error`] and carries an exit code so `main`
//! can translate failures to the right process status.

use std::fmt;

use rsmf_core::RsmfError;

use super::{EXIT_INTEGRITY, EXIT_USER_ERROR};

/// Error returned by every CLI subcommand.
#[derive(Debug)]
pub struct CliError {
    inner: anyhow::Error,
    exit: i32,
}

impl CliError {
    /// Construct a user-error (exit code `1`).
    #[must_use]
    pub fn user(err: impl Into<anyhow::Error>) -> Self {
        Self {
            inner: err.into(),
            exit: EXIT_USER_ERROR,
        }
    }

    /// Construct an integrity error (exit code `2`).
    #[must_use]
    pub fn integrity(err: impl Into<anyhow::Error>) -> Self {
        Self {
            inner: err.into(),
            exit: EXIT_INTEGRITY,
        }
    }

    /// Return the exit code this error should translate to.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        self.exit
    }
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#}", self.inner)
    }
}

impl From<anyhow::Error> for CliError {
    fn from(err: anyhow::Error) -> Self {
        Self::user(err)
    }
}

impl From<RsmfError> for CliError {
    fn from(err: RsmfError) -> Self {
        match err {
            RsmfError::ChecksumMismatch { .. } => Self::integrity(anyhow::Error::new(err)),
            other => Self::user(anyhow::Error::new(other)),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(err: std::io::Error) -> Self {
        Self::user(anyhow::Error::new(err))
    }
}
