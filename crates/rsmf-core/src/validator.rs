//! Structural and checksum validation of an open [`crate::reader::RsmfFile`].
//!
//! The reader already performs structural validation at `open` time. This
//! module exposes the [`Validator`] API mandated by the skill so callers can
//! re-run each pass independently and so the CLI `verify` subcommand has a
//! canonical entry point.

use crate::error::Result;
use crate::reader::RsmfFile;

/// Static validator. Thin wrapper around [`RsmfFile`]'s own checks.
pub struct Validator;

impl Validator {
    /// Run the quick structural pass. Since [`RsmfFile::open`] already runs
    /// this, this function is a no-op on a correctly-opened file and returns
    /// `Ok(())`.
    ///
    /// It is still useful as a separate entry point so that:
    /// * callers who hold a long-lived `RsmfFile` can assert it is well-formed
    ///   without re-opening;
    /// * the CLI `verify` subcommand can log an explicit structural step.
    pub fn structural(_file: &RsmfFile) -> Result<()> {
        Ok(())
    }

    /// Run the full checksum pass. Delegates to [`RsmfFile::full_verify`].
    pub fn full(file: &RsmfFile) -> Result<()> {
        file.full_verify()
    }
}
