#[cfg(feature = "async_io")]
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeek, AsyncSeekExt};

#[cfg(feature = "async_io")]
use crate::error::{Result, RsmfError};
#[cfg(feature = "async_io")]
use crate::manifest::Manifest;
#[cfg(feature = "async_io")]
use crate::preamble::{PREAMBLE_LEN, Preamble};
#[cfg(feature = "async_io")]
use crate::section::{SECTION_DESC_LEN, SectionDescriptor, SectionKind};

#[cfg(feature = "async_io")]
/// An asynchronous reader for RSMF files.
pub struct AsyncRsmfFile<R> {
    reader: R,
    preamble: Preamble,
    sections: Vec<SectionDescriptor>,
    manifest: Manifest,
}

#[cfg(feature = "async_io")]
impl<R: AsyncRead + AsyncSeek + Unpin> AsyncRsmfFile<R> {
    /// Open and validate an RSMF file asynchronously from a stream.
    pub async fn from_async_read(mut reader: R) -> Result<Self> {
        let mut preamble_bytes = [0u8; PREAMBLE_LEN as usize];
        reader
            .read_exact(&mut preamble_bytes)
            .await
            .map_err(|e| RsmfError::Io(e))?;

        let preamble = Preamble::decode(&preamble_bytes)?;

        if preamble.section_tbl_off != PREAMBLE_LEN {
            return Err(RsmfError::structural(format!(
                "preamble.section_tbl_off {} must equal preamble length {PREAMBLE_LEN}",
                preamble.section_tbl_off
            )));
        }

        let table_len = preamble
            .section_tbl_count
            .checked_mul(SECTION_DESC_LEN)
            .ok_or_else(|| RsmfError::structural("section table length overflow"))?;
        
        let mut table_bytes = vec![0u8; table_len as usize];
        reader
            .seek(std::io::SeekFrom::Start(preamble.section_tbl_off))
            .await
            .map_err(|e| RsmfError::Io(e))?;
        reader
            .read_exact(&mut table_bytes)
            .await
            .map_err(|e| RsmfError::Io(e))?;

        let mut sections = Vec::with_capacity(preamble.section_tbl_count as usize);
        for i in 0..preamble.section_tbl_count {
            let off = (i * SECTION_DESC_LEN) as usize;
            sections.push(SectionDescriptor::decode(
                &table_bytes[off..off + SECTION_DESC_LEN as usize],
            )?);
        }

        // We skip validate_section_table here since file_len might not be known

        let manifest_section_idx = sections
            .iter()
            .position(|s| s.kind == SectionKind::Manifest)
            .ok_or_else(|| RsmfError::structural("no manifest section".to_string()))?;
        let manifest_section = &sections[manifest_section_idx];

        let mut manifest_bytes = vec![0u8; manifest_section.length as usize];
        reader
            .seek(std::io::SeekFrom::Start(manifest_section.offset))
            .await
            .map_err(|e| RsmfError::Io(e))?;
        reader
            .read_exact(&mut manifest_bytes)
            .await
            .map_err(|e| RsmfError::Io(e))?;

        let manifest = Manifest::decode(&manifest_bytes)?;

        Ok(Self {
            reader,
            preamble,
            sections,
            manifest,
        })
    }

    /// Return the parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Return the parsed preamble.
    pub fn preamble(&self) -> &Preamble {
        &self.preamble
    }

    /// Return the parsed section table.
    pub fn sections(&self) -> &[SectionDescriptor] {
        &self.sections
    }

    /// Consume the file and return the inner reader.
    pub fn into_inner(self) -> R {
        self.reader
    }
}
