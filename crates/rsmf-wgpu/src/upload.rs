//! Chunked staging upload of a canonical tensor's bytes into a GPU buffer.
//!
//! The upload path uses [`wgpu::Queue::write_buffer`] which performs the
//! staging copy internally. For large tensors we still split into explicit
//! chunks so the GPU driver's staging buffer is not forced to allocate the
//! whole payload at once — this is the "chunked staging uploads" requirement
//! from the brief.

use crate::device::DeviceHandle;

/// Upload configuration.
#[derive(Debug, Clone, Copy)]
pub struct UploadOptions {
    /// Maximum chunk size, in bytes. Must be a multiple of
    /// [`wgpu::COPY_BUFFER_ALIGNMENT`] (4).
    pub chunk_bytes: u64,
    /// Buffer usage flags. Defaults to `STORAGE | COPY_DST`.
    pub usage: wgpu::BufferUsages,
    /// Optional label for the resulting buffer (for renderdoc / captures).
    pub label: Option<&'static str>,
}

impl Default for UploadOptions {
    fn default() -> Self {
        Self {
            chunk_bytes: 4 * 1024 * 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label: Some("rsmf-canonical-tensor"),
        }
    }
}

/// Reasons [`upload_canonical_tensor`] can fail. The device is trusted; the
/// only failure modes are user/config errors.
#[derive(Debug, thiserror::Error)]
pub enum UploadError {
    /// `UploadOptions::chunk_bytes` is not a multiple of 4.
    #[error("chunk_bytes ({chunk}) must be a multiple of {align}")]
    BadChunkAlignment {
        /// Chunk size that was given.
        chunk: u64,
        /// Required alignment (4).
        align: u64,
    },
    /// Empty tensor upload.
    #[error("cannot upload an empty tensor")]
    Empty,
}

/// Allocate a GPU buffer and upload `bytes` into it, using chunked writes
/// no larger than [`UploadOptions::chunk_bytes`] each.
pub fn upload_canonical_tensor(
    device: &DeviceHandle,
    bytes: &[u8],
    opts: &UploadOptions,
) -> Result<wgpu::Buffer, UploadError> {
    if bytes.is_empty() {
        return Err(UploadError::Empty);
    }
    if opts.chunk_bytes % wgpu::COPY_BUFFER_ALIGNMENT != 0 || opts.chunk_bytes == 0 {
        return Err(UploadError::BadChunkAlignment {
            chunk: opts.chunk_bytes,
            align: wgpu::COPY_BUFFER_ALIGNMENT,
        });
    }
    let padded_len = pad_up(bytes.len() as u64, wgpu::COPY_BUFFER_ALIGNMENT);
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: opts.label,
        size: padded_len,
        usage: opts.usage,
        mapped_at_creation: false,
    });

    let chunk = opts
        .chunk_bytes
        .min(padded_len)
        .max(wgpu::COPY_BUFFER_ALIGNMENT);
    let mut offset: u64 = 0;
    let total = bytes.len() as u64;
    while offset < total {
        let remaining = total - offset;
        let this_chunk = remaining.min(chunk);
        let end = offset + this_chunk;
        device
            .queue
            .write_buffer(&buffer, offset, &bytes[offset as usize..end as usize]);
        offset = end;
    }
    device.queue.submit(std::iter::empty());
    Ok(buffer)
}

fn pad_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        return value;
    }
    let rem = value % align;
    if rem == 0 {
        value
    } else {
        value + (align - rem)
    }
}
