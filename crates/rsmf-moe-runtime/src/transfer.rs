//! Internal transfer execution accounting for MoE placement devices.

use std::time::Duration;

#[cfg(any(feature = "wgpu", test))]
use crate::{MoeRuntimeError, Result};
use crate::{MoeTransferKind, TransferRunReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferBackend {
    CpuRam,
    #[cfg(any(feature = "wgpu", test))]
    Wgpu,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TransferExecutor {
    backend: TransferBackend,
}

impl TransferExecutor {
    pub(crate) const fn cpu_ram() -> Self {
        Self {
            backend: TransferBackend::CpuRam,
        }
    }

    #[cfg(any(feature = "wgpu", test))]
    pub(crate) const fn wgpu() -> Self {
        Self {
            backend: TransferBackend::Wgpu,
        }
    }

    pub(crate) fn idle_report(&self) -> TransferRunReport {
        TransferRunReport {
            kind: match self.backend {
                TransferBackend::CpuRam => MoeTransferKind::None,
                #[cfg(any(feature = "wgpu", test))]
                TransferBackend::Wgpu => MoeTransferKind::HostToDevice,
            },
            bytes: 0,
            duration: Duration::ZERO,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn execute(&self, event: TransferEvent) -> Result<TransferRunReport> {
        match (self.backend, event.kind) {
            (_, MoeTransferKind::Unsupported) => Err(unsupported("unsupported", event.kind)),
            (TransferBackend::CpuRam, MoeTransferKind::None)
            | (TransferBackend::Wgpu, MoeTransferKind::None) => Ok(TransferRunReport {
                kind: MoeTransferKind::None,
                bytes: 0,
                duration: Duration::ZERO,
                cache_hits: usize::from(event.cache_hit),
                cache_misses: usize::from(!event.cache_hit),
            }),
            (TransferBackend::CpuRam, kind) => Err(unsupported("CPU/RAM", kind)),
            (TransferBackend::Wgpu, MoeTransferKind::HostToDevice) => Ok(TransferRunReport {
                kind: MoeTransferKind::HostToDevice,
                bytes: if event.cache_hit { 0 } else { event.bytes },
                duration: if event.cache_hit {
                    Duration::ZERO
                } else {
                    event.duration
                },
                cache_hits: usize::from(event.cache_hit),
                cache_misses: usize::from(!event.cache_hit),
            }),
            (
                TransferBackend::Wgpu,
                kind @ (MoeTransferKind::DeviceToHost | MoeTransferKind::PeerToPeer),
            ) => Err(unsupported("WGPU", kind)),
        }
    }

    #[cfg(any(feature = "wgpu", test))]
    pub(crate) fn accumulator(&self) -> TransferAccumulator {
        TransferAccumulator::new(self.idle_report().kind)
    }
}

#[cfg(any(feature = "wgpu", test))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TransferEvent {
    kind: MoeTransferKind,
    bytes: usize,
    duration: Duration,
    cache_hit: bool,
}

#[cfg(any(feature = "wgpu", test))]
impl TransferEvent {
    pub(crate) const fn host_to_device(bytes: usize, duration: Duration, cache_hit: bool) -> Self {
        Self {
            kind: MoeTransferKind::HostToDevice,
            bytes,
            duration,
            cache_hit,
        }
    }

    #[cfg(test)]
    const fn device_to_host(bytes: usize, duration: Duration) -> Self {
        Self {
            kind: MoeTransferKind::DeviceToHost,
            bytes,
            duration,
            cache_hit: false,
        }
    }

    #[cfg(test)]
    const fn peer_to_peer(bytes: usize, duration: Duration) -> Self {
        Self {
            kind: MoeTransferKind::PeerToPeer,
            bytes,
            duration,
            cache_hit: false,
        }
    }
}

#[cfg(any(feature = "wgpu", test))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TransferAccumulator {
    kind: MoeTransferKind,
    bytes: usize,
    duration: Duration,
    cache_hits: usize,
    cache_misses: usize,
}

#[cfg(any(feature = "wgpu", test))]
impl TransferAccumulator {
    fn new(kind: MoeTransferKind) -> Self {
        Self {
            kind,
            bytes: 0,
            duration: Duration::ZERO,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub(crate) fn record(&mut self, report: &TransferRunReport) -> Result<()> {
        if report.kind != MoeTransferKind::None && report.kind != self.kind {
            return Err(MoeRuntimeError::Unsupported(format!(
                "cannot accumulate {:?} transfer into {:?} transfer report",
                report.kind, self.kind
            )));
        }
        self.bytes = self
            .bytes
            .checked_add(report.bytes)
            .ok_or_else(|| MoeRuntimeError::Shape("transfer byte count overflow".to_string()))?;
        self.duration = self
            .duration
            .checked_add(report.duration)
            .ok_or_else(|| MoeRuntimeError::Shape("transfer duration overflow".to_string()))?;
        self.cache_hits = self
            .cache_hits
            .checked_add(report.cache_hits)
            .ok_or_else(|| {
                MoeRuntimeError::Shape("transfer cache hit count overflow".to_string())
            })?;
        self.cache_misses = self
            .cache_misses
            .checked_add(report.cache_misses)
            .ok_or_else(|| {
                MoeRuntimeError::Shape("transfer cache miss count overflow".to_string())
            })?;
        Ok(())
    }

    pub(crate) fn finish(self) -> TransferRunReport {
        TransferRunReport {
            kind: self.kind,
            bytes: self.bytes,
            duration: self.duration,
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
        }
    }
}

#[cfg(any(feature = "wgpu", test))]
fn unsupported(backend: &str, kind: MoeTransferKind) -> MoeRuntimeError {
    MoeRuntimeError::Unsupported(format!(
        "{backend} transfer executor does not support {kind:?} transfers"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ram_only_allows_noop_transfers() {
        let executor = TransferExecutor::cpu_ram();

        let err = executor
            .execute(TransferEvent::host_to_device(
                16,
                Duration::from_micros(1),
                false,
            ))
            .unwrap_err();

        assert!(
            matches!(err, MoeRuntimeError::Unsupported(message) if message.contains("CPU/RAM") && message.contains("HostToDevice"))
        );
    }

    #[test]
    fn wgpu_reports_host_to_device_miss_and_hit() {
        let executor = TransferExecutor::wgpu();

        let miss = executor
            .execute(TransferEvent::host_to_device(
                32,
                Duration::from_micros(7),
                false,
            ))
            .unwrap();
        let hit = executor
            .execute(TransferEvent::host_to_device(
                32,
                Duration::from_micros(7),
                true,
            ))
            .unwrap();

        assert_eq!(miss.kind, MoeTransferKind::HostToDevice);
        assert_eq!(miss.bytes, 32);
        assert_eq!(miss.cache_misses, 1);
        assert_eq!(hit.bytes, 0);
        assert_eq!(hit.duration, Duration::ZERO);
        assert_eq!(hit.cache_hits, 1);
    }

    #[test]
    fn wgpu_rejects_device_to_host_and_peer_to_peer() {
        let executor = TransferExecutor::wgpu();

        let device_to_host = executor
            .execute(TransferEvent::device_to_host(16, Duration::ZERO))
            .unwrap_err();
        let peer_to_peer = executor
            .execute(TransferEvent::peer_to_peer(16, Duration::ZERO))
            .unwrap_err();

        assert!(
            matches!(device_to_host, MoeRuntimeError::Unsupported(message) if message.contains("DeviceToHost"))
        );
        assert!(
            matches!(peer_to_peer, MoeRuntimeError::Unsupported(message) if message.contains("PeerToPeer"))
        );
    }

    #[test]
    fn accumulator_sums_reports() {
        let executor = TransferExecutor::wgpu();
        let mut accumulator = executor.accumulator();
        let first = executor
            .execute(TransferEvent::host_to_device(
                32,
                Duration::from_micros(3),
                false,
            ))
            .unwrap();
        let second = executor
            .execute(TransferEvent::host_to_device(
                32,
                Duration::from_micros(3),
                true,
            ))
            .unwrap();

        accumulator.record(&first).unwrap();
        accumulator.record(&second).unwrap();
        let report = accumulator.finish();

        assert_eq!(report.kind, MoeTransferKind::HostToDevice);
        assert_eq!(report.bytes, 32);
        assert_eq!(report.cache_hits, 1);
        assert_eq!(report.cache_misses, 1);
    }
}
