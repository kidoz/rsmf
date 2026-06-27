use super::super::*;
use super::onnx::{add_graph_engine, dynamic_add_graph_engine};
use super::{f32_output, f32_output_shape};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ort::session::RunOptions;
use tempfile::tempdir;

#[test]
fn executor_runs_same_priority_fifo() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-fifo.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(add_request("first", 1.0, 10.0).with_priority(7))
        .unwrap();
    let second = executor
        .submit(add_request("second", 2.0, 20.0).with_priority(7))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.receiver.try_recv().unwrap().unwrap();
    assert_eq!(first_response.request_id, "first");
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 11.0]);
    assert!(matches!(
        second.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    assert!(executor.execute_next().unwrap());
    let second_response = second.wait().unwrap();
    assert_eq!(second_response.request_id, "second");
    assert_eq!(f32_output(&second_response, "z"), vec![22.0, 22.0]);
}

#[test]
fn executor_runs_higher_priority_first() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-priority.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let low = executor
        .submit(add_request("low", 1.0, 10.0).with_priority(1))
        .unwrap();
    let high = executor
        .submit(add_request("high", 2.0, 20.0).with_priority(9))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let high_response = high.receiver.try_recv().unwrap().unwrap();
    assert_eq!(high_response.request_id, "high");
    assert_eq!(f32_output(&high_response, "z"), vec![22.0, 22.0]);
    assert!(matches!(
        low.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
}

#[test]
fn executor_rejects_expired_deadline_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-deadline.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let expired = Instant::now() - Duration::from_secs(1);
    let handle = executor
        .submit(RuntimeRequest::new("expired", 99, RuntimeInputs::new()).with_deadline(expired))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "expired"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 1);
    assert_eq!(metrics.cancelled, 0);
    assert_eq!(metrics.current_active_input_tensor_bytes, 0);
    assert_eq!(metrics.current_active_output_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_input_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_output_tensor_bytes, 0);
}

#[test]
fn executor_rejects_zero_timeout_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-timeout.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor
        .submit(
            RuntimeRequest::new("timeout", 99, RuntimeInputs::new()).with_timeout(Duration::ZERO),
        )
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestDeadlineExceeded { request_id } if request_id == "timeout"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 1);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn executor_cancels_queued_request_before_runtime_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-cancel.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor
        .submit(RuntimeRequest::new("cancelled", 99, RuntimeInputs::new()))
        .unwrap();

    assert_eq!(handle.cancel(), RuntimeCancellationResult::Cancelled);
    assert_eq!(handle.cancel(), RuntimeCancellationResult::AlreadyCancelled);
    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::RequestCancelled { request_id } if request_id == "cancelled"
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 0);
    assert_eq!(metrics.cancelled, 1);
    assert_eq!(metrics.current_active_input_tensor_bytes, 0);
    assert_eq!(metrics.current_active_output_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_input_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_output_tensor_bytes, 0);
}

#[test]
fn cancellation_after_completion_reports_completed() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-completed-cancel.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let handle = executor.submit(add_request("done", 1.0, 10.0)).unwrap();
    let token = handle.cancellation_token();

    assert!(executor.execute_next().unwrap());
    let response = handle.wait().unwrap();
    assert_eq!(response.request_id, "done");
    assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyCompleted);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.cancelled, 0);
}

#[test]
fn running_cancellation_requests_ort_termination() {
    let token = RuntimeCancellationToken::new();
    assert!(token.try_mark_running().is_ok());
    let run_options = Arc::new(RunOptions::new().unwrap());
    token.attach_run_options(Arc::clone(&run_options)).unwrap();

    assert_eq!(
        token.cancel(),
        RuntimeCancellationResult::RunningCancellationRequested
    );
    assert!(token.is_cancellation_requested());
}

#[test]
fn pre_requested_running_cancellation_terminates_ort_run() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-preterminated-run.rsmf"));
    let handle = engine.session_handle(0, SessionOptions::default()).unwrap();
    let token = RuntimeCancellationToken::new();
    assert!(token.try_mark_running().is_ok());
    assert_eq!(token.cancel(), RuntimeCancellationResult::AlreadyRunning);

    let err = handle
        .run_with_cancellation(
            HashMap::from([
                (
                    "x".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![1.0, 2.0],
                    },
                ),
                (
                    "y".to_string(),
                    RuntimeTensor::F32 {
                        shape: vec![2],
                        data: vec![10.0, 20.0],
                    },
                ),
            ]),
            Some(&token),
        )
        .unwrap_err();

    assert!(matches!(err, RuntimeError::Ort { message, .. } if message.contains("terminate")));
    token.mark_completed();
}

#[test]
fn executor_preserves_runtime_errors() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-error.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );
    let mut request = add_request("missing-graph", 1.0, 10.0);
    request.graph_idx = 99;
    let handle = executor.submit(request).unwrap();

    assert!(executor.execute_next().unwrap());
    let err = handle.wait().unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::GraphNotFound {
            graph_idx: 99,
            graph_count: 1
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.completed, 0);
    assert_eq!(metrics.failed, 1);
    assert_eq!(metrics.deadline_expired, 0);
    assert_eq!(metrics.cancelled, 0);
    assert_eq!(metrics.current_active_input_tensor_bytes, 0);
    assert_eq!(metrics.current_active_output_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_input_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_active_output_tensor_bytes, 0);
}

#[test]
fn executor_queue_capacity_is_enforced() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-capacity.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 1,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let _handle = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorQueueFull { capacity: 1 }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_capacity, 1);
    assert_eq!(metrics.rejected_by_memory, 0);
    assert_eq!(metrics.current_queue_depth, 1);
    assert_eq!(metrics.max_observed_queue_depth, 1);
    assert_eq!(metrics.current_queued_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);
}

#[test]
fn executor_memory_budget_is_enforced_and_reported() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-memory-budget.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes: Some(16),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorQueueBytesExceeded {
            requested_bytes: 16,
            queued_bytes: 16,
            capacity_bytes: 16,
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_capacity, 0);
    assert_eq!(metrics.rejected_by_memory, 1);
    assert_eq!(metrics.current_queue_depth, 1);
    assert_eq!(metrics.current_queued_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_queue_depth, 1);
    assert_eq!(metrics.max_observed_queued_tensor_bytes, 16);

    assert!(executor.execute_next().unwrap());
    let response = first.wait().unwrap();
    assert_eq!(f32_output(&response, "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.current_queue_depth, 0);
    assert_eq!(metrics.current_queued_tensor_bytes, 0);
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.active_runtime_invocations, 0);
    assert_eq!(metrics.active_batch_size, 0);
    assert_eq!(metrics.max_active_requests, 1);
    assert_eq!(metrics.max_active_runtime_invocations, 1);
    assert_eq!(metrics.max_active_batch_size, 1);
    assert_eq!(metrics.current_active_input_tensor_bytes, 0);
    assert_eq!(metrics.current_active_output_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_input_tensor_bytes, 16);
    assert_eq!(metrics.max_observed_active_output_tensor_bytes, 8);
}

#[test]
fn executor_hard_memory_pressure_is_enforced_and_reported() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-hard-pressure.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    hard_queued_tensor_bytes: Some(16),
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let err = executor
        .submit(add_request("second", 2.0, 20.0))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorMemoryPressureExceeded {
            requested_bytes: 16,
            queued_bytes: 16,
            hard_limit_bytes: 16,
        }
    ));
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.rejected_by_memory, 0);
    assert_eq!(metrics.rejected_by_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_hard_rejections, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Hard
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

#[test]
fn executor_soft_memory_pressure_is_observable_and_released() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-soft-pressure.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 4,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    soft_queued_tensor_bytes: Some(16),
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor.submit(add_request("first", 1.0, 10.0)).unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 1);
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Soft
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

#[test]
fn executor_enforces_tenant_queue_capacity_and_releases_on_dispatch() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-tenant-capacity.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_requests_per_tenant: Some(1),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(add_request("alpha-1", 1.0, 10.0).with_tenant_id("alpha"))
        .unwrap();
    let err = executor
        .submit(add_request("alpha-2", 2.0, 20.0).with_tenant_id("alpha"))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorTenantQueueFull {
            tenant_id,
            capacity: 1,
        } if tenant_id == "alpha"
    ));

    let beta = executor
        .submit(add_request("beta-1", 3.0, 30.0).with_tenant_id("beta"))
        .unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.rejected_by_tenant_capacity, 1);
    assert_eq!(tenant_metric(&metrics, "alpha").current_queued_requests, 1);
    assert_eq!(tenant_metric(&metrics, "alpha").rejected_by_capacity, 1);
    assert_eq!(tenant_metric(&metrics, "beta").current_queued_requests, 1);

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 11.0]);
    let second_alpha = executor
        .submit(add_request("alpha-3", 4.0, 40.0).with_tenant_id("alpha"))
        .unwrap();
    let metrics = executor.metrics().unwrap();
    assert_eq!(tenant_metric(&metrics, "alpha").current_queued_requests, 1);
    assert_eq!(
        tenant_metric(&metrics, "alpha").max_observed_queued_requests,
        1
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&beta.wait().unwrap(), "z"), vec![33.0, 33.0]);
    assert!(executor.execute_next().unwrap());
    assert_eq!(
        f32_output(&second_alpha.wait().unwrap(), "z"),
        vec![44.0, 44.0]
    );
}

#[test]
fn executor_enforces_tenant_queued_tensor_byte_budget() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("executor-tenant-memory.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: None,
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes_per_tenant: Some(16),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let _alpha = executor
        .submit(add_request("alpha-1", 1.0, 10.0).with_tenant_id("alpha"))
        .unwrap();
    let err = executor
        .submit(add_request("alpha-2", 2.0, 20.0).with_tenant_id("alpha"))
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::ExecutorTenantQueueBytesExceeded {
            tenant_id,
            requested_bytes: 16,
            queued_bytes: 16,
            capacity_bytes: 16,
        } if tenant_id == "alpha"
    ));
    let _beta = executor
        .submit(add_request("beta-1", 3.0, 30.0).with_tenant_id("beta"))
        .unwrap();

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.rejected_by_tenant_memory, 1);
    assert_eq!(
        tenant_metric(&metrics, "alpha").current_queued_tensor_bytes,
        16
    );
    assert_eq!(tenant_metric(&metrics, "alpha").rejected_by_memory, 1);
    assert_eq!(
        tenant_metric(&metrics, "beta").current_queued_tensor_bytes,
        16
    );
}

#[test]
fn executor_batches_compatible_requests() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::ZERO,
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.wait().unwrap();
    let second_response = second.wait().unwrap();
    assert_eq!(f32_output_shape(&first_response, "z"), vec![1, 2]);
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output_shape(&second_response, "z"), vec![1, 2]);
    assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.failed, 0);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batched_requests, 2);
    assert_eq!(metrics.batch_fallbacks, 0);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 1);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
    assert_eq!(metrics.active_requests, 0);
    assert_eq!(metrics.active_runtime_invocations, 0);
    assert_eq!(metrics.active_batch_size, 0);
    assert_eq!(metrics.max_active_requests, 2);
    assert_eq!(metrics.max_active_runtime_invocations, 1);
    assert_eq!(metrics.max_active_batch_size, 2);
    assert_eq!(metrics.current_active_input_tensor_bytes, 0);
    assert_eq!(metrics.current_active_output_tensor_bytes, 0);
    assert_eq!(metrics.max_observed_active_input_tensor_bytes, 32);
    assert_eq!(metrics.max_observed_active_output_tensor_bytes, 16);
}

#[test]
fn executor_skips_incompatible_batch_candidates() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-batch-skip.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::ZERO,
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let incompatible = executor
        .submit(dynamic_add_request("incompatible", &[3.0, 4.0], &[30.0, 40.0]).with_priority(-1))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    let first_response = first.wait().unwrap();
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert!(matches!(
        incompatible.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    assert!(executor.execute_next().unwrap());
    let incompatible_response = incompatible.wait().unwrap();
    assert_eq!(f32_output(&incompatible_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.submitted, 2);
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.runtime_invocations, 2);
    assert_eq!(metrics.batches_executed, 0);
    assert_eq!(metrics.batched_requests, 0);
    assert_eq!(metrics.batch_fallbacks, 0);
}

#[test]
fn executor_reports_full_batch_flush_reason() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-full-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 2,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 1);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn background_scheduler_collects_compatible_arrivals_until_delay() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-delay-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 1,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_millis(100),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(matches!(
        first.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    let first_response = first.wait().unwrap();
    let second_response = second.wait().unwrap();
    assert_eq!(f32_output(&first_response, "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second_response, "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.completed, 2);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batched_requests, 2);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 1);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn background_scheduler_flushes_open_batch_on_shutdown() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-shutdown-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 1,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(60),
            }),
            admission: RuntimeAdmissionConfig::default(),
        },
    );

    let handle = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    std::thread::sleep(Duration::from_millis(20));
    assert!(matches!(
        handle.receiver.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    executor.close().unwrap();
    let response = handle.wait().unwrap();
    assert_eq!(f32_output(&response, "z"), vec![11.0, 22.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.completed, 1);
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 0);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 1);
}

#[test]
fn executor_reports_memory_pressure_batch_flush_reason() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-pressure-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig {
                max_queued_tensor_bytes: Some(32),
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_flushes, 1);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
}

#[test]
fn executor_flushes_dynamic_batch_on_soft_memory_pressure() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("executor-soft-pressure-batch.rsmf"));
    let executor = RuntimeExecutor::new(
        engine,
        RuntimeExecutorConfig {
            worker_threads: 0,
            queue_capacity: 8,
            dynamic_batching: Some(DynamicBatchingConfig {
                max_batch_size: 4,
                max_queue_delay: Duration::from_secs(1),
            }),
            admission: RuntimeAdmissionConfig {
                memory_pressure: RuntimeMemoryPressureConfig {
                    soft_queued_tensor_bytes: Some(32),
                    flush_dynamic_batches_on_soft_pressure: true,
                    ..RuntimeMemoryPressureConfig::default()
                },
                ..RuntimeAdmissionConfig::default()
            },
        },
    );

    let first = executor
        .submit(dynamic_add_request("first", &[1.0, 2.0], &[10.0, 20.0]))
        .unwrap();
    let second = executor
        .submit(dynamic_add_request("second", &[3.0, 4.0], &[30.0, 40.0]))
        .unwrap();

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.memory_pressure_soft_events, 1);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Soft
    );

    assert!(executor.execute_next().unwrap());
    assert_eq!(f32_output(&first.wait().unwrap(), "z"), vec![11.0, 22.0]);
    assert_eq!(f32_output(&second.wait().unwrap(), "z"), vec![33.0, 44.0]);

    let metrics = executor.metrics().unwrap();
    assert_eq!(metrics.runtime_invocations, 1);
    assert_eq!(metrics.batches_executed, 1);
    assert_eq!(metrics.batch_flushes_full, 0);
    assert_eq!(metrics.batch_flushes_delay, 0);
    assert_eq!(metrics.batch_flushes_memory_pressure, 1);
    assert_eq!(metrics.memory_pressure_flushes, 1);
    assert_eq!(metrics.batch_flushes_manual, 0);
    assert_eq!(metrics.batch_flushes_shutdown, 0);
    assert_eq!(
        metrics.memory_pressure_level,
        RuntimeMemoryPressureLevel::Normal
    );
}

pub(super) fn add_request(request_id: &str, x: f32, y: f32) -> RuntimeRequest {
    RuntimeRequest::new(
        request_id,
        0,
        HashMap::from([
            (
                "x".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![2],
                    data: vec![x, x],
                },
            ),
            (
                "y".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![2],
                    data: vec![y, y],
                },
            ),
        ]),
    )
}

pub(super) fn dynamic_add_request(request_id: &str, x: &[f32], y: &[f32]) -> RuntimeRequest {
    RuntimeRequest::new(
        request_id,
        0,
        HashMap::from([
            (
                "x".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![1, x.len()],
                    data: x.to_vec(),
                },
            ),
            (
                "y".to_string(),
                RuntimeTensor::F32 {
                    shape: vec![1, y.len()],
                    data: y.to_vec(),
                },
            ),
        ]),
    )
}

fn tenant_metric<'a>(
    metrics: &'a RuntimeExecutorMetrics,
    tenant_id: &str,
) -> &'a RuntimeTenantMetrics {
    metrics
        .tenant_metrics
        .iter()
        .find(|metrics| metrics.tenant_id == tenant_id)
        .unwrap()
}
