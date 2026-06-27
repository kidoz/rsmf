use super::super::*;
use super::onnx::{add_graph_engine, dynamic_add_graph_engine};

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

use tempfile::tempdir;

#[test]
fn network_server_reports_health_and_metrics() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-health.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());

    let (status, body) = http_json(server.local_addr(), "GET", "/health", None);
    assert_eq!(status, 200);
    assert_eq!(body["status"], "ok");
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);

    let (status, body) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);
    assert_eq!(body["submitted"], 0);
    assert_eq!(body["runtime_invocations"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_runs_json_inference_request() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-run.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "protocol_version": RUNTIME_NETWORK_PROTOCOL_VERSION,
        "request_id": "net-run",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 200);
    assert_eq!(body["protocol_version"], RUNTIME_NETWORK_PROTOCOL_VERSION);
    assert_eq!(body["request_id"], "net-run");
    assert_eq!(body["outputs"]["z"]["dtype"], "f32");
    assert_eq!(body["outputs"]["z"]["shape"], serde_json::json!([2]));
    assert_eq!(
        body["outputs"]["z"]["data"],
        serde_json::json!([11.0, 22.0])
    );

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["submitted"], 1);
    assert_eq!(metrics["completed"], 1);
    assert_eq!(metrics["runtime_invocations"], 1);
    assert_eq!(metrics["current_active_input_tensor_bytes"], 0);
    assert_eq!(metrics["current_active_output_tensor_bytes"], 0);
    assert_eq!(metrics["max_observed_active_input_tensor_bytes"], 16);
    assert_eq!(metrics["max_observed_active_output_tensor_bytes"], 8);

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_unsupported_protocol_version() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-protocol-version.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "protocol_version": RUNTIME_NETWORK_PROTOCOL_VERSION + 1,
        "request_id": "net-version",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "unsupported_protocol_version");
    assert_eq!(
        body["error"]["message"],
        "unsupported protocol version 2; supported version is 1"
    );

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["submitted"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_oversized_request_body() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-body-limit.rsmf"));
    let server = start_test_network_server_with_network_config(
        engine,
        RuntimeExecutorConfig::default(),
        RuntimeNetworkServerConfig {
            max_body_bytes: 8,
            ..RuntimeNetworkServerConfig::default()
        },
    );
    let (status, body) = http_raw_json(
        server.local_addr(),
        "POST /v1/run HTTP/1.1\r\nhost: test\r\ncontent-type: application/json\r\ncontent-length: 9\r\nconnection: close\r\n\r\n123456789",
    );

    assert_eq!(status, 413);
    assert_eq!(body["error"]["code"], "payload_too_large");
    assert_eq!(
        body["error"]["message"],
        "request body is 9 bytes, limit is 8"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_rejects_unsupported_content_type() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-content-type.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let body = serde_json::json!({
        "request_id": "net-content-type",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    })
    .to_string();
    let request = format!(
        "POST /v1/run HTTP/1.1\r\nhost: test\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    let (status, body) = http_raw_json(server.local_addr(), &request);

    assert_eq!(status, 400);
    assert_eq!(body["error"]["code"], "bad_request");
    assert_eq!(
        body["error"]["message"],
        "content-type must be application/json"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_sanitizes_runtime_error_response() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-sanitized-error.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "request_id": "net-runtime-error",
        "graph_idx": 99,
        "inputs": {}
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 500);
    assert_eq!(body["error"]["code"], "runtime_error");
    assert_eq!(body["error"]["message"], "runtime request failed");

    server.shutdown().unwrap();
}

#[test]
fn network_server_enforces_response_body_limit() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-response-limit.rsmf"));
    let server = start_test_network_server_with_network_config(
        engine,
        RuntimeExecutorConfig::default(),
        RuntimeNetworkServerConfig {
            max_response_body_bytes: 128,
            ..RuntimeNetworkServerConfig::default()
        },
    );
    let request = serde_json::json!({
        "request_id": "net-response-limit",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 500);
    assert_eq!(body["error"]["code"], "response_too_large");
    assert_eq!(
        body["error"]["message"],
        "response exceeded configured size limit"
    );

    server.shutdown().unwrap();
}

#[test]
fn network_server_propagates_tenant_id_to_metrics() {
    let dir = tempdir().unwrap();
    let engine = add_graph_engine(dir.path().join("network-tenant.rsmf"));
    let server = start_test_network_server(engine, RuntimeExecutorConfig::default());
    let request = serde_json::json!({
        "request_id": "net-tenant",
        "tenant_id": "tenant-a",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [2], "data": [10.0, 20.0] }
        }
    });

    let (status, body) = http_json(server.local_addr(), "POST", "/v1/run", Some(&request));
    assert_eq!(status, 200);
    assert_eq!(body["request_id"], "net-tenant");

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["tenant_metrics"][0]["tenant_id"], "tenant-a");
    assert_eq!(
        metrics["tenant_metrics"][0]["max_observed_queued_requests"],
        1
    );
    assert_eq!(metrics["tenant_metrics"][0]["current_queued_requests"], 0);

    server.shutdown().unwrap();
}

#[test]
fn network_server_cancels_inflight_request() {
    let dir = tempdir().unwrap();
    let engine = dynamic_add_graph_engine(dir.path().join("network-cancel.rsmf"));
    let server = start_test_network_server(
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
    let addr = server.local_addr();
    let request = serde_json::json!({
        "request_id": "net-cancel",
        "graph_idx": 0,
        "inputs": {
            "x": { "dtype": "f32", "shape": [1, 2], "data": [1.0, 2.0] },
            "y": { "dtype": "f32", "shape": [1, 2], "data": [10.0, 20.0] }
        }
    });
    let request_thread =
        std::thread::spawn(move || http_json(addr, "POST", "/v1/run", Some(&request)));
    std::thread::sleep(Duration::from_millis(20));

    let (status, body) = http_json(server.local_addr(), "GET", "/v1/requests/net-cancel", None);
    assert_eq!(status, 200);
    assert_eq!(body["status"], "inflight");

    let (status, body) = http_json(
        server.local_addr(),
        "DELETE",
        "/v1/requests/net-cancel",
        None,
    );
    assert_eq!(status, 200);
    assert_eq!(body["cancellation"], "Cancelled");

    let (status, body) = request_thread.join().unwrap();
    assert_eq!(status, 499);
    assert_eq!(body["error"]["code"], "cancelled");

    let (status, metrics) = http_json(server.local_addr(), "GET", "/metrics", None);
    assert_eq!(status, 200);
    assert_eq!(metrics["cancelled"], 1);

    server.shutdown().unwrap();
}

fn start_test_network_server(
    engine: Engine,
    executor_config: RuntimeExecutorConfig,
) -> RuntimeNetworkServerHandle {
    start_test_network_server_with_network_config(
        engine,
        executor_config,
        RuntimeNetworkServerConfig::default(),
    )
}

fn start_test_network_server_with_network_config(
    engine: Engine,
    executor_config: RuntimeExecutorConfig,
    network_config: RuntimeNetworkServerConfig,
) -> RuntimeNetworkServerHandle {
    RuntimeNetworkServer::new(
        RuntimeExecutor::new(engine, executor_config),
        network_config,
    )
    .start()
    .unwrap()
}

fn http_json(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&serde_json::Value>,
) -> (u16, serde_json::Value) {
    let body = body
        .map(serde_json::to_string)
        .transpose()
        .unwrap()
        .unwrap_or_default();
    let request = format!(
        "{method} {path} HTTP/1.1\r\nhost: {addr}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    http_raw_json(addr, &request)
}

fn http_raw_json(addr: SocketAddr, request: &str) -> (u16, serde_json::Value) {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    let status = headers
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u16>()
        .unwrap();
    (status, serde_json::from_str(body).unwrap())
}
