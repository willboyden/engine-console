//! Observability for the console itself (port of services/telemetry.py; ADR 10).
//!
//! * Prometheus text: hand-written so the output is prometheus_client's `generate_latest` byte for byte (metric
//!   order, `_created` gauges, sorted labels, Go-style floats, escaping). Metric names, types, labels and help
//!   texts are a fixed contract (PORTING.md §10): Grafana boards use them.
//! * OTLP traces over gRPC to `otlp_endpoint` (service.name "engine-console", one span "HTTP <METHOD>" per request
//!   with http.method / http.route / http.status_code). Export failures are non-fatal.
//! * JSON log lines on stderr: {"ts", "level", "logger", "msg"} with secrets scrubbed (services/common.rs redact).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::sync::Recover as _;

use crate::domain::models::HostMemInfo;
use crate::pyjson::go_float;
use crate::services::common::now;

pub const BUCKETS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
];
pub const INSTANCE_STATES: [&str; 8] = [
    "stopped",
    "starting",
    "loading",
    "ready",
    "stopping",
    "failed",
    "auth_required",
    "unreachable",
];
pub const HOST_GAUGES: [&str; 4] = ["ram_used", "ram_available", "shmem", "swap_used"];
pub const METRICS_CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

#[derive(Debug, Clone)]
struct Child {
    labels: Vec<String>,
    value: f64,
    created: f64,
}

#[derive(Debug, Clone)]
struct HistChild {
    labels: Vec<String>,
    buckets: [f64; 15],
    count: f64,
    sum: f64,
    created: f64,
}

#[derive(Debug, Default)]
struct State {
    requests: Vec<Child>,
    latency: Vec<HistChild>,
    bench_runs: f64,
    bench_created: f64,
    instances: Vec<Child>,
    host: HashMap<&'static str, f64>,
    instance_ram: Vec<Child>,
    host_up: Vec<Child>,
    host_gpu_used: Vec<Child>,
    instance_vram: Vec<Child>,
}

fn upsert(children: &mut Vec<Child>, labels: Vec<String>, f: impl FnOnce(&mut f64)) {
    if let Some(c) = children.iter_mut().find(|c| c.labels == labels) {
        f(&mut c.value);
    } else {
        let mut c = Child {
            labels,
            value: 0.0,
            created: now(),
        };
        f(&mut c.value);
        children.push(c);
    }
}

/// The console's own Prometheus registry.
#[derive(Debug)]
pub struct Telemetry {
    state: Mutex<State>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new()
    }
}

impl Telemetry {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                bench_created: now(),
                ..State::default()
            }),
        }
    }

    fn with<T>(&self, f: impl FnOnce(&mut State) -> T) -> Option<T> {
        Some(f(&mut self.state.guard()))
    }

    /// engine_console_http_requests_total{method,route,status}
    pub fn request(&self, method: &str, route: &str, status: u16) {
        self.with(|s| {
            upsert(
                &mut s.requests,
                vec![method.into(), route.into(), status.to_string()],
                |v| *v += 1.0,
            );
        });
    }

    /// engine_console_http_request_seconds{method,route}
    pub fn observe_latency(&self, method: &str, route: &str, seconds: f64) {
        self.with(|s| {
            let labels = vec![method.to_string(), route.to_string()];
            let idx = match s.latency.iter().position(|c| c.labels == labels) {
                Some(i) => i,
                None => {
                    s.latency.push(HistChild {
                        labels,
                        buckets: [0.0; 15],
                        count: 0.0,
                        sum: 0.0,
                        created: now(),
                    });
                    s.latency.len() - 1
                }
            };
            let c = &mut s.latency[idx];
            for (i, b) in BUCKETS.iter().enumerate() {
                if seconds <= *b {
                    c.buckets[i] += 1.0;
                }
            }
            c.buckets[14] += 1.0;
            c.count += 1.0;
            c.sum += seconds;
        });
    }

    /// engine_console_bench_runs_total
    pub fn bench_run(&self) {
        self.with(|s| s.bench_runs += 1.0);
    }

    pub fn set_instance_states(&self, counts: &HashMap<String, i64>) {
        self.with(|s| {
            for st in INSTANCE_STATES {
                #[allow(clippy::cast_precision_loss)]
                let v = counts.get(st).copied().unwrap_or(0) as f64;
                upsert(&mut s.instances, vec![st.to_string()], |x| *x = v);
            }
        });
    }

    pub fn set_host_memory(&self, m: Option<&HostMemInfo>, per_instance: &[(String, f64)]) {
        self.with(|s| {
            if let Some(m) = m {
                for (k, v) in [
                    ("ram_used", m.used_gib),
                    ("ram_available", m.available_gib),
                    ("shmem", m.shmem_gib),
                    ("swap_used", m.swap_used_gib),
                ] {
                    s.host.insert(k, v);
                }
            }
            s.instance_ram.clear();
            for (iid, gib) in per_instance {
                upsert(&mut s.instance_ram, vec![iid.clone()], |x| *x = *gib);
            }
        });
    }

    /// Host links, per-GPU VRAM use and per-instance VRAM; label sets are cleared first (bounded cardinality).
    pub fn set_fleet(
        &self,
        up: &[(String, bool)],
        gpu_used: &[((String, i64), f64)],
        vram: &[((String, String, i64), f64)],
    ) {
        self.with(|s| {
            s.host_up.clear();
            s.host_gpu_used.clear();
            s.instance_vram.clear();
            for (h, is_up) in up {
                upsert(&mut s.host_up, vec![h.clone()], |x| {
                    *x = if *is_up { 1.0 } else { 0.0 };
                });
            }
            for ((h, g), gib) in gpu_used {
                upsert(&mut s.host_gpu_used, vec![h.clone(), g.to_string()], |x| {
                    *x = *gib;
                });
            }
            for ((i, h, g), gib) in vram {
                upsert(
                    &mut s.instance_vram,
                    vec![i.clone(), h.clone(), g.to_string()],
                    |x| *x = *gib,
                );
            }
        });
    }

    /// `generate_latest(registry)`
    pub fn render(&self) -> String {
        self.with(|s| render_state(s)).unwrap_or_default()
    }
}

fn esc_help(s: &str) -> String {
    s.replace('\\', r"\\").replace('\n', r"\n")
}

fn esc_label(s: &str) -> String {
    s.replace('\\', r"\\")
        .replace('\n', r"\n")
        .replace('"', "\\\"")
}

fn labelstr(names: &[&str], values: &[String], extra: Option<(&str, &str)>) -> String {
    let mut pairs: Vec<(&str, String)> =
        names.iter().copied().zip(values.iter().cloned()).collect();
    if let Some((k, v)) = extra {
        pairs.push((k, v.to_string()));
    }
    if pairs.is_empty() {
        return String::new();
    }
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    let body: Vec<String> = pairs
        .iter()
        .map(|(k, v)| format!("{k}=\"{}\"", esc_label(v)))
        .collect();
    format!("{{{}}}", body.join(","))
}

fn head(out: &mut String, name: &str, help: &str, typ: &str) {
    let _ = writeln!(out, "# HELP {name} {}", esc_help(help));
    let _ = writeln!(out, "# TYPE {name} {typ}");
}

fn gauge(out: &mut String, name: &str, help: &str, names: &[&str], children: &[Child]) {
    head(out, name, help, "gauge");
    for c in children {
        let _ = writeln!(
            out,
            "{name}{} {}",
            labelstr(names, &c.labels, None),
            go_float(c.value)
        );
    }
}

fn counter(out: &mut String, base: &str, help: &str, names: &[&str], children: &[Child]) {
    head(out, &format!("{base}_total"), help, "counter");
    for c in children {
        let _ = writeln!(
            out,
            "{base}_total{} {}",
            labelstr(names, &c.labels, None),
            go_float(c.value)
        );
    }
    if !children.is_empty() {
        head(out, &format!("{base}_created"), help, "gauge");
        for c in children {
            let _ = writeln!(
                out,
                "{base}_created{} {}",
                labelstr(names, &c.labels, None),
                go_float(c.created)
            );
        }
    }
}

fn render_state(s: &State) -> String {
    let mut out = String::new();
    counter(
        &mut out,
        "engine_console_http_requests",
        "HTTP requests",
        &["method", "route", "status"],
        &s.requests,
    );
    let hname = "engine_console_http_request_seconds";
    head(&mut out, hname, "HTTP request latency", "histogram");
    let names = ["method", "route"];
    for c in &s.latency {
        for (i, b) in BUCKETS
            .iter()
            .map(|b| go_float(*b))
            .chain(std::iter::once("+Inf".to_string()))
            .enumerate()
        {
            let _ = writeln!(
                out,
                "{hname}_bucket{} {}",
                labelstr(&names, &c.labels, Some(("le", &b))),
                go_float(c.buckets[i])
            );
        }
        let _ = writeln!(
            out,
            "{hname}_count{} {}",
            labelstr(&names, &c.labels, None),
            go_float(c.count)
        );
        let _ = writeln!(
            out,
            "{hname}_sum{} {}",
            labelstr(&names, &c.labels, None),
            go_float(c.sum)
        );
    }
    if !s.latency.is_empty() {
        head(
            &mut out,
            &format!("{hname}_created"),
            "HTTP request latency",
            "gauge",
        );
        for c in &s.latency {
            let _ = writeln!(
                out,
                "{hname}_created{} {}",
                labelstr(&names, &c.labels, None),
                go_float(c.created)
            );
        }
    }
    let bench = [Child {
        labels: Vec::new(),
        value: s.bench_runs,
        created: s.bench_created,
    }];
    counter(
        &mut out,
        "engine_console_bench_runs",
        "Benchmarks started",
        &[],
        &bench,
    );
    gauge(
        &mut out,
        "engine_console_instances",
        "Instances by state",
        &["state"],
        &s.instances,
    );
    for k in HOST_GAUGES {
        let v = s.host.get(k).copied().unwrap_or(0.0);
        let help = format!("Host {} in GiB", k.replace('_', " "));
        gauge(
            &mut out,
            &format!("engine_console_host_{k}_gib"),
            &help,
            &[],
            &[Child {
                labels: Vec::new(),
                value: v,
                created: 0.0,
            }],
        );
    }
    gauge(
        &mut out,
        "engine_console_instance_host_ram_gib",
        "Host RAM held by an instance's container (cgroup)",
        &["instance"],
        &s.instance_ram,
    );
    gauge(
        &mut out,
        "engine_console_host_up",
        "1 when the node of a host is connected",
        &["host"],
        &s.host_up,
    );
    gauge(
        &mut out,
        "engine_console_host_gpu_used_gib",
        "VRAM in use on a host's GPU",
        &["host", "gpu"],
        &s.host_gpu_used,
    );
    gauge(
        &mut out,
        "engine_console_instance_vram_gib",
        "VRAM the driver attributes to an instance",
        &["instance", "host", "gpu"],
        &s.instance_vram,
    );
    out
}

// ── tracing (OTLP) ───────────────────────────────────────────────────────────────────────────────────────
pub mod otel {
    use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
    use opentelemetry::{KeyValue, global};
    use opentelemetry_otlp::WithExportConfig as _;
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    use crate::config::Settings;

    /// OTLP/gRPC to the local collector. Export failures are non-fatal (the batch processor drops them).
    /// Must run inside the tokio runtime.
    pub fn setup_tracing(cfg: &Settings) -> Option<SdkTracerProvider> {
        if !cfg.otlp_enabled {
            return None;
        }
        let endpoint = if cfg.otlp_endpoint.contains("://") {
            cfg.otlp_endpoint.clone()
        } else {
            format!("http://{}", cfg.otlp_endpoint) // insecure=True, as in Python
        };
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .ok()?;
        let provider = SdkTracerProvider::builder()
            .with_resource(
                Resource::builder()
                    .with_service_name("engine-console")
                    .build(),
            )
            .with_batch_exporter(exporter)
            .build();
        global::set_tracer_provider(provider.clone());
        Some(provider)
    }

    /// One finished request span: "HTTP <METHOD>" with method, route and status attributes.
    pub fn request_span(method: &str, route: &str, status: u16, start: std::time::SystemTime) {
        let tracer = global::tracer_provider().tracer("engine_console");
        let mut span = tracer
            .span_builder(format!("HTTP {method}"))
            .with_start_time(start)
            .start(&tracer);
        span.set_attribute(KeyValue::new("http.method", method.to_string()));
        span.set_attribute(KeyValue::new("http.route", route.to_string()));
        span.set_attribute(KeyValue::new("http.status_code", i64::from(status)));
        span.end();
    }
}

// ── logging ──────────────────────────────────────────────────────────────────────────────────────────────
pub mod logging {
    use std::io::Write as _;

    use serde_json::{Map, Value};
    use tracing::field::{Field, Visit};
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::{Context, SubscriberExt as _};
    use tracing_subscriber::util::SubscriberInitExt as _;

    use crate::pyjson;
    use crate::services::common::redact;

    /// JSON lines like Python's JsonFormatter: {"ts": "%Y-%m-%dT%H:%M:%S%z", "level", "logger", "msg"}.
    /// The logger name is the event's target ("engine_console.*" in this crate's calls). Times are UTC
    /// (PORTING.md deviation D-LOG-TZ).
    struct JsonLayer;

    #[derive(Default)]
    struct Msg(String);

    impl Visit for Msg {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            if field.name() == "message" {
                self.0 = value.to_string();
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for JsonLayer {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
            let meta = event.metadata();
            let level = match *meta.level() {
                tracing::Level::ERROR => "ERROR",
                tracing::Level::WARN => "WARNING",
                tracing::Level::INFO => "INFO",
                _ => "DEBUG",
            };
            if level == "DEBUG" {
                return;
            }
            let mut m = Msg::default();
            event.record(&mut m);
            let ts = httpdate_like_iso();
            let mut doc = Map::new();
            doc.insert("ts".into(), Value::String(ts));
            doc.insert("level".into(), Value::String(level.into()));
            doc.insert(
                "logger".into(),
                Value::String(meta.target().replace("::", ".")),
            );
            doc.insert("msg".into(), redact(&Value::String(m.0)));
            let line = pyjson::dumps(&Value::Object(doc));
            let _ = writeln!(std::io::stderr(), "{line}");
        }
    }

    fn httpdate_like_iso() -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let days = i64::try_from(secs / 86_400).unwrap_or(0);
        let rem = secs % 86_400;
        let (y, mo, d) = civil_from_days(days);
        format!(
            "{y:04}-{mo:02}-{d:02}T{:02}:{:02}:{:02}+0000",
            rem / 3600,
            (rem % 3600) / 60,
            rem % 60
        )
    }

    /// Howard Hinnant's days-to-civil.
    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
        let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).unwrap_or(1);
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    /// `configure_logging()`: install the JSON layer once (INFO and up).
    pub fn configure_logging() {
        let _ = tracing_subscriber::registry().with(JsonLayer).try_init();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_matches_prometheus_client_layout() {
        let t = Telemetry::new();
        let empty = t.render();
        assert!(empty.starts_with(
            "# HELP engine_console_http_requests_total HTTP requests\n# TYPE engine_console_http_requests_total counter\n# HELP engine_console_http_request_seconds HTTP request latency\n"
        ));
        assert!(empty.contains(
            "engine_console_bench_runs_total 0.0\n# HELP engine_console_bench_runs_created"
        ));
        assert!(empty.contains("# HELP engine_console_host_ram_used_gib Host ram used in GiB\n"));
        t.request("GET", "a\"b\\c\nd", 200);
        t.observe_latency("GET", "/api/v1/health", 0.0123);
        t.set_fleet(
            &[("local".into(), true)],
            &[(("local".into(), 0), 3.5)],
            &[(("inst_a".into(), "local".into(), 0), 2.0)],
        );
        let out = t.render();
        assert!(out.contains("engine_console_http_requests_total{method=\"GET\",route=\"a\\\"b\\\\c\\nd\",status=\"200\"} 1.0\n"));
        assert!(out.contains("engine_console_http_request_seconds_bucket{le=\"0.01\",method=\"GET\",route=\"/api/v1/health\"} 0.0\n"));
        assert!(out.contains("engine_console_http_request_seconds_bucket{le=\"0.025\",method=\"GET\",route=\"/api/v1/health\"} 1.0\n"));
        assert!(out.contains("engine_console_http_request_seconds_bucket{le=\"+Inf\",method=\"GET\",route=\"/api/v1/health\"} 1.0\n"));
        assert!(out.contains("engine_console_http_request_seconds_sum{method=\"GET\",route=\"/api/v1/health\"} 0.0123\n"));
        assert!(out.contains(
            "engine_console_instance_vram_gib{gpu=\"0\",host=\"local\",instance=\"inst_a\"} 2.0\n"
        ));
        assert!(out.ends_with(
            "engine_console_instance_vram_gib{gpu=\"0\",host=\"local\",instance=\"inst_a\"} 2.0\n"
        ));
    }
}
