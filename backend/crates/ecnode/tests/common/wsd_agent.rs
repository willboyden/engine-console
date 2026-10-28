//! Node test kit, agent half (conftest.agent_for): an Agent over the fake machine, the fake binaries and the safe
//! prober. Include after wsd_kit: `#[path = "common/wsd_agent.rs"] mod wsd_agent;`.
#![allow(
    dead_code,
    unreachable_pub,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ec_proto::Obj;
use ecnode::agent::{Agent, AgentParts};
use ecnode::errors::{Call, HandlerError, Stream};
use ecnode::gpus::Gpus;
use ecnode::util::{EnvMap, ProcRunner, Runner};
use serde_json::Value;

use crate::wsd_kit::{
    FakeBin, FakeStream, Machine, SafeProber, clean_args, machine_statvfs, store_for,
};

/// tmp dir + fake binaries + fake machine + safe prober (the conftest fixtures together).
pub struct World {
    pub tmp: tempfile::TempDir,
    pub fb: FakeBin,
    pub machine: Machine,
    pub prober: SafeProber,
}

impl World {
    pub fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let fb = FakeBin::new(tmp.path());
        let machine = Machine::new(tmp.path());
        Self {
            tmp,
            fb,
            machine,
            prober: SafeProber::default(),
        }
    }

    pub fn path(&self) -> &Path {
        self.tmp.path()
    }

    pub fn parts(&self, env: EnvMap) -> AgentParts {
        let runner: Arc<dyn Runner> = Arc::new(ProcRunner::new(env.clone()));
        AgentParts {
            paths: self.machine.paths.clone(),
            gpus: Some(Gpus::new(false, Arc::clone(&runner))),
            runner,
            env,
            statvfs: Some(machine_statvfs()),
            prober: self.prober.prober(),
            quiet_prober: self.prober.quiet_prober(),
            metrics_get: self.prober.metrics_get(),
            default_socket: self
                .tmp
                .path()
                .join("no-such.sock")
                .to_string_lossy()
                .into_owned(),
            ..AgentParts::default()
        }
    }

    /// conftest.agent_for(policy)
    pub fn agent(&self, policy: Option<&Value>) -> Agent {
        Agent::with_parts(store_for(self.path(), policy), self.parts(self.fb.env()))
    }

    pub fn agent_with(&self, policy: Option<&Value>, parts: AgentParts) -> Agent {
        Agent::with_parts(store_for(self.path(), policy), parts)
    }

    pub fn state(&self) -> PathBuf {
        self.machine.paths.state.clone()
    }
}

/// test_docker.call: clean the args like the server, then run the handler.
pub fn call(
    a: &Agent,
    op: &str,
    args: Value,
    stream: Option<&FakeStream>,
) -> Result<Obj, HandlerError> {
    let cleaned = clean_args(op, args);
    let missing = ec_proto::missing_args(op, &cleaned);
    assert!(missing.is_empty(), "{op} misses {missing:?}");
    let empty = FakeStream::empty();
    let s: &dyn Stream = match stream {
        Some(s) => s,
        None => &empty,
    };
    ecnode::ops::dispatch(
        a,
        &Call {
            op,
            args: &cleaned,
            stream: s,
        },
    )
}

pub fn ok(a: &Agent, op: &str, args: Value) -> Obj {
    match call(a, op, args, None) {
        Ok(r) => r,
        Err(e) => panic!("{op}: {e:?}"),
    }
}

/// The OpError (code, message) an op fails with.
pub fn err(a: &Agent, op: &str, args: Value) -> (String, String) {
    match call(a, op, args, None) {
        Err(HandlerError::Op(e)) => (e.code, e.message),
        other => panic!("{op}: expected an OpError, got {other:?}"),
    }
}
