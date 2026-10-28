//! Composition root (port of container.py): builds every service from its ports, in Python's order, with the
//! late-bound links (library <-> lifecycle, fit -> lifecycle residents, fleet -> lifecycle resolver/counts).
//! Tests inject fakes through `Overrides`.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock, Weak};

use crate::adapters::{AdapterRegistry, load_default_adapters};
use crate::config::Settings;
use crate::domain::fleet::{Fleet, FleetPort};
use crate::domain::ports::HubClient;
use crate::services::audit::AuditService;
use crate::services::bench::BenchService;
use crate::services::chat::ChatService;
use crate::services::common::{EventBus, SseLimiter};
use crate::services::downloads::DownloadService;
use crate::services::fitting::FitService;
use crate::services::fleet::FleetService;
use crate::services::hardware::HardwareService;
use crate::services::hf::HfService;
use crate::services::hf_http::{EgressConfig, HfHttpClient};
use crate::services::hosts::HostsError;
use crate::services::library::LibraryService;
use crate::services::lifecycle::{LifecycleDeps, LifecycleService};
use crate::services::metrics::MetricsService;
use crate::services::nodehttp::NodeHttp;
use crate::services::profiles::ProfileService;
use crate::services::scripts::ScriptService;
use crate::services::secrets_store::SecretStore;
use crate::services::settings::SettingsService;
use crate::services::store::Store;
use crate::services::telemetry::Telemetry;
use crate::services::usage::UsageService;

pub struct Container {
    pub cfg: Arc<Settings>,
    pub store: Arc<Store>,
    pub bus: Arc<EventBus>,
    pub adapters: AdapterRegistry,
    pub settings: Arc<SettingsService>,
    pub fleet: Arc<dyn Fleet>,
    pub hub: Arc<dyn HubClient>,
    pub hardware: Arc<HardwareService>,
    pub hf: Arc<HfService>,
    pub library: Arc<LibraryService>,
    pub downloads: Arc<DownloadService>,
    pub fit: Arc<FitService>,
    pub scripts: Arc<ScriptService>,
    pub nodehttp: Arc<NodeHttp>,
    pub lifecycle: Arc<LifecycleService>,
    pub profiles: Arc<ProfileService>,
    pub metrics: Arc<MetricsService>,
    pub bench: Arc<BenchService>,
    pub usage: Arc<UsageService>,
    pub chat: Arc<ChatService>,
    pub audit: Arc<AuditService>,
    pub telemetry: Arc<Telemetry>,
    pub sse: Arc<SseLimiter>,
    pub chat_streams: Arc<SseLimiter>,
    /// the resolved frontend dir when it exists (the SPA route is registered only then)
    pub frontend: Option<PathBuf>,
    /// None: the packaged learn content (learn/mod.rs `bundle`)
    pub learn: Option<Arc<crate::domain::learn::LearnBundle>>,
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Container")
    }
}

impl Container {
    /// The learn content served by /learn and merged into /engines (the packaged one unless overridden).
    pub fn learn_bundle(&self) -> Result<Arc<crate::domain::learn::LearnBundle>, String> {
        match &self.learn {
            Some(b) => Ok(Arc::clone(b)),
            None => crate::learn::bundle(),
        }
    }

    /// The fleet as the narrower port services take.
    pub fn fleet_port(&self) -> Arc<dyn FleetPort> {
        Arc::clone(&self.fleet) as Arc<dyn FleetPort>
    }
}

/// What a test may replace.
#[derive(Default)]
pub struct Overrides {
    pub adapters: Option<AdapterRegistry>,
    pub fleet: Option<Arc<dyn Fleet>>,
    pub hub: Option<Arc<dyn HubClient>>,
    pub store: Option<Arc<Store>>,
    pub frontend: Option<Option<PathBuf>>,
    /// the learn content to serve instead of the packaged one (Python tests monkeypatch `learn_bundle`)
    pub learn: Option<Arc<crate::domain::learn::LearnBundle>>,
    /// the library the runtime services (the lifecycle) see instead of the real one: Python's `library=` override.
    /// The routes keep the real `LibraryService` (its concrete API is the library's own); PORTING.md §A4.
    pub library: Option<Arc<dyn crate::services::libport::LibraryPort>>,
}

/// Why the container could not be built.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    /// hosts.json is not acceptable (exit 2, like Python's HostsError)
    #[error(transparent)]
    Hosts(#[from] HostsError),
    /// the database could not be opened or migrated
    #[error("{0}")]
    Store(String),
}

pub fn egress_config(cfg: &Settings) -> EgressConfig {
    EgressConfig {
        proxy: cfg.egress_proxy.clone(),
        ca_bundle: cfg.egress_ca_bundle.clone(),
        require_proxy: cfg.require_egress_proxy,
    }
}

fn resolve_frontend(cfg: &Settings) -> Option<PathBuf> {
    let root = cfg
        .frontend_dir
        .clone()
        .unwrap_or_else(crate::api::statics::default_frontend_dir);
    if !root.is_dir() {
        return None;
    }
    std::fs::canonicalize(&root).ok()
}

pub fn build_container(cfg: Arc<Settings>, o: Overrides) -> Result<Arc<Container>, BuildError> {
    let library_override = o.library;
    let store = match o.store {
        Some(s) => s,
        None => Arc::new(Store::open(&cfg.db_path()).map_err(BuildError::Store)?),
    };
    let bus = Arc::new(EventBus::new());
    let reg = o.adapters.unwrap_or_else(load_default_adapters);
    let images_reg = reg.clone();
    let settings = Arc::new(SettingsService::new(
        Arc::clone(&store),
        Arc::clone(&cfg),
        Box::new(move || {
            images_reg
                .list()
                .iter()
                .map(|a| a.default_image().to_string())
                .collect()
        }),
    ));
    let hub: Arc<dyn HubClient> = match o.hub {
        Some(h) => h,
        None => {
            let s = Arc::clone(&settings);
            Arc::new(HfHttpClient::new(
                &cfg.hf_endpoint,
                cfg.hf_allowed_hosts.clone(),
                Arc::new(move || s.hf_token()),
                egress_config(&cfg),
            ))
        }
    };
    let hf = Arc::new(HfService::new(Arc::clone(&hub)));
    let fleet: Arc<dyn Fleet> = match o.fleet {
        Some(f) => f,
        None => Arc::new(FleetService::new(Arc::clone(&cfg), Arc::clone(&bus))?),
    };
    let port = Arc::clone(&fleet) as Arc<dyn FleetPort>;
    let hardware = Arc::new(HardwareService::new(Arc::clone(&port)));
    let nodehttp = Arc::new(NodeHttp::new(Arc::clone(&port)));
    let scripts = Arc::new(ScriptService::new(Arc::clone(&port)));
    let usage = Arc::new(UsageService::new(Arc::clone(&store)));
    let fit = Arc::new(FitService::new(
        reg.clone(),
        Arc::clone(&hf),
        Arc::clone(&hardware),
        Arc::clone(&settings),
    ));
    // the library needs to know which instances use a model, and the lifecycle needs the library: bind the first late
    let holder: Arc<OnceLock<Weak<LifecycleService>>> = Arc::new(OnceLock::new());
    let h2 = Arc::clone(&holder);
    let library = Arc::new(LibraryService::new(
        Arc::clone(&store),
        Arc::clone(&port),
        reg.clone(),
        Arc::clone(&bus),
        Arc::new(move |host: &str, path: &str| {
            h2.get()
                .and_then(Weak::upgrade)
                .map_or_else(Vec::new, |l| l.instances_using(host, path))
        }),
    ));
    let lifecycle = Arc::new(LifecycleService::new(LifecycleDeps {
        db: Arc::clone(&store),
        adapters: reg.clone(),
        fleet: Arc::clone(&port),
        hardware: Arc::clone(&hardware),
        settings: Arc::clone(&settings),
        fit: Arc::clone(&fit),
        http: Arc::clone(&nodehttp),
        cfg: Arc::clone(&cfg),
        bus: Arc::clone(&bus),
        secrets: SecretStore::new(cfg.data_dir.join("secrets")),
    }));
    let _ = holder.set(Arc::downgrade(&lifecycle));
    lifecycle.set_library(
        library_override.unwrap_or_else(|| {
            Arc::clone(&library) as Arc<dyn crate::services::libport::LibraryPort>
        }),
    );
    lifecycle.set_scripts(Arc::clone(&scripts));
    let weak = Arc::downgrade(&lifecycle);
    fit.set_resident_provider(Arc::new(move |host: &str| {
        weak.upgrade().map_or_else(Vec::new, |l| l.resident(host))
    }));
    let downloads = Arc::new(DownloadService::new(
        Arc::clone(&store),
        Arc::clone(&hub),
        Arc::clone(&hf),
        Arc::clone(&settings),
        Arc::clone(&bus),
        Arc::clone(&port),
        Arc::clone(&library),
        cfg.download_concurrency,
        cfg.max_download_queue,
    ));
    // what the fleet shows next to each host: which instance a GPU process belongs to, how many engines/models
    let w1 = Arc::downgrade(&lifecycle);
    fleet.set_instance_resolver(Arc::new(move |host: &str, container: &str| {
        w1.upgrade()
            .and_then(|l| l.instance_for_container(host, container))
    }));
    let w2 = Arc::downgrade(&lifecycle);
    fleet.set_counts_provider(Arc::new(move |host: &str| {
        w2.upgrade().map(|l| l.counts(host)).unwrap_or_default()
    }));
    let profiles = Arc::new(ProfileService::new(Arc::clone(&store), reg.clone()));
    let metrics = Arc::new(MetricsService::new(
        Arc::clone(&store),
        Arc::clone(&lifecycle),
        reg.clone(),
        Arc::clone(&port),
        Arc::clone(&nodehttp),
        Arc::clone(&bus),
        cfg.scrape_interval_s,
    ));
    let bench = Arc::new(BenchService::new(
        Arc::clone(&store),
        Arc::clone(&lifecycle),
        Arc::clone(&nodehttp),
        Arc::clone(&bus),
    ));
    let chat = Arc::new(ChatService::new(
        Arc::clone(&store),
        Arc::clone(&lifecycle),
        Arc::clone(&nodehttp),
        Arc::clone(&usage),
    ));
    let audit = Arc::new(AuditService::new(Arc::clone(&store)));
    let frontend = o.frontend.unwrap_or_else(|| resolve_frontend(&cfg));
    Ok(Arc::new(Container {
        sse: Arc::new(SseLimiter::new(
            cfg.max_sse_connections,
            Some(cfg.max_sse_per_key),
        )),
        chat_streams: Arc::new(SseLimiter::new(
            cfg.max_chat_streams,
            Some(cfg.max_chat_streams_per_key),
        )),
        cfg,
        store,
        bus,
        adapters: reg,
        settings,
        fleet,
        hub,
        hardware,
        hf,
        library,
        downloads,
        fit,
        scripts,
        nodehttp,
        lifecycle,
        profiles,
        metrics,
        bench,
        usage,
        chat,
        audit,
        telemetry: Arc::new(Telemetry::new()),
        frontend,
        learn: o.learn,
    }))
}
