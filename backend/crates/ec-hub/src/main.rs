//! `engine-console`: serve the UI and API (no subcommand); `engine-console hosts ...` manages the fleet's machines
//! (cli.rs); `engine-console open|stop` is the desktop launcher (desktop.rs). Port of main.py `run()`.

use std::process::ExitCode;
use std::sync::Arc;

use ec_hub::config::{Settings, env_file, legacy_env_warning};
use ec_hub::container::{BuildError, Overrides, build_container};
use ec_hub::services::telemetry::logging::configure_logging;

fn code(c: i32) -> ExitCode {
    ExitCode::from(u8::try_from(c).unwrap_or(1))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match argv.first().map(String::as_str) {
        Some("hosts") => return code(ec_hub::cli::hosts_cli(&argv[1..])),
        Some("open") => return code(ec_hub::desktop::open_console(&argv[1..])),
        Some("stop") => return code(ec_hub::desktop::stop_console(&argv[1..])),
        _ => {}
    }
    // anything else serves, as Python's run() does (its argv is not parsed); `--from-launcher` marks a console
    // started by `engine-console open` (desktop.rs), which is the only kind `stop` will signal
    let cfg = match Settings::from_env() {
        Ok(c) => c,
        Err(e) => {
            // names and reasons only: never the value (PORTING.md deviation D-SETTINGS-ERR)
            #[allow(clippy::print_stderr)]
            {
                eprintln!("engine-console: invalid setting {}: {}", e.field, e.message);
            }
            return ExitCode::from(1);
        }
    };
    configure_logging();
    warn_legacy_env();
    let Ok(rt) = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    else {
        return ExitCode::from(1);
    };
    // one process hook: an exception an adapter raises is answered (500) or caught (`try_*`) and logged, so it stays
    // off stderr; any other panic is written with its secrets masked (panics.rs)
    ec_hub::panics::install_panic_hook();
    rt.block_on(async move {
        let c = match build_container(Arc::new(cfg), Overrides::default()) {
            Ok(c) => c,
            Err(BuildError::Hosts(e)) => {
                #[allow(clippy::print_stderr)]
                {
                    eprintln!("engine-console: {e}");
                }
                return ExitCode::from(2);
            }
            Err(e) => {
                tracing::error!(target: "engine_console", "cannot start: {}", e);
                return ExitCode::from(1);
            }
        };
        code(ec_hub::app::serve(c).await)
    })
}

/// One warning when an old unprefixed name is still set (names only, never values).
fn warn_legacy_env() {
    let environ: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
        .collect();
    if let Some(msg) = legacy_env_warning(&environ, Some(&env_file())) {
        tracing::warn!(target: "engine_console", "{}", msg);
    }
}
