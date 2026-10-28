//! `ecnode`: see `ecnode::entry` (port of node/src/ecnode/__main__.py).

fn main() {
    // every panic is reported as one redacted line on stderr, never the raw message (util::panics)
    ecnode::util::panics::install_hook();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = ecnode::entry::main_with(&args, &mut std::io::stdout(), &mut std::io::stderr());
    std::process::exit(code);
}
