use clap::Parser;

mod cmd;
use crate::cmd::{CliError, Command};

#[derive(Debug, Parser)]
#[command(name = "rsmf")]
#[command(about = "Rust Split Model Format utility", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

fn main() {
    // Initialize tracing if requested via RUST_LOG environment variable.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    if let Err::<(), CliError>(e) = cli.command.run() {
        eprintln!("error: {e}");
        std::process::exit(e.exit_code());
    }
}
