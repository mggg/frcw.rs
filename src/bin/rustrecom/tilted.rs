//! The `tilted` subcommand: a tilted run optimizer for redistricting
//! (formerly the `frcw_tilted` binary).
//!
//! Arguments arrive either from CLI flags or, via `--config`, from a versioned
//! JSON config whose fields mirror those flags. Config mode preserves the
//! exact raw config string as provenance: it becomes the metadata sidecar's
//! content (or the BENDL Metadata asset).

use crate::common;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use rustrecom::config::{parse_tilted_config, region_weights_from_map};
use rustrecom::objectives::{ensure_derived_perim_column, polsby_popper_autoderive};
use rustrecom::recom::tilted::{
    core::REVERSIBLE_UNSUPPORTED, multi_tilted_runs_with_writer, ExponentialAcceptance,
    FixedAcceptance, IncrementalBackend, LinearAcceptance,
};
use rustrecom::recom::RecomParams;
use serde_json::json;

pub fn command() -> Command {
    Command::new("tilted")
        .about("A tilted run optimizer for redistricting")
        .arg(common::config_arg())
        .arg(common::config_optional(common::graph_json_arg()))
        .arg(common::config_optional(
            Arg::new("n_steps")
                .long("n-steps")
                .value_parser(value_parser!(u64))
                .help("The total number of chain steps (accepted + rejected)."),
        ))
        .arg(common::config_optional(common::tol_arg()))
        .arg(common::config_optional(common::pop_col_arg()))
        .arg(common::config_optional(common::assignment_col_arg()))
        .arg(common::config_optional(common::rng_seed_arg()))
        .arg(
            Arg::new("n_threads")
                .long("n-threads")
                .required(false)
                .value_parser(value_parser!(usize))
                .default_value("1")
                .help("Number of worker threads for parallel tree drawing."),
        )
        .arg(
            Arg::new("accept_rule")
                .long("accept-rule")
                .required(false)
                .value_parser(["fixed", "linear", "exponential"])
                .default_value("linear")
                .help(
                    "Acceptance rule for proposals with a worse score. \
                    'fixed' accepts with constant probability '--accept-worse-prob'. \
                    'linear' accepts with probability max(0, 1 - beta * score_loss). \
                    'exponential' accepts with probability exp(beta * delta) using \
                    '--acceptance-beta'.",
                ),
        )
        .arg(
            Arg::new("accept_worse_prob")
                .long("accept-worse-prob")
                .required(false)
                .value_parser(value_parser!(f64))
                .help(
                    "Probability of accepting a proposal with a worse score under \
                    '--accept-rule fixed'. Must be in [0, 1]. Use 0.0 for pure \
                    hill-climbing, 1.0 for a random walk.",
                ),
        )
        .arg(
            Arg::new("acceptance_beta")
                .long("acceptance-beta")
                .required(false)
                .allow_hyphen_values(true)
                .value_parser(value_parser!(f64))
                .help(
                    "Tilt strength for '--accept-rule linear' and '--accept-rule \
                    exponential'. Must be positive for linear and non-negative \
                    for exponential. Defaults to 1.0.",
                ),
        )
        .arg(common::sum_cols_arg())
        .arg(common::partial_sum_cols_arg())
        .arg(common::edge_weight_keys_arg())
        .arg(common::config_optional(common::objective_arg()))
        .arg(common::region_weights_arg())
        .arg(common::maximize_arg())
        .arg(common::optimizer_variant_arg())
        .arg(
            Arg::new("writer")
                .long("writer")
                .value_parser(value_parser!(String))
                .default_value("assignments")
                .help(
                    "Writer for chain records when --output-file is provided.\n\
                    \tassignments (default): TXT output with only assignment vectors\n\
                    \tcanonicalized-assignments: TXT output with canonicalized assignment vectors\n\
                    \tcanonical: Standardized JSONL output with assignment vector and sample number\n\
                    \tjsonl: JSON Lines with basic summary statistics\n\
                    \tjsonl-full: JSON Lines with basic summary statistics and recombined nodes\n\
                    \ttsv: Tab-separated proposal statistics\n\
                    \tpcompress: Compressed binary format for post-processing with pcompress\n\
                    \tben: Compressed binary format for post-processing with BEN\n\
                    \tbendl: Self-describing BENDL file (graph + metadata + BEN stream in one \
                        file); suppresses the _metadata.jsonl sidecar",
                ),
        )
        .arg(common::bendl_graph_order_arg())
        .arg(common::optimizer_output_file_arg())
        .arg(
            Arg::new("scores-output-file")
                .long("scores-output-file")
                .help(
                    "Path to write per-step objective scores as CSV with step, score, and per-district score columns.",
                ),
        )
        .arg(
            Arg::new("write-improved-scores-only")
                .long("write-improved-scores-only")
                .action(ArgAction::SetTrue)
                .help(
                    "When set, the score writer records only rows that improve the global best \
                    objective score. Has no effect without --scores-output-file.",
                ),
        )
        .arg(common::overwrite_output_arg())
        .arg(common::show_progress_arg())
}

enum AcceptanceConfig {
    Fixed(FixedAcceptance),
    Linear(LinearAcceptance),
    Exponential(ExponentialAcceptance),
}

/// The run's arguments, resolved from either the CLI or a config document.
struct ResolvedTiltedArgs {
    n_steps: u64,
    n_threads: usize,
    rng_seed: u64,
    tol: f64,
    accept_rule: String,
    accept_worse_prob: Option<f64>,
    acceptance_beta: Option<f64>,
    maximize: bool,
    variant: String,
    writer: String,
    show_progress: bool,
    write_improved_scores_only: bool,
    output_file: Option<String>,
    scores_output_file: Option<String>,
    bendl_graph_order: String,
    objective_config: String,
    inputs: common::OptimizerInputs,
    raw_config: Option<String>,
}

impl ResolvedTiltedArgs {
    fn from_cli(matches: &ArgMatches) -> Self {
        Self {
            n_steps: *matches
                .get_one::<u64>("n_steps")
                .expect("n_steps is required"),
            n_threads: *matches
                .get_one::<usize>("n_threads")
                .expect("n_threads has a default value"),
            rng_seed: *matches
                .get_one::<u64>("rng_seed")
                .expect("rng_seed is required"),
            tol: *matches.get_one::<f64>("tol").expect("tol is required"),
            accept_rule: matches
                .get_one::<String>("accept_rule")
                .expect("accept_rule has a default value")
                .clone(),
            accept_worse_prob: matches.get_one::<f64>("accept_worse_prob").copied(),
            acceptance_beta: matches.get_one::<f64>("acceptance_beta").copied(),
            maximize: *matches
                .get_one::<bool>("maximize")
                .expect("maximize has a default value"),
            variant: matches
                .get_one::<String>("variant")
                .expect("variant has a default value")
                .clone(),
            writer: matches
                .get_one::<String>("writer")
                .expect("writer has a default value")
                .clone(),
            show_progress: matches.get_flag("show-progress"),
            write_improved_scores_only: matches.get_flag("write-improved-scores-only"),
            output_file: matches.get_one::<String>("output-file").cloned(),
            scores_output_file: matches.get_one::<String>("scores-output-file").cloned(),
            bendl_graph_order: matches
                .get_one::<String>("bendl_graph_order")
                .expect("bendl_graph_order has a default value")
                .clone(),
            objective_config: common::load_json_arg(
                matches
                    .get_one::<String>("objective")
                    .expect("objective is required"),
                "objective file",
            ),
            inputs: common::parse_optimizer_inputs(matches),
            raw_config: None,
        }
    }

    fn from_config(loaded: rustrecom::config::LoadedTiltedConfig) -> Self {
        let document = loaded.document;
        let inputs = common::OptimizerInputs::from_config_values(
            &document.graph_json,
            document.pop_col,
            document.assignment_col,
            document.sum_cols,
            document.partial_sum_cols,
            document.edge_weight_keys,
            region_weights_from_map(&document.region_weights),
        );
        Self {
            n_steps: document.n_steps,
            n_threads: document.n_threads,
            rng_seed: document.rng_seed,
            tol: document.tol,
            accept_rule: document.accept_rule,
            accept_worse_prob: document.accept_worse_prob,
            acceptance_beta: document.acceptance_beta,
            maximize: document.maximize,
            variant: document.variant,
            writer: document.writer,
            show_progress: document.show_progress,
            write_improved_scores_only: document.write_improved_scores_only,
            output_file: document.output_file,
            scores_output_file: document.scores_output_file,
            bendl_graph_order: document.bendl_graph_order,
            objective_config: document.objective.to_string(),
            inputs,
            raw_config: Some(loaded.raw),
        }
    }
}

pub fn run(matches: &ArgMatches) -> Result<(), String> {
    let overwrite_output = matches.get_flag("overwrite-output");
    // Mixed CLI arguments are rejected before anything is loaded.
    let resolved = match common::resolve_config_argument(matches) {
        Some(raw) => {
            let loaded =
                parse_tilted_config(&raw).unwrap_or_else(|error| panic!("Config error: {error}"));
            ResolvedTiltedArgs::from_config(loaded)
        }
        None => ResolvedTiltedArgs::from_cli(matches),
    };
    let ResolvedTiltedArgs {
        n_steps,
        n_threads,
        rng_seed,
        tol,
        accept_rule,
        accept_worse_prob,
        acceptance_beta,
        maximize,
        variant,
        writer,
        show_progress,
        write_improved_scores_only,
        output_file,
        scores_output_file,
        bendl_graph_order,
        objective_config,
        mut inputs,
        raw_config,
    } = resolved;
    let accept_rule_str = accept_rule.as_str();
    let variant_str = variant.as_str();
    let writer_str = writer.as_str();

    // Intentional change: reject reversible at the CLI layer, before any output
    // file is created or truncated. The engine guard remains and this reuses
    // its message, so the two layers cannot drift.
    if variant_str == "reversible" {
        return Err(REVERSIBLE_UNSUPPORTED.to_string());
    }

    // BENDL needs a seekable file and embeds its provenance in the bundle's
    // Metadata asset, so the separate _metadata.jsonl sidecar is suppressed.
    let (is_bendl, bendl_order) =
        common::resolve_bendl_options(writer_str, &bendl_graph_order, output_file.is_some());
    let metadata_path = common::plan_optimizer_output_paths(
        output_file.as_deref(),
        scores_output_file.as_deref(),
        is_bendl,
        overwrite_output,
    );

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }
    if n_threads == 0 {
        panic!("Parameter error: '--n-threads' must be at least 1.");
    }
    let accept_config = match accept_rule_str {
        "fixed" => {
            let prob = accept_worse_prob.expect(
                "Parameter error: '--accept-worse-prob' is required when '--accept-rule' is 'fixed'.",
            );
            if prob < 0.0 || prob > 1.0 {
                panic!("Parameter error: '--accept-worse-prob' must be between 0 and 1.");
            }
            if acceptance_beta.is_some() {
                panic!(
                    "Parameter error: '--acceptance-beta' is only valid with '--accept-rule linear' \
                     or '--accept-rule exponential'."
                );
            }
            AcceptanceConfig::Fixed(FixedAcceptance { prob })
        }
        "linear" => {
            if accept_worse_prob.is_some() {
                panic!(
                    "Parameter error: '--accept-worse-prob' is only valid with '--accept-rule fixed'."
                );
            }
            let beta = acceptance_beta.unwrap_or(1.0);
            if !beta.is_finite() || beta <= 0.0 {
                panic!("Parameter error: '--acceptance-beta' must be a finite positive number.");
            }
            AcceptanceConfig::Linear(LinearAcceptance { beta })
        }
        "exponential" => {
            let beta = acceptance_beta.unwrap_or(1.0);
            if !beta.is_finite() || beta < 0.0 {
                panic!(
                    "Parameter error: '--acceptance-beta' must be a finite non-negative number."
                );
            }
            if accept_worse_prob.is_some() {
                panic!(
                    "Parameter error: '--accept-worse-prob' is only valid with '--accept-rule fixed'."
                );
            }
            AcceptanceConfig::Exponential(ExponentialAcceptance { beta })
        }
        // Reachable only from config mode; clap restricts the CLI's value set.
        bad => panic!("Parameter error: unknown acceptance rule '{bad}'."),
    };

    let (objective_config, objective, edge_cols) =
        common::prepare_objective_config(objective_config, &mut inputs);

    let loaded = common::load_graph_with_provenance(
        &inputs.graph_json,
        &inputs.pop_col,
        &inputs.assignment_col,
        inputs.sum_cols.clone(),
        inputs.partial_cols.clone(),
        edge_cols,
        is_bendl,
        &bendl_order,
    );
    let common::LoadedGraph {
        mut graph,
        partition,
        embed_bytes,
        source_graph_sha3,
        embedded_graph_sha3,
    } = loaded;
    if let Some((perim_col, boundary_perim_col, shared_perim_col)) =
        polsby_popper_autoderive(&objective_config)
    {
        ensure_derived_perim_column(
            &mut graph,
            &perim_col,
            &boundary_perim_col,
            &shared_perim_col,
        );
    }
    objective.cache_graph_cols(&mut graph);
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let variant =
        common::optimizer_variant(variant_str, &inputs.region_weights, REVERSIBLE_UNSUPPORTED);

    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).ceil() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).floor() as u32,
        num_steps: n_steps,
        rng_seed,
        balance_ub: 0,
        variant,
        region_weights: inputs.region_weights.clone(),
        edge_weight_keys: inputs.edge_weight_keys.clone(),
    };

    let mut meta = json!({
        "assignment_col": inputs.assignment_col,
        "tol": tol,
        "pop_col": inputs.pop_col,
        "graph_path": inputs.graph_json,
        "graph_sha3": source_graph_sha3,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "type": "tilted_run",
        "accept_rule": accept_rule_str,
        "maximize": maximize,
        "overwrite_output": overwrite_output,
        "show_progress": show_progress,
        "write_improved_scores_only": write_improved_scores_only,
        "graph_json": inputs.graph_json,
    });
    // Emit only the parameter relevant to the active acceptance rule.
    match accept_rule_str {
        "fixed" => {
            if let Some(prob) = accept_worse_prob {
                meta.as_object_mut()
                    .unwrap()
                    .insert("accept_worse_prob".to_string(), json!(prob));
            }
        }
        "linear" | "exponential" => {
            if let Some(beta) = acceptance_beta {
                meta.as_object_mut()
                    .unwrap()
                    .insert("acceptance_beta".to_string(), json!(beta));
            }
        }
        _ => {}
    }
    if let Some(path) = &output_file {
        meta.as_object_mut()
            .unwrap()
            .insert("output_file".to_string(), json!(path));
        meta.as_object_mut()
            .unwrap()
            .insert("writer".to_string(), json!(writer_str));
    }
    if let Some(path) = &scores_output_file {
        meta.as_object_mut()
            .unwrap()
            .insert("scores_output_file".to_string(), json!(path));
    }
    if let Some(path) = &metadata_path {
        meta.as_object_mut()
            .unwrap()
            .insert("metadata_file".to_string(), json!(path));
    }
    if inputs.region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(inputs.region_weights));
    }
    if is_bendl {
        common::enrich_bendl_meta(
            &mut meta,
            &source_graph_sha3,
            &embedded_graph_sha3,
            &bendl_order,
        );
    }

    let mut writers = common::build_optimizer_writers(
        output_file.as_deref(),
        scores_output_file.as_deref(),
        is_bendl,
        writer_str,
        overwrite_output,
        &metadata_path,
        &meta,
        embed_bytes,
        raw_config.as_deref(),
    );

    let backend = IncrementalBackend { objective };
    let output = match accept_config {
        AcceptanceConfig::Fixed(rule) => multi_tilted_runs_with_writer(
            &graph,
            partition,
            &params,
            n_threads,
            backend,
            rule,
            maximize,
            writers
                .stats
                .as_mut()
                .map(|writer| &mut **writer as &mut dyn rustrecom::stats::StatsWriter),
            writers.scores.as_mut(),
            show_progress,
            write_improved_scores_only,
        ),
        AcceptanceConfig::Linear(rule) => multi_tilted_runs_with_writer(
            &graph,
            partition,
            &params,
            n_threads,
            backend,
            rule,
            maximize,
            writers
                .stats
                .as_mut()
                .map(|writer| &mut **writer as &mut dyn rustrecom::stats::StatsWriter),
            writers.scores.as_mut(),
            show_progress,
            write_improved_scores_only,
        ),
        AcceptanceConfig::Exponential(rule) => multi_tilted_runs_with_writer(
            &graph,
            partition,
            &params,
            n_threads,
            backend,
            rule,
            maximize,
            writers
                .stats
                .as_mut()
                .map(|writer| &mut **writer as &mut dyn rustrecom::stats::StatsWriter),
            writers.scores.as_mut(),
            show_progress,
            write_improved_scores_only,
        ),
    };
    output.map(|_| ())
}
