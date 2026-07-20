//! The `short-bursts` subcommand: a short bursts optimizer for redistricting
//! (formerly the `frcw_short_bursts` binary).
//!
//! Arguments arrive either from CLI flags or, via `--config`, from a versioned
//! JSON config whose fields mirror those flags. Config mode preserves the
//! exact raw config string as provenance: it becomes the metadata sidecar's
//! content (or the BENDL Metadata asset).

use crate::common;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use rustrecom::config::{parse_short_bursts_config, region_weights_from_map};
use rustrecom::objectives::{ensure_derived_perim_column, polsby_popper_autoderive};
use rustrecom::recom::short_bursts::{
    core::REVERSIBLE_UNSUPPORTED, multi_short_bursts_with_writer,
};
use rustrecom::recom::{IncrementalBackend, RecomParams};
use serde_json::json;

pub fn command() -> Command {
    Command::new("short-bursts")
        .about("A short bursts optimizer for redistricting")
        .arg(common::config_arg())
        .arg(common::config_optional(common::graph_json_arg()))
        .arg(common::config_optional(
            Arg::new("n_steps")
                .long("n-steps")
                .value_parser(value_parser!(u64))
                .help("The number of proposals to generate."),
        ))
        .arg(common::config_optional(common::tol_arg()))
        .arg(common::config_optional(common::pop_col_arg()))
        .arg(common::config_optional(common::assignment_col_arg()))
        .arg(common::config_optional(common::rng_seed_arg()))
        .arg(common::n_threads_arg())
        .arg(common::config_optional(
            Arg::new("burst_length")
                .long("burst-length")
                .value_parser(value_parser!(usize))
                .help("The number of accepted steps per short burst."),
        ))
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
                        file); suppresses the _metadata.jsonl sidecar\n\
                    Note: short bursts workers return full partitions, not individual proposals.\n\
                    Writers that require proposal-level data (tsv, jsonl, pcompress) will have\n\
                    empty proposal fields. Prefer assignments, canonical, ben, or bendl.",
                ),
        )
        .arg(common::bendl_graph_order_arg())
        .arg(common::optimizer_output_file_arg())
        .arg(
            Arg::new("scores-output-file")
                .long("scores-output-file")
                .help(
                    "Path to write per-burst objective scores as CSV with step, score, and per-district score columns.",
                ),
        )
        .arg(common::overwrite_output_arg())
        .arg(common::show_progress_arg())
        .arg(
            Arg::new("write-improved-scores-only")
                .long("write-improved-scores-only")
                .action(ArgAction::SetTrue)
                .help(
                    "When set, the scores writer records only rows that improve the global \
                    best objective score. Has no effect without --scores-output-file.",
                ),
        )
}

/// The run's arguments, resolved from either the CLI or a config document.
struct ResolvedShortBurstsArgs {
    n_steps: u64,
    n_threads: usize,
    rng_seed: u64,
    tol: f64,
    burst_length: usize,
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

impl ResolvedShortBurstsArgs {
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
            burst_length: *matches
                .get_one::<usize>("burst_length")
                .expect("burst_length is required"),
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

    fn from_config(loaded: rustrecom::config::LoadedShortBurstsConfig) -> Self {
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
            burst_length: document.burst_length,
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
            let loaded = parse_short_bursts_config(&raw)
                .unwrap_or_else(|error| panic!("Config error: {error}"));
            ResolvedShortBurstsArgs::from_config(loaded)
        }
        None => ResolvedShortBurstsArgs::from_cli(matches),
    };
    let ResolvedShortBurstsArgs {
        n_steps,
        n_threads,
        rng_seed,
        tol,
        burst_length,
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
    let variant_str = variant.as_str();
    let writer_str = writer.as_str();

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
        "type": "short_bursts",
        "burst_length": burst_length,
        "maximize": maximize,
        "variant": variant_str,
        "overwrite_output": overwrite_output,
        "show_progress": show_progress,
        "write_improved_scores_only": write_improved_scores_only,
        "graph_json": inputs.graph_json,
    });
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
    multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        backend,
        maximize,
        burst_length,
        writers
            .stats
            .as_mut()
            .map(|writer| &mut **writer as &mut dyn rustrecom::stats::StatsWriter),
        writers.scores.as_mut(),
        show_progress,
        write_improved_scores_only,
    )
    .map(|_| ())
}
