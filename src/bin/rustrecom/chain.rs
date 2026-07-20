//! The `chain` subcommand: a minimal implementation of the ReCom Markov chain
//! (formerly the bare `frcw` binary).
//!
//! Arguments arrive either from CLI flags or, via `--config`, from a versioned JSON
//! config whose fields mirror those flags. The two sources resolve into one
//! [`ResolvedChainArgs`] so the run path is identical; config mode additionally
//! preserves the exact raw config string as provenance (primary metadata record,
//! BENDL Metadata asset, or a `<stem>_metadata.jsonl` sidecar).

use crate::common;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use rustrecom::config::{parse_chain_config, region_weights_from_map, LoadedChainConfig};
use rustrecom::constraints::{make_constraint, make_constraint_value, ConstraintConfig};
use rustrecom::recom::run::multi_chain_with_constraint;
use rustrecom::recom::{RecomParams, RecomVariant};
use rustrecom::stats::{BendlBenStreamWriter, StatsWriter};
use serde_json::json;
use std::fs;
use std::io::{self, Write};

pub fn command() -> Command {
    let mut cli =
        Command::new("chain")
            .about("A minimal implementation of the ReCom Markov chain")
            .arg(common::config_arg())
            .arg(common::config_optional(common::graph_json_arg()))
            .arg(common::config_optional(
                Arg::new("n_steps")
                    .long("n-steps")
                    .value_parser(value_parser!(u64))
                    .help("The number of proposals to generate."),
            ))
            .arg(
                Arg::new("target_pop")
                    .long("target-pop")
                    .value_parser(value_parser!(u64))
                    .help("The target population for the districts."),
            )
            .arg(common::config_optional(common::tol_arg()))
            .arg(common::config_optional(common::pop_col_arg()))
            .arg(common::config_optional(common::assignment_col_arg()))
            .arg(common::config_optional(common::rng_seed_arg()))
            .arg(
                Arg::new("balance_ub")
                    .long("balance-ub")
                    .short('M') // Variable used in RevReCom paper
                    .value_parser(value_parser!(u32))
                    .default_value("0")
                    .help("The normalizing constant (reversible ReCom only)."),
            )
            .arg(common::n_threads_arg())
            .arg(
                Arg::new("batch_size")
                    .long("batch-size")
                    .required(false)
                    .value_parser(value_parser!(usize))
                    .default_value("1")
                    .help("The number of proposals per batch job."),
            )
            .arg(common::config_optional(
                Arg::new("variant")
                    .long("variant")
                    .value_parser(value_parser!(String))
                    .help(
                        "The ReCom variant to use. The options are\n\
                \tcut-edges-rmst (ReCom-A)\n\
                \tdistrict-pairs-rmst (ReCom-B)\n\
                \tcut-edges-ust (ReCom-C)\n\
                \tdistrict-pairs-ust (ReCom-D)\n\
                \tcut-edges-region-aware (Recom-AW)\n\
                \tdistrict-pairs-region-aware (Recom-BW)\n\
                \treversible (RevReCom)",
                    ),
            ))
            .arg(
                Arg::new("writer")
                    .long("writer")
                    .value_parser(value_parser!(String))
                    .default_value("jsonl")
                    .help(
                        "The output writer to use.\n\
                \tjsonl (default): JSON Lines with basic summary statistics \n\
                    \t\t(no assignment vectors)\n\
                \tjsonl-full: JSON Lines object with basic summary statistics and a \"nodes\"\n\
                    \t\tattribute containing node assignments for recombined pairs\n\
                \ttsv: Tab-separated accepted-proposal and self-loop statistics\n\
                \tpcompress: Compressed binary format for post-processing with pcompress\n\
                    \t\t(old compression format)\n\
                \tassignments: TXT output with only assignment vectors\n\
                \tcanonicalized-assignments: TXT output with canonicalized (increasing order)\n\
                    \t\tassignment vectors\n\
                \tcanonical: Standardized JSONL output with assignment vector and sample\n\
                    \t\tnumber\n\
                \tben: Compressed binary format for post-processing with BEN (recommended for \
                    storing ensembles).\n\
                \tbendl: Self-describing BENDL file (graph + metadata + BEN stream in one \
                    file). Requires --output-file.",
                    ),
            )
            .arg(common::sum_cols_arg())
            .arg(
                Arg::new("constraint")
                    .long("constraint")
                    .required(false)
                    .help("Constraint config as inline JSON or a path to a JSON file."),
            )
            .arg(common::region_weights_arg())
            .arg(common::edge_weight_keys_arg())
            .arg(
                Arg::new("cut_edges_count")
                    .long("cut-edges-count")
                    .action(ArgAction::SetTrue)
                    .help("Whether to compute and output the cut edges count at each step."),
            )
            .arg(Arg::new("output-file").long("output-file").short('o').help(
                "The path to write the output to. If not provided, ouput is printed to console.",
            ))
            .arg(common::overwrite_output_arg())
            .arg(common::show_progress_arg())
            .arg(common::bendl_graph_order_arg());

    if cfg!(feature = "linalg") {
        cli = cli.arg(
            Arg::new("spanning_tree_counts")
                .long("st-counts")
                .action(ArgAction::SetTrue)
                .help("Whether to compute and output the spanning tree counts at each step."),
        );
    }
    cli
}

struct ResolvedChainArgs {
    graph_path: String,
    n_steps: u64,
    target_pop: Option<u64>,
    tol: f64,
    pop_col: String,
    assignment_col: String,
    rng_seed: u64,
    balance_ub: u32,
    n_threads: usize,
    batch_size: usize,
    variant: String,
    writer: String,
    sum_cols: Vec<String>,
    constraint: ConstraintConfig,
    cli_constraint_json: Option<String>,
    region_weights: Option<Vec<(String, f64)>>,
    edge_weight_keys: Vec<String>,
    cut_edges_count: bool,
    output_file: Option<String>,
    bendl_graph_order: String,
    show_progress: bool,
    st_counts: bool,
    raw_config: Option<String>,
}

impl ResolvedChainArgs {
    fn from_cli(matches: &ArgMatches) -> Self {
        let region_weights_raw = matches
            .get_one::<String>("region_weights")
            .expect("region_weights has a default value");
        let cli_constraint_json = matches
            .get_one::<String>("constraint")
            .filter(|arg| !arg.is_empty())
            .map(|arg| common::load_json_arg(arg, "JSON config"));
        let constraint = cli_constraint_json
            .as_deref()
            .map(make_constraint)
            .unwrap_or(ConstraintConfig::None);

        Self {
            graph_path: matches
                .get_one::<String>("graph_json")
                .expect("graph_json is required")
                .clone(),
            n_steps: *matches
                .get_one::<u64>("n_steps")
                .expect("n_steps is required"),
            target_pop: matches.get_one::<u64>("target_pop").copied(),
            tol: *matches.get_one::<f64>("tol").expect("tol is required"),
            pop_col: matches
                .get_one::<String>("pop_col")
                .expect("pop_col is required")
                .clone(),
            assignment_col: matches
                .get_one::<String>("assignment_col")
                .expect("assignment_col is required")
                .clone(),
            rng_seed: *matches
                .get_one::<u64>("rng_seed")
                .expect("rng_seed is required"),
            balance_ub: *matches
                .get_one::<u32>("balance_ub")
                .expect("balance_ub has a default value"),
            n_threads: *matches
                .get_one::<usize>("n_threads")
                .expect("n_threads has a default value"),
            batch_size: *matches
                .get_one::<usize>("batch_size")
                .expect("batch_size has a default value"),
            variant: matches
                .get_one::<String>("variant")
                .expect("variant is required")
                .clone(),
            writer: matches
                .get_one::<String>("writer")
                .expect("writer has a default value")
                .clone(),
            sum_cols: matches
                .get_many::<String>("sum_cols")
                .unwrap_or_default()
                .cloned()
                .collect(),
            constraint,
            cli_constraint_json,
            region_weights: rustrecom::config::parse_region_weights_config(region_weights_raw),
            edge_weight_keys: matches
                .get_many::<String>("edge_weight_keys")
                .unwrap_or_default()
                .cloned()
                .collect(),
            cut_edges_count: matches.get_flag("cut_edges_count"),
            output_file: matches.get_one::<String>("output-file").cloned(),
            bendl_graph_order: matches
                .get_one::<String>("bendl_graph_order")
                .expect("bendl_graph_order has a default value")
                .clone(),
            show_progress: matches.get_flag("show-progress"),
            st_counts: if cfg!(feature = "linalg") {
                matches.get_flag("spanning_tree_counts")
            } else {
                false
            },
            raw_config: None,
        }
    }

    fn from_config(loaded: LoadedChainConfig) -> Self {
        let document = loaded.document;
        let constraint = document
            .constraint
            .as_ref()
            .map(make_constraint_value)
            .unwrap_or(ConstraintConfig::None);
        let region_weights = region_weights_from_map(&document.region_weights);

        Self {
            graph_path: document.graph_json,
            n_steps: document.n_steps,
            target_pop: document.target_pop,
            tol: document.tol,
            pop_col: document.pop_col,
            assignment_col: document.assignment_col,
            rng_seed: document.rng_seed,
            balance_ub: document.balance_ub,
            n_threads: document.n_threads,
            batch_size: document.batch_size,
            variant: document.variant,
            writer: document.writer,
            sum_cols: document.sum_cols,
            constraint,
            cli_constraint_json: None,
            region_weights,
            edge_weight_keys: document.edge_weight_keys,
            cut_edges_count: document.cut_edges_count,
            output_file: document.output_file,
            bendl_graph_order: document.bendl_graph_order,
            show_progress: document.show_progress,
            st_counts: false,
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
                parse_chain_config(&raw).unwrap_or_else(|error| panic!("Config error: {error}"));
            ResolvedChainArgs::from_config(loaded)
        }
        None => ResolvedChainArgs::from_cli(matches),
    };
    let ResolvedChainArgs {
        graph_path,
        n_steps,
        target_pop: target_pop_opt,
        tol,
        pop_col,
        assignment_col,
        rng_seed,
        balance_ub,
        n_threads,
        batch_size,
        variant,
        writer,
        mut sum_cols,
        constraint,
        cli_constraint_json,
        region_weights,
        edge_weight_keys,
        cut_edges_count,
        output_file,
        bendl_graph_order,
        show_progress,
        st_counts,
        raw_config,
    } = resolved;

    let graph_json = common::canonicalize_graph_path(&graph_path);
    let pop_col = pop_col.as_str();
    let assignment_col = assignment_col.as_str();
    let variant_str = variant.as_str();
    let writer_str = writer.as_str();

    // When region weights are supplied, transparently upgrade the RMST/cut-edges
    // variants to their region-aware counterparts so the weights actually take
    // effect (the plain RMST sampler ignores them). UST and reversible variants
    // have no region-aware implementation, so reject the combination instead of
    // silently dropping the weights.
    let variant = match variant_str {
        "reversible" => match region_weights {
            None => RecomVariant::Reversible,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for reversible recom.")
            }
        },
        "cut-edges-ust" => match region_weights {
            None => RecomVariant::CutEdgesUST,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for uniform spanning tree sampling.")
            }
        },
        "cut-edges-rmst" => match region_weights {
            None => RecomVariant::CutEdgesRMST,
            Some(_) => RecomVariant::CutEdgesRegionAware,
        },
        "cut-edges-region-aware" => RecomVariant::CutEdgesRegionAware,
        "district-pairs-ust" => match region_weights {
            None => RecomVariant::DistrictPairsUST,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for uniform spanning tree sampling.")
            }
        },
        "district-pairs-rmst" => match region_weights {
            None => RecomVariant::DistrictPairsRMST,
            Some(_) => RecomVariant::DistrictPairsRegionAware,
        },
        "district-pairs-region-aware" => RecomVariant::DistrictPairsRegionAware,
        bad => panic!("Parameter error: invalid variant '{}'", bad),
    };

    // BENDL needs a seekable file and embeds a possibly-reordered graph, so its
    // arm diverges from the boxed-output path. Validate its flags before doing
    // any work or opening the output file.
    let (is_bendl, bendl_order) =
        common::resolve_bendl_options(writer_str, &bendl_graph_order, output_file.is_some());

    // Config mode preserves the exact raw config as provenance. Writers with a
    // primary metadata facility (jsonl records, the BENDL Metadata asset) embed
    // it there; every other writer gets a `<stem>_metadata.jsonl` sidecar next
    // to the output file. Check the sidecar path before any file is opened.
    let metadata_file = if raw_config.is_some()
        && output_file.is_some()
        && !matches!(writer_str, "jsonl" | "jsonl-full" | "bendl")
    {
        let path = common::metadata_path(output_file.as_ref().unwrap());
        common::assert_can_write_output(&path, overwrite_output);
        Some(path)
    } else {
        None
    };

    if variant == RecomVariant::Reversible && balance_ub == 0 {
        panic!("For reversible ReCom, specify M > 0.");
    }

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }

    common::merge_region_weight_cols(&mut sum_cols, &region_weights);
    // Constraint columns must live in `graph.attr` so `cache_graph_cols` can parse
    // them into `graph.float_attr`. Columns the user did not also list in `sum_cols`
    // are constraint-only: load them, but drop them from `graph.attr` after caching
    // (below) so the integer district-sum machinery never tries to sum a fractional
    // share column (`parse_as_int` panics on e.g. "0.25").
    let constraint_only: Vec<String> = constraint
        .required_node_cols()
        .into_iter()
        .filter(|col| !sum_cols.contains(col))
        .collect();
    sum_cols.extend(constraint_only.iter().cloned());

    let loaded = common::load_graph_with_provenance(
        &graph_json,
        pop_col,
        assignment_col,
        sum_cols,
        vec![],
        edge_weight_keys.clone(),
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
    constraint.cache_graph_cols(&mut graph);
    for col in &constraint_only {
        graph.attr.remove(col);
    }

    let target_pop = match target_pop_opt {
        Some(p) => p as f64,
        None => (graph.total_pop as f64) / (partition.num_dists as f64),
    };

    // NOTE: We have to round towards the target_pop here so that a population tolerance
    // of 0.000001 on a graph with a small target population (e.g., 10) does not allow in
    // districts with 9 people in them.
    let params = RecomParams {
        min_pop: ((1.0 - tol) * target_pop as f64).ceil() as u32,
        max_pop: ((1.0 + tol) * target_pop as f64).floor() as u32,
        num_steps: n_steps,
        rng_seed,
        balance_ub,
        variant,
        region_weights: region_weights.clone(),
        edge_weight_keys,
    };

    let mut meta = json!({
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
        "graph_sha3": source_graph_sha3,
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
        "graph_json": graph_json,
        "chain_variant": variant_str,
    });
    if let Some(path) = &output_file {
        meta.as_object_mut()
            .unwrap()
            .insert("output_file".to_string(), json!(path));
        meta.as_object_mut()
            .unwrap()
            .insert("overwrite_output".to_string(), json!(overwrite_output));
    }
    if variant == RecomVariant::Reversible {
        meta.as_object_mut()
            .unwrap()
            .insert("balance_ub".to_string(), json!(balance_ub));
    }
    if region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(region_weights));
    }
    if let Some(config) = &cli_constraint_json {
        meta.as_object_mut()
            .unwrap()
            .insert("constraint".to_string(), json!(config));
    }
    if is_bendl {
        common::enrich_bendl_meta(
            &mut meta,
            &source_graph_sha3,
            &embedded_graph_sha3,
            &bendl_order,
        );
    }
    // CLI mode keeps printing the metadata record to stdout, which is where
    // callers that redirect stdout into their output file expect it. Config mode
    // owns its own output file (there is no redirect to piggyback on), so it
    // writes the record into the output sink instead; see the writer arm below.
    if raw_config.is_none() && (writer_str == "jsonl" || writer_str == "jsonl-full") {
        // hotfix for pcompress writing
        // TODO: move this into init
        println!("{}", json!({ "meta": meta }).to_string());
    }

    // Build the output writer. Config mode stores the exact environment string
    // in the primary metadata facility where one exists, otherwise in a sidecar.
    let writer: Box<dyn StatsWriter> = if is_bendl {
        let path = output_file.as_ref().expect("bendl requires --output-file");
        let bundle_file = common::bendl_output_file(path, overwrite_output);
        let metadata = raw_config
            .as_ref()
            .map(|raw| raw.as_bytes().to_vec())
            .unwrap_or_else(|| meta.to_string().into_bytes());
        Box::new(BendlBenStreamWriter::new(
            bundle_file,
            embed_bytes.expect("bendl computes the embed bytes"),
            metadata,
        ))
    } else {
        let mut output_buffer: Box<dyn io::Write + Send> = match &output_file {
            Some(path) => common::output_buffer(path, overwrite_output),
            None => Box::new(io::BufWriter::with_capacity(
                common::OUTPUT_BUFFER_CAPACITY,
                std::io::stdout(),
            )),
        };
        if let (Some(raw), "jsonl" | "jsonl-full") = (&raw_config, writer_str) {
            let metadata = json!({"meta": {"config": raw}});
            writeln!(output_buffer, "{metadata}").expect("Could not write metadata record");
        }
        common::make_stats_writer(writer_str, st_counts, cut_edges_count, output_buffer)
    };

    multi_chain_with_constraint(
        &graph,
        &partition,
        writer,
        &params,
        n_threads,
        batch_size,
        show_progress,
        constraint,
    )?;
    // The sidecar is written only after a successful run, so a failed run never
    // leaves provenance pointing at a missing or truncated output file.
    if let (Some(path), Some(raw)) = (&metadata_file, &raw_config) {
        fs::write(path, format!("{raw}\n"))
            .unwrap_or_else(|error| panic!("Could not write config metadata: {error}"));
    }
    Ok(())
}
