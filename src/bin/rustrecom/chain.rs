//! The `chain` subcommand: a minimal implementation of the ReCom Markov chain
//! (formerly the bare `frcw` binary).

use crate::common;
use clap::{value_parser, Arg, ArgAction, ArgMatches, Command};
use frcw::constraints::{make_constraint, ConstraintConfig};
use frcw::recom::run::multi_chain_with_constraint;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{BendlBenStreamWriter, StatsWriter};
use serde_json::json;
use std::io;

pub fn command() -> Command {
    let mut cli =
        Command::new("chain")
            .about("A minimal implementation of the ReCom Markov chain")
            .arg(common::graph_json_arg())
            .arg(
                Arg::new("n_steps")
                    .long("n-steps")
                    .required(true)
                    .value_parser(value_parser!(u64))
                    .help("The number of proposals to generate."),
            )
            .arg(
                Arg::new("target_pop")
                    .long("target-pop")
                    .value_parser(value_parser!(u64))
                    .help("The target population for the districts."),
            )
            .arg(common::tol_arg())
            .arg(common::pop_col_arg())
            .arg(common::assignment_col_arg())
            .arg(common::rng_seed_arg())
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
            .arg(
                Arg::new("variant")
                    .long("variant")
                    .required(true)
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
            )
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
                \ttsv: Tab-separated assignment vectors\n\
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

pub fn run(matches: &ArgMatches) -> Result<(), String> {
    let n_steps = *matches
        .get_one::<u64>("n_steps")
        .expect("n_steps is required");
    let rng_seed = *matches
        .get_one::<u64>("rng_seed")
        .expect("rng_seed is required");
    let target_pop_opt: Option<u64> = matches.get_one::<u64>("target_pop").copied();
    let tol = *matches.get_one::<f64>("tol").expect("tol is required");
    let balance_ub = *matches
        .get_one::<u32>("balance_ub")
        .expect("balance_ub has a default value");
    let n_threads = *matches
        .get_one::<usize>("n_threads")
        .expect("n_threads is required");
    let batch_size = *matches
        .get_one::<usize>("batch_size")
        .expect("batch_size is required");

    let graph_path = matches
        .get_one::<String>("graph_json")
        .expect("graph_json is required");
    let graph_json = common::canonicalize_graph_path(graph_path);

    let pop_col = matches
        .get_one::<String>("pop_col")
        .expect("pop_col is required")
        .as_str();
    let assignment_col = matches
        .get_one::<String>("assignment_col")
        .expect("assignment_col is required")
        .as_str();
    let variant_str = matches
        .get_one::<String>("variant")
        .expect("variant has a default value")
        .as_str();
    let writer_str = matches
        .get_one::<String>("writer")
        .expect("writer has a default value")
        .as_str();
    let overwrite_output = matches.get_flag("overwrite-output");

    let st_counts = if cfg!(feature = "linalg") {
        matches.get_flag("spanning_tree_counts")
    } else {
        false
    };
    let cut_edges_count = matches.get_flag("cut_edges_count");
    let mut sum_cols: Vec<String> = matches
        .get_many::<String>("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let edge_weight_keys: Vec<String> = matches
        .get_many::<String>("edge_weight_keys")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let region_weights_raw = (*matches.get_one::<String>("region_weights").unwrap()).as_str();
    let region_weights = frcw::config::parse_region_weights_config(region_weights_raw);
    let constraint_json = matches
        .get_one::<String>("constraint")
        .filter(|arg| !arg.is_empty())
        .map(|arg| common::load_json_arg(arg, "JSON config"));
    let constraint = constraint_json
        .as_deref()
        .map(make_constraint)
        .unwrap_or(ConstraintConfig::None);

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
    let (is_bendl, bendl_order) = common::resolve_bendl_options(matches, writer_str);
    let output_file = matches.get_one::<String>("output-file").cloned();

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
    if let Some(config) = &constraint_json {
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
    if writer_str == "jsonl" || writer_str == "jsonl-full" {
        // hotfix for pcompress writing
        // TODO: move this into init
        println!("{}", json!({ "meta": meta }).to_string());
    }

    // Build the output writer. The bendl arm embeds the provenance-enriched
    // metadata and writes to a concrete seekable file; all others use the boxed
    // sink (file or stdout).
    let writer: Box<dyn StatsWriter> = if is_bendl {
        let path = output_file.as_ref().expect("bendl requires --output-file");
        let bundle_file = common::bendl_output_file(path, overwrite_output);
        Box::new(BendlBenStreamWriter::new(
            bundle_file,
            embed_bytes.expect("bendl computes the embed bytes"),
            meta.to_string().into_bytes(),
        ))
    } else {
        let output_buffer: Box<dyn io::Write + Send> = match &output_file {
            Some(path) => common::output_buffer(path, overwrite_output),
            None => Box::new(io::BufWriter::with_capacity(
                common::OUTPUT_BUFFER_CAPACITY,
                std::io::stdout(),
            )),
        };
        common::make_stats_writer(writer_str, st_counts, cut_edges_count, output_buffer)
    };

    let show_progress = matches.get_flag("show-progress");

    multi_chain_with_constraint(
        &graph,
        &partition,
        writer,
        &params,
        n_threads,
        batch_size,
        show_progress,
        constraint,
    )
}
