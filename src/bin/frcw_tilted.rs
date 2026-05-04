//! Tilted run optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, ArgAction, Command};
use frcw::config::parse_region_weights_config;
use frcw::init::from_networkx;
use frcw::objectives::{
    ensure_derived_perim_column, make_objective, partial_node_cols, polsby_popper_autoderive,
    required_edge_cols, required_node_cols,
};
use frcw::recom::tilted::{
    multi_tilted_runs_with_writer, FixedAcceptance, IncrementalBackend, MetropolisAcceptance,
};
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{
    AssignmentsOnlyWriter, BenWriter, CanonicalWriter, JSONLWriter, PcompressWriter, ScoresWriter,
    StatsWriter, TSVWriter,
};
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::path::{Path, PathBuf};
use std::{fs, io};

fn assert_can_write_output(path: &Path, overwrite_output: bool) {
    if path.exists() && !overwrite_output {
        panic!("Output file already exists. Use --overwrite-output to replace it.");
    };
}

const OUTPUT_BUFFER_CAPACITY: usize = 128 * 1024;

fn output_buffer(path: &str, overwrite_output: bool) -> Box<dyn io::Write + Send> {
    let path = std::path::Path::new(path);
    assert_can_write_output(path, overwrite_output);
    Box::new(io::BufWriter::with_capacity(
        OUTPUT_BUFFER_CAPACITY,
        fs::File::create(path).unwrap(),
    ))
}

fn metadata_path(output_path: &str) -> PathBuf {
    let output_path = PathBuf::from(output_path);
    let file_stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("Output path must have a valid UTF-8 file name.");
    output_path.with_file_name(format!("{}_metadata.jsonl", file_stem))
}

/// Resolves the value of `--objective` to a JSON string.
/// Values that start with `{` are used as-is (inline JSON).
/// All other values are treated as file paths and the file contents are returned.
fn load_objective_config(arg: &str) -> String {
    if arg.trim_start().starts_with('{') {
        arg.to_string()
    } else {
        fs::read_to_string(arg)
            .unwrap_or_else(|e| panic!("Could not read objective file '{}': {}", arg, e))
    }
}

fn make_stats_writer(
    writer_str: &str,
    output_buffer: Box<dyn io::Write + Send>,
) -> Box<dyn StatsWriter> {
    match writer_str {
        "tsv" => Box::new(TSVWriter::new(output_buffer)),
        "jsonl" => Box::new(JSONLWriter::new(false, false, false, output_buffer)),
        "jsonl-full" => Box::new(JSONLWriter::new(true, false, false, output_buffer)),
        "pcompress" => Box::new(PcompressWriter::new(output_buffer)),
        "assignments" => Box::new(AssignmentsOnlyWriter::new(false, output_buffer)),
        "canonicalized-assignments" => Box::new(AssignmentsOnlyWriter::new(true, output_buffer)),
        "canonical" => Box::new(CanonicalWriter::new(output_buffer)),
        "ben" => Box::new(BenWriter::new(output_buffer)),
        bad => panic!("Parameter error: invalid writer '{}'", bad),
    }
}

fn main() {
    let cli = Command::new("frcw_tilted")
        .version("0.1.3")
        .author("Peter Rock <peter@mggg.org>")
        .about("A tilted run optimizer for redistricting")
        .arg(
            Arg::new("graph_json")
                .long("graph-json")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The path of the dual graph (in NetworkX format)."),
        )
        .arg(
            Arg::new("n_steps")
                .long("n-steps")
                .required(true)
                .value_parser(value_parser!(u64))
                .help("The total number of chain steps (accepted + rejected)."),
        )
        .arg(
            Arg::new("tol")
                .long("tol")
                .required(true)
                .value_parser(value_parser!(f64))
                .help("The relative population tolerance."),
        )
        .arg(
            Arg::new("pop_col")
                .long("pop-col")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The name of the total population column in the graph metadata."),
        )
        .arg(
            Arg::new("assignment_col")
                .long("assignment-col")
                .required(true)
                .value_parser(value_parser!(String))
                .help("The name of the assignment column in the graph metadata."),
        )
        .arg(
            Arg::new("rng_seed")
                .long("rng-seed")
                .required(true)
                .value_parser(value_parser!(u64))
                .help("The seed of the RNG used to draw proposals."),
        )
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
                .value_parser(["fixed", "metropolis"])
                .default_value("fixed")
                .help(
                    "Acceptance rule for proposals with a worse score. \
                    'fixed' accepts with constant probability '--accept-worse-prob'. \
                    'metropolis' accepts with probability exp(beta * delta) using \
                    '--metropolis-beta'.",
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
            Arg::new("metropolis_beta")
                .long("metropolis-beta")
                .required(false)
                .value_parser(value_parser!(f64))
                .help(
                    "Inverse temperature for '--accept-rule metropolis'. Must be \
                    non-negative. 0.0 reduces to a random walk; larger values are \
                    pickier about how much worse a proposal can be.",
                ),
        )
        .arg(
            Arg::new("sum_cols")
                .long("sum-cols")
                .value_parser(value_parser!(Option<String>))
                .num_args(1..)
                .default_value(None)
                .help("Additional columns in the graph metadata to sum over districts."),
        )
        .arg(
            Arg::new("partial_sum_cols")
                .long("partial-sum-cols")
                .value_parser(value_parser!(Option<String>))
                .num_args(1..)
                .default_value(None)
                .help(
                    "Additional sum columns that may be missing on some nodes. \
                    Missing entries are treated as zero instead of causing a load-time panic. \
                    Use this for attributes that legitimately apply to only a subset of nodes.",
                ),
        )
        .arg(
            Arg::new("objective")
                .long("objective")
                .required(true)
                .help(
                    "Objective function configuration. Either a JSON string \
                     (must start with '{') or a path to a JSON file.",
                ),
        )
        .arg(
            Arg::new("region_weights")
                .long("region-weights")
                .value_parser(value_parser!(String))
                .default_value("")
                .help(
                    "Region columns with weights for region-aware ReCom. \
                    Must be entered into the command line using the format:\n\
                    \t'{\"region_col1\": weight1, \"region_col2\": weight2, ...}'",
                ),
        )
        .arg(
            Arg::new("maximize")
                .long("maximize")
                .value_parser(value_parser!(bool))
                .default_value("true")
                .help("If true, maximize the objective. If false, minimize it."),
        )
        .arg(
            Arg::new("variant")
                .long("variant")
                .value_parser(value_parser!(String))
                .default_value("district-pairs-rmst")
                .help(
                    "The variant of the ReCom proposal to use.\n\
                    \tcut-edges-rmst (ReCom-A): sample district pairs by selecting one of the cut \
                        edges of the previous plan unifromly at random. Sample using minimum \
                        spanning trees.\n\
                    \tdistrict-pairs-rmst (ReCom-B, default): sample pairs of districts uniformly \
                        at random from the space of all possible pairings. Sample using minimum \
                        spanning trees.\n\
                    \tcut-edges-ust (ReCom-C): sample district pairs by selecting one of the cut \
                        edges of the previous plan unifromly at random. Sample using uniform \
                        spanning trees. \n\
                    \tdistrict-pairs-ust (ReCom-D): sample pairs of districts uniformly at random \
                        from the space of all possible pairings. Sample using uniform spanning \
                        trees.\n\
                    \treversible (RevReCom): Run a reversible recom chain",
                ),
        )
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
                    \tben: Compressed binary format for post-processing with BEN",
                ),
        )
        .arg(
            Arg::new("output-file")
                .long("output-file")
                .short('o')
                .help(
                    "Path to write chain records. If omitted, chain records are not written.",
                ),
        )
        .arg(
            Arg::new("scores-output-file")
                .long("scores-output-file")
                .help(
                    "Path to write per-step objective scores as CSV with step, score, and best_score.",
                ),
        )
        .arg(
            Arg::new("overwrite-output")
                .long("overwrite-output")
                .action(ArgAction::SetTrue)
                .help("Overwrite existing output files instead of failing."),
        )
        .arg(
            Arg::new("show-progress")
                .long("show-progress")
                .action(ArgAction::SetTrue)
                .help("Whether to show a progress bar during execution."),
        );

    let matches = cli.get_matches();

    let n_steps = *matches
        .get_one::<u64>("n_steps")
        .expect("n_steps is required");
    let n_threads = *matches
        .get_one::<usize>("n_threads")
        .expect("n_threads is required");
    let rng_seed = *matches
        .get_one::<u64>("rng_seed")
        .expect("rng_seed is required");
    let tol = *matches.get_one::<f64>("tol").expect("tol is required");
    let accept_rule_str = matches
        .get_one::<String>("accept_rule")
        .expect("accept_rule has a default value")
        .as_str();
    let accept_worse_prob = matches.get_one::<f64>("accept_worse_prob").copied();
    let metropolis_beta = matches.get_one::<f64>("metropolis_beta").copied();

    let maximize = *matches
        .get_one::<bool>("maximize")
        .expect("maximize is required");
    let writer_str = matches
        .get_one::<String>("writer")
        .expect("writer has a default value")
        .as_str();
    let overwrite_output = matches.get_flag("overwrite-output");
    let show_progress = matches.get_flag("show-progress");
    let metadata_base_path = matches
        .get_one::<String>("output-file")
        .or_else(|| matches.get_one::<String>("scores-output-file"));
    let metadata_path = metadata_base_path.map(|path| metadata_path(path));
    if let (Some(output_path), Some(scores_path)) = (
        matches.get_one::<String>("output-file"),
        matches.get_one::<String>("scores-output-file"),
    ) {
        if PathBuf::from(output_path) == PathBuf::from(scores_path) {
            panic!(
                "Parameter error: '--output-file' and '--scores-output-file' must be different."
            );
        }
    }
    if let Some(metadata_path) = &metadata_path {
        if matches
            .get_one::<String>("output-file")
            .is_some_and(|path| PathBuf::from(path) == *metadata_path)
            || matches
                .get_one::<String>("scores-output-file")
                .is_some_and(|path| PathBuf::from(path) == *metadata_path)
        {
            panic!("Parameter error: derived metadata path conflicts with an output path.");
        }
    }
    let output_paths = [
        matches.get_one::<String>("output-file").map(PathBuf::from),
        matches
            .get_one::<String>("scores-output-file")
            .map(PathBuf::from),
        metadata_path.clone(),
    ];
    for path in output_paths.into_iter().flatten() {
        assert_can_write_output(&path, overwrite_output);
    }

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }
    if n_threads == 0 {
        panic!("Parameter error: '--n-threads' must be at least 1.");
    }
    enum AcceptanceConfig {
        Fixed(FixedAcceptance),
        Metropolis(MetropolisAcceptance),
    }
    let accept_config = match accept_rule_str {
        "fixed" => {
            let prob = accept_worse_prob.expect(
                "Parameter error: '--accept-worse-prob' is required when '--accept-rule' is 'fixed'.",
            );
            if prob < 0.0 || prob > 1.0 {
                panic!("Parameter error: '--accept-worse-prob' must be between 0 and 1.");
            }
            if metropolis_beta.is_some() {
                panic!(
                    "Parameter error: '--metropolis-beta' is only valid with '--accept-rule metropolis'."
                );
            }
            AcceptanceConfig::Fixed(FixedAcceptance { prob })
        }
        "metropolis" => {
            let beta = metropolis_beta.expect(
                "Parameter error: '--metropolis-beta' is required when '--accept-rule' is 'metropolis'.",
            );
            if !beta.is_finite() || beta < 0.0 {
                panic!("Parameter error: '--metropolis-beta' must be a finite non-negative number.");
            }
            if accept_worse_prob.is_some() {
                panic!(
                    "Parameter error: '--accept-worse-prob' is only valid with '--accept-rule fixed'."
                );
            }
            AcceptanceConfig::Metropolis(MetropolisAcceptance { beta })
        }
        other => panic!("Parameter error: unknown acceptance rule '{other}'."),
    };

    let graph_path = matches
        .get_one::<String>("graph_json")
        .expect("graph_json is required");
    let graph_path_buf = PathBuf::from(graph_path);
    let graph_json = match fs::canonicalize(&graph_path_buf)
        .map_err(|e| format!("Could not canonicalize path {:?}: {e}", graph_path_buf))
        .expect(format!("Could not create fs buffer from {graph_path}").as_str())
        .into_os_string()
        .into_string()
        .map_err(|_| {
            format!(
                "path for --graph-json is not valid UTF-8: {:?}",
                graph_path_buf
            )
        }) {
        Ok(s) => s,
        Err(e) => panic!("{}", e),
    };

    let pop_col = matches
        .get_one::<String>("pop_col")
        .expect("pop_col is required")
        .as_str();
    let assignment_col = matches
        .get_one::<String>("assignment_col")
        .expect("assignment_col is required")
        .as_str();
    let mut sum_cols: Vec<String> = matches
        .get_many::<String>("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let mut partial_cols: Vec<String> = matches
        .get_many::<String>("partial_sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let region_weights_raw = (*matches.get_one::<String>("region_weights").unwrap()).as_str();
    let region_weights = parse_region_weights_config(region_weights_raw);
    // Add region weight keys to sum_cols so the user doesn't have to specify them twice.
    if let Some(weight_pairs_vec) = &region_weights {
        for (key, _) in weight_pairs_vec.iter() {
            if !sum_cols.contains(&key) {
                sum_cols.push(key.clone().to_string());
            }
        }
    }

    let objective_config = load_objective_config(
        matches
            .get_one::<String>("objective")
            .expect("objective is required"),
    );
    let objective_config = objective_config.as_str();
    let objective = make_objective(objective_config);
    let edge_cols = required_edge_cols(objective_config);
    for col in required_node_cols(objective_config) {
        if !sum_cols.contains(&col) {
            sum_cols.push(col);
        }
    }
    for col in partial_node_cols(objective_config) {
        if !partial_cols.contains(&col) && !sum_cols.contains(&col) {
            partial_cols.push(col);
        }
    }

    let (mut graph, partition) = from_networkx(
        &graph_json,
        pop_col,
        assignment_col,
        sum_cols,
        partial_cols,
        edge_cols,
    )
    .unwrap();
    if let Some((perim_col, boundary_perim_col, shared_perim_col)) =
        polsby_popper_autoderive(objective_config)
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
    let variant = match matches
        .get_one::<String>("variant")
        .expect("variant has a default value")
        .as_str()
    {
        "cut-edges-rmst" => match region_weights {
            None => RecomVariant::CutEdgesRMST,
            Some(_) => RecomVariant::CutEdgesRegionAware,
        },
        "district-pairs-rmst" => match region_weights {
            None => RecomVariant::DistrictPairsRMST,
            Some(_) => RecomVariant::DistrictPairsRegionAware,
        },
        "cut-edges-ust" => match region_weights {
            None => RecomVariant::CutEdgesUST,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for uniform spanning tree sampling.")
            }
        },
        "district-pairs-ust" => match region_weights {
            None => RecomVariant::DistrictPairsUST,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for uniform spanning tree sampling.")
            }
        },
        "reversible" => match region_weights {
            None => RecomVariant::Reversible,
            Some(_) => {
                panic!("Region-aware variants are not currently implemented for reversible recom.")
            }
        },
        bad => panic!("Parameter error: invalid variant '{}'", bad),
    };

    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).ceil() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).floor() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: 0,
        variant,
        region_weights: region_weights.clone(),
    };

    let mut graph_file = fs::File::open(&graph_json).unwrap();
    let mut graph_hasher = Sha3_256::new();
    io::copy(&mut graph_file, &mut graph_hasher).unwrap();
    let graph_hash = format!("{:x}", graph_hasher.finalize());
    let mut meta = json!({
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
        "graph_sha3": graph_hash,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "type": "tilted_run",
        "accept_rule": accept_rule_str,
        "accept_worse_prob": accept_worse_prob,
        "metropolis_beta": metropolis_beta,
        "maximize": maximize,
        "overwrite_output": overwrite_output,
        "show_progress": show_progress,
        "graph_json": graph_json,
    });
    if let Some(path) = matches.get_one::<String>("output-file") {
        meta.as_object_mut()
            .unwrap()
            .insert("output_file".to_string(), json!(path));
        meta.as_object_mut()
            .unwrap()
            .insert("writer".to_string(), json!(writer_str));
    }
    if let Some(path) = matches.get_one::<String>("scores-output-file") {
        meta.as_object_mut()
            .unwrap()
            .insert("scores_output_file".to_string(), json!(path));
    }
    if let Some(path) = &metadata_path {
        meta.as_object_mut()
            .unwrap()
            .insert("metadata_file".to_string(), json!(path));
    }
    if region_weights.is_some() {
        meta.as_object_mut()
            .unwrap()
            .insert("region_weights".to_string(), json!(region_weights));
    }
    let mut metadata_writer: Option<Box<dyn io::Write + Send>> = metadata_path
        .as_ref()
        .map(|path| output_buffer(path.to_str().unwrap(), overwrite_output));
    if let Some(writer) = metadata_writer.as_mut() {
        writeln!(writer, "{}", json!({ "meta": meta })).unwrap();
        writer.flush().unwrap();
    }

    let mut stats_writer: Option<Box<dyn StatsWriter>> =
        match matches.get_one::<String>("output-file") {
            Some(path) => Some(make_stats_writer(
                writer_str,
                output_buffer(path, overwrite_output),
            )),
            None => {
                if writer_str != "assignments" {
                    panic!("Parameter error: '--writer' requires '--output-file'.");
                }
                None
            }
        };
    let mut scores_writer: Option<ScoresWriter> = matches
        .get_one::<String>("scores-output-file")
        .map(|path| ScoresWriter::new(output_buffer(path, overwrite_output)));

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
            stats_writer
                .as_mut()
                .map(|writer| &mut **writer as &mut dyn StatsWriter),
            scores_writer.as_mut(),
            show_progress,
        ),
        AcceptanceConfig::Metropolis(rule) => multi_tilted_runs_with_writer(
            &graph,
            partition,
            &params,
            n_threads,
            backend,
            rule,
            maximize,
            stats_writer
                .as_mut()
                .map(|writer| &mut **writer as &mut dyn StatsWriter),
            scores_writer.as_mut(),
            show_progress,
        ),
    };

    match output {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error during optimization: {}", e);
        }
    }
}
