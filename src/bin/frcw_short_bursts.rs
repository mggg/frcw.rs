//! Short bursts optimization CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, ArgAction, Command};
use frcw::bendl::{reorder_graph_json, BendlGraphOrder};
use frcw::config::parse_region_weights_config;
use frcw::init::{from_networkx, from_networkx_value};
use frcw::objectives::{
    ensure_derived_perim_column, make_objective, partial_node_cols, polsby_popper_autoderive,
    required_edge_cols, required_node_cols,
};
use frcw::recom::short_bursts::multi_short_bursts_with_writer;
use frcw::recom::{IncrementalBackend, RecomParams, RecomVariant};
use frcw::stats::{
    AssignmentsOnlyWriter, BenWriter, BendlBenStreamWriter, CanonicalWriter, JSONLWriter,
    PcompressWriter, ScoresWriter, StatsWriter, TSVWriter,
};
use serde_json::{json, Value};
use sha3::{Digest, Sha3_256};
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::{fs, io};

/// SHA3-256 hex digest of `bytes`, matching the file-hash provenance format.
fn sha3_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Open a concrete seekable output file for the `bendl` writer, enforcing the
/// same overwrite guard as [`output_buffer`]. BENDL patches its header and
/// directory by seeking, so it cannot use the boxed (stdout-capable) sink.
fn bendl_output_file(path: &str, overwrite_output: bool) -> BufWriter<fs::File> {
    assert_can_write_output(Path::new(path), overwrite_output);
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap_or_else(|e| panic!("Could not open output file {}: {}", path, e));
    BufWriter::with_capacity(OUTPUT_BUFFER_CAPACITY, file)
}

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
    let cli = Command::new("frcw_short_bursts")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("A short bursts optimizer for redistricting")
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
                .help("The number of proposals to generate."),
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
                .help("The number of threads to use."),
        )
        .arg(
            Arg::new("burst_length")
                .long("burst-length")
                .value_parser(value_parser!(usize))
                .required(true)
                .help("The number of accepted steps per short burst."),
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
            Arg::new("edge_weight_keys")
                .long("edge-weight-keys")
                .value_parser(value_parser!(String))
                .num_args(1..)
                .help(
                    "Per-edge attribute columns whose values are added to edge weights \
                    in RMST / region-aware spanning-tree sampling. An edge missing a key \
                    contributes 0; a key present on no edge is an error. \
                    Only valid with the rmst and region-aware variants.",
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
                        edges of the previous plan uniformly at random. Sample using minimum \
                        spanning trees.\n\
                    \tdistrict-pairs-rmst (ReCom-B, default): sample pairs of districts uniformly \
                        at random from the space of all possible pairings. Sample using minimum \
                        spanning trees.\n\
                    \tcut-edges-ust (ReCom-C): sample district pairs by selecting one of the cut \
                        edges of the previous plan uniformly at random. Sample using uniform \
                        spanning trees. \n\
                    \tdistrict-pairs-ust (ReCom-D): sample pairs of districts uniformly at random \
                        from the space of all possible pairings. Sample using uniform spanning \
                        trees.",
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
                    \tben: Compressed binary format for post-processing with BEN\n\
                    \tbendl: Self-describing BENDL file (graph + metadata + BEN stream in one \
                        file); suppresses the _metadata.jsonl sidecar\n\
                    Note: short bursts workers return full partitions, not individual proposals.\n\
                    Writers that require proposal-level data (tsv, jsonl, pcompress) will have\n\
                    empty proposal fields. Prefer assignments, canonical, ben, or bendl.",
                ),
        )
        .arg(
            Arg::new("bendl_graph_order")
                .long("bendl-graph-order")
                .value_parser(value_parser!(String))
                .default_value("none")
                .help(
                    "Graph reordering applied before the chain runs, for better BENDL stream \
                    compression. Only valid with '--writer bendl'.\n\
                    \tnone (default): embed the graph as-is\n\
                    \trcm: Reverse Cuthill-McKee ordering\n\
                    \tmlc: multilevel-cluster ordering\n\
                    \tkey:<attr>: sort nodes by node attribute <attr> (must be on every node)",
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
                    "Path to write per-burst objective scores as CSV with step, score, and per-district score columns.",
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
        )
        .arg(
            Arg::new("write-improved-scores-only")
                .long("write-improved-scores-only")
                .action(ArgAction::SetTrue)
                .help(
                    "When set, the scores writer records only rows that improve the global \
                    best objective score. Has no effect without --scores-output-file.",
                ),
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
    let burst_length = *matches
        .get_one::<usize>("burst_length")
        .expect("burst_length is required");
    let maximize = *matches
        .get_one::<bool>("maximize")
        .expect("maximize is required");
    let variant_str = matches
        .get_one::<String>("variant")
        .expect("variant has a default value")
        .as_str();
    let writer_str = matches
        .get_one::<String>("writer")
        .expect("writer has a default value")
        .as_str();
    let overwrite_output = matches.get_flag("overwrite-output");
    let show_progress = matches.get_flag("show-progress");
    let write_improved_scores_only = matches.get_flag("write-improved-scores-only");

    // BENDL needs a seekable file and embeds its provenance in the bundle's
    // Metadata asset, so the separate _metadata.jsonl sidecar is suppressed.
    let is_bendl = writer_str == "bendl";
    let bendl_order = BendlGraphOrder::parse(
        matches
            .get_one::<String>("bendl_graph_order")
            .expect("bendl_graph_order has a default value"),
    )
    .unwrap_or_else(|e| panic!("Parameter error: {}", e));
    if !bendl_order.is_none() && !is_bendl {
        panic!("Parameter error: '--bendl-graph-order' is only valid with '--writer bendl'.");
    }
    if is_bendl && matches.get_one::<String>("output-file").is_none() {
        panic!(
            "Parameter error: '--writer bendl' requires '--output-file' \
             (BENDL needs a seekable file and cannot stream to stdout)."
        );
    }

    let metadata_base_path = matches
        .get_one::<String>("output-file")
        .or_else(|| matches.get_one::<String>("scores-output-file"));
    let metadata_path = if is_bendl {
        None
    } else {
        metadata_base_path.map(|path| metadata_path(path))
    };

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
    let edge_weight_keys: Vec<String> = matches
        .get_many::<String>("edge_weight_keys")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let region_weights_raw = (*matches.get_one::<String>("region_weights").unwrap()).as_str();
    let region_weights = parse_region_weights_config(region_weights_raw);
    // Add the keys in the region weights to sum_cols so the user doesn't have to specify them twice.
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
    let mut edge_cols = required_edge_cols(objective_config);
    for key in edge_weight_keys.iter() {
        if !edge_cols.contains(key) {
            edge_cols.push(key.clone());
        }
    }
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

    // Load the graph and partition and compute graph provenance. The bendl arm
    // reorders the JSON in memory and builds the chain from the exact bytes it
    // will embed, so the run and the embedded Graph asset can never diverge.
    let (mut graph, partition, embed_bytes, source_graph_sha3, embedded_graph_sha3) = if is_bendl {
        let source_bytes = fs::read(&graph_json)
            .unwrap_or_else(|e| panic!("Could not read graph file {}: {}", graph_json, e));
        let source_sha3 = sha3_hex(&source_bytes);
        let embed_bytes = reorder_graph_json(&source_bytes, &bendl_order)
            .unwrap_or_else(|e| panic!("Could not reorder graph for BENDL: {}", e));
        let embedded_sha3 = sha3_hex(&embed_bytes);
        let data: Value = serde_json::from_slice(&embed_bytes)
            .unwrap_or_else(|e| panic!("Could not parse reordered graph JSON: {}", e));
        let (graph, partition) = from_networkx_value(
            data,
            pop_col,
            assignment_col,
            sum_cols,
            partial_cols,
            edge_cols,
        )
        .unwrap();
        (
            graph,
            partition,
            Some(embed_bytes),
            source_sha3,
            Some(embedded_sha3),
        )
    } else {
        let (graph, partition) = from_networkx(
            &graph_json,
            pop_col,
            assignment_col,
            sum_cols,
            partial_cols,
            edge_cols,
        )
        .unwrap();
        let mut graph_file = fs::File::open(&graph_json).unwrap();
        let mut graph_hasher = Sha3_256::new();
        io::copy(&mut graph_file, &mut graph_hasher).unwrap();
        (
            graph,
            partition,
            None,
            format!("{:x}", graph_hasher.finalize()),
            None,
        )
    };
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
    let variant = match variant_str {
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
        "reversible" => {
            panic!("Reversible ReCom is not supported by the short bursts optimizer.")
        }
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
        edge_weight_keys: edge_weight_keys,
    };

    let mut meta = json!({
        "assignment_col": assignment_col,
        "tol": tol,
        "pop_col": pop_col,
        "graph_path": graph_json,
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
    // For BENDL the original-file hash no longer verifies the (possibly
    // reordered) embedded asset, so replace `graph_sha3` with explicit
    // source/embedded provenance plus the ordering applied. When order=none the
    // two hashes are equal.
    if is_bendl {
        let meta_object = meta.as_object_mut().unwrap();
        meta_object.remove("graph_sha3");
        meta_object.insert("source_graph_sha3".to_string(), json!(source_graph_sha3));
        meta_object.insert(
            "embedded_graph_sha3".to_string(),
            json!(embedded_graph_sha3
                .clone()
                .expect("bendl computes the embedded hash")),
        );
        meta_object.insert("bendl_graph_order".to_string(), json!(bendl_order.label()));
    }

    // The sidecar is suppressed in bendl mode (`metadata_path` is `None`); the
    // same provenance lives in the bundle's Metadata asset.
    let mut metadata_writer: Option<Box<dyn io::Write + Send>> = metadata_path
        .as_ref()
        .map(|path| output_buffer(path.to_str().unwrap(), overwrite_output));
    if let Some(writer) = metadata_writer.as_mut() {
        writeln!(writer, "{}", json!({ "meta": meta })).unwrap();
        writer.flush().unwrap();
    }

    let mut stats_writer: Option<Box<dyn StatsWriter>> = if is_bendl {
        let path = matches
            .get_one::<String>("output-file")
            .expect("bendl requires --output-file");
        let bundle_file = bendl_output_file(path, overwrite_output);
        Some(Box::new(BendlBenStreamWriter::new(
            bundle_file,
            embed_bytes.expect("bendl computes the embed bytes"),
            meta.to_string().into_bytes(),
        )))
    } else {
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
        }
    };
    let mut scores_writer: Option<ScoresWriter> = matches
        .get_one::<String>("scores-output-file")
        .map(|path| ScoresWriter::new(output_buffer(path, overwrite_output)));

    let backend = IncrementalBackend { objective };
    let output = multi_short_bursts_with_writer(
        &graph,
        partition,
        &params,
        n_threads,
        backend,
        maximize,
        burst_length,
        stats_writer
            .as_mut()
            .map(|writer| &mut **writer as &mut dyn StatsWriter),
        scores_writer.as_mut(),
        show_progress,
        write_improved_scores_only,
    );

    match output {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Error during optimization: {}", e);
        }
    }
}
