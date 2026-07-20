//! Helpers shared by the rustrecom subcommands.
//!
//! Everything here is a consolidation of code that was byte-identical (or
//! differed only in a message label, passed through as a parameter) across the
//! former `frcw`, `frcw_short_bursts`, and `frcw_tilted` binaries. Panic
//! messages and check ordering are preserved verbatim; behavior differences
//! that are real (variant policy, metadata asymmetries, per-command defaults)
//! stay in the command modules.

use clap::{value_parser, Arg, ArgAction, ArgMatches};
use rustrecom::bendl::{reorder_graph_json, BendlGraphOrder};
use rustrecom::graph::Graph;
use rustrecom::init::{from_networkx, from_networkx_value};
use rustrecom::objectives::{
    make_objective, partial_node_cols, required_edge_cols, required_node_cols, ObjectiveConfig,
};
use rustrecom::partition::Partition;
use rustrecom::recom::RecomVariant;
use rustrecom::stats::{
    AssignmentsOnlyWriter, BenWriter, BendlBenStreamWriter, CanonicalWriter, JSONLWriter,
    PcompressWriter, ScoresWriter, StatsWriter, TSVWriter,
};
use serde_json::{json, Value};
use sha3::{Digest, Sha3_256};
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::{fs, io};

pub const OUTPUT_BUFFER_CAPACITY: usize = 128 * 1024;

/// SHA3-256 hex digest of `bytes`, matching the file-hash provenance format.
pub fn sha3_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn assert_can_write_output(path: &Path, overwrite_output: bool) {
    if path.exists() && !overwrite_output {
        panic!("Output file already exists. Use --overwrite-output to replace it.");
    };
}

pub fn output_buffer(path: &str, overwrite_output: bool) -> Box<dyn io::Write + Send> {
    let path = std::path::Path::new(path);
    assert_can_write_output(path, overwrite_output);
    Box::new(io::BufWriter::with_capacity(
        OUTPUT_BUFFER_CAPACITY,
        fs::File::create(path).unwrap(),
    ))
}

/// Open a concrete seekable output file for the `bendl` writer, enforcing the
/// same overwrite guard as [`output_buffer`]. BENDL patches its header and
/// directory by seeking, so it cannot use the boxed (stdout-capable) sink.
pub fn bendl_output_file(path: &str, overwrite_output: bool) -> BufWriter<fs::File> {
    assert_can_write_output(Path::new(path), overwrite_output);
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .unwrap_or_else(|e| panic!("Could not open output file {}: {}", path, e));
    BufWriter::with_capacity(OUTPUT_BUFFER_CAPACITY, file)
}

pub fn metadata_path(output_path: &str) -> PathBuf {
    let output_path = PathBuf::from(output_path);
    let file_stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .expect("Output path must have a valid UTF-8 file name.");
    output_path.with_file_name(format!("{}_metadata.jsonl", file_stem))
}

/// Resolves an inline-JSON-or-file argument to a JSON string.
/// Values that start with `{` are used as-is (inline JSON).
/// All other values are treated as file paths and the file contents are returned.
/// `label` names the argument in the read-failure panic so each command keeps
/// its original message ("JSON config" for chain, "objective file" for the
/// optimizers).
pub fn load_json_arg(arg: &str, label: &str) -> String {
    if arg.trim_start().starts_with('{') {
        arg.to_string()
    } else {
        fs::read_to_string(arg)
            .unwrap_or_else(|e| panic!("Could not read {} '{}': {}", label, arg, e))
    }
}

/// Chain passes real `st_counts` / `cut_edges_count` values; the optimizers
/// pass `false, false` (their writers carry no per-proposal statistics).
pub fn make_stats_writer(
    writer_str: &str,
    st_counts: bool,
    cut_edges_count: bool,
    output_buffer: Box<dyn io::Write + Send>,
) -> Box<dyn StatsWriter> {
    match writer_str {
        "tsv" => Box::new(TSVWriter::new(output_buffer)),
        "jsonl" => Box::new(JSONLWriter::new(
            false,
            st_counts,
            cut_edges_count,
            output_buffer,
        )),
        "jsonl-full" => Box::new(JSONLWriter::new(
            true,
            st_counts,
            cut_edges_count,
            output_buffer,
        )),
        "pcompress" => Box::new(PcompressWriter::new(output_buffer)),
        "assignments" => Box::new(AssignmentsOnlyWriter::new(false, output_buffer)),
        "canonicalized-assignments" => Box::new(AssignmentsOnlyWriter::new(true, output_buffer)),
        "canonical" => Box::new(CanonicalWriter::new(output_buffer)),
        "ben" => Box::new(BenWriter::new(output_buffer)),
        bad => panic!("Parameter error: invalid writer '{}'", bad),
    }
}

/// Canonicalize the `--graph-json` argument to an absolute UTF-8 path.
pub fn canonicalize_graph_path(graph_path: &str) -> String {
    let graph_path_buf = PathBuf::from(graph_path);
    match fs::canonicalize(&graph_path_buf)
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
    }
}

/// Parse the bendl graph order and enforce the bendl flag pairing rules shared
/// by all commands. Takes resolved values rather than `ArgMatches` because
/// chain's config mode sources them from the config document, not the CLI.
/// Returns `(is_bendl, bendl_order)`.
pub fn resolve_bendl_options(
    writer_str: &str,
    bendl_graph_order: &str,
    has_output_file: bool,
) -> (bool, BendlGraphOrder) {
    let is_bendl = writer_str == "bendl";
    let bendl_order = BendlGraphOrder::parse(bendl_graph_order)
        .unwrap_or_else(|e| panic!("Parameter error: {}", e));
    if !bendl_order.is_none() && !is_bendl {
        panic!("Parameter error: '--bendl-graph-order' is only valid with '--writer bendl'.");
    }
    if is_bendl && !has_output_file {
        panic!(
            "Parameter error: '--writer bendl' requires '--output-file' \
             (BENDL needs a seekable file and cannot stream to stdout)."
        );
    }
    (is_bendl, bendl_order)
}

/// Add the keys in the region weights to `sum_cols` if they are not there
/// already so that the user doesn't have to.
pub fn merge_region_weight_cols(
    sum_cols: &mut Vec<String>,
    region_weights: &Option<Vec<(String, f64)>>,
) {
    if let Some(weight_pairs_vec) = region_weights {
        for (key, _) in weight_pairs_vec.iter() {
            if !sum_cols.contains(key) {
                sum_cols.push(key.clone().to_string());
            }
        }
    }
}

pub struct LoadedGraph {
    pub graph: Graph,
    pub partition: Partition,
    pub embed_bytes: Option<Vec<u8>>,
    pub source_graph_sha3: String,
    pub embedded_graph_sha3: Option<String>,
}

/// Load the graph and partition and compute graph provenance. The bendl arm
/// reorders the JSON in memory and builds the chain from the exact bytes it
/// will embed, so the run and the embedded Graph asset can never diverge.
#[allow(clippy::too_many_arguments)]
pub fn load_graph_with_provenance(
    graph_json: &str,
    pop_col: &str,
    assignment_col: &str,
    sum_cols: Vec<String>,
    partial_cols: Vec<String>,
    edge_cols: Vec<String>,
    is_bendl: bool,
    bendl_order: &BendlGraphOrder,
) -> LoadedGraph {
    if is_bendl {
        let source_bytes = fs::read(graph_json)
            .unwrap_or_else(|e| panic!("Could not read graph file {}: {}", graph_json, e));
        let source_sha3 = sha3_hex(&source_bytes);
        let embed_bytes = reorder_graph_json(&source_bytes, bendl_order)
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
        .unwrap_or_else(|e| {
            panic!(
                "Could not load graph and partition from {}: {}",
                graph_json, e
            )
        });
        LoadedGraph {
            graph,
            partition,
            embed_bytes: Some(embed_bytes),
            source_graph_sha3: source_sha3,
            embedded_graph_sha3: Some(embedded_sha3),
        }
    } else {
        let (graph, partition) = from_networkx(
            graph_json,
            pop_col,
            assignment_col,
            sum_cols,
            partial_cols,
            edge_cols,
        )
        .unwrap_or_else(|e| {
            panic!(
                "Could not load graph and partition from {}: {}",
                graph_json, e
            )
        });
        let mut graph_file = fs::File::open(graph_json).unwrap();
        let mut graph_hasher = Sha3_256::new();
        io::copy(&mut graph_file, &mut graph_hasher).unwrap();
        LoadedGraph {
            graph,
            partition,
            embed_bytes: None,
            source_graph_sha3: format!("{:x}", graph_hasher.finalize()),
            embedded_graph_sha3: None,
        }
    }
}

/// For BENDL the original-file hash no longer verifies the (possibly
/// reordered) embedded asset, so replace `graph_sha3` with explicit
/// source/embedded provenance plus the ordering applied. When order=none the
/// two hashes are equal.
pub fn enrich_bendl_meta(
    meta: &mut Value,
    source_graph_sha3: &str,
    embedded_graph_sha3: &Option<String>,
    bendl_order: &BendlGraphOrder,
) {
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

// ---------------------------------------------------------------------------
// Optimizer-only helpers (shared by short-bursts and tilted).
// ---------------------------------------------------------------------------

/// Derive the optimizer metadata sidecar path and enforce the shared output
/// path rules: sidecar base from `--output-file` falling back to
/// `--scores-output-file`, no collisions among output, score, and metadata
/// paths, and every overwrite check before any destination is opened. The
/// sidecar is suppressed (`None`) in bendl mode.
pub fn plan_optimizer_output_paths(
    output_file: Option<&str>,
    scores_output_file: Option<&str>,
    is_bendl: bool,
    overwrite_output: bool,
) -> Option<PathBuf> {
    let metadata_base_path = output_file.or(scores_output_file);
    let metadata_path = if is_bendl {
        None
    } else {
        metadata_base_path.map(metadata_path)
    };

    if let (Some(output_path), Some(scores_path)) = (output_file, scores_output_file) {
        if PathBuf::from(output_path) == PathBuf::from(scores_path) {
            panic!(
                "Parameter error: '--output-file' and '--scores-output-file' must be different."
            );
        }
    }
    if let Some(metadata_path) = &metadata_path {
        if output_file.is_some_and(|path| PathBuf::from(path) == *metadata_path)
            || scores_output_file.is_some_and(|path| PathBuf::from(path) == *metadata_path)
        {
            panic!("Parameter error: derived metadata path conflicts with an output path.");
        }
    }
    let output_paths = [
        output_file.map(PathBuf::from),
        scores_output_file.map(PathBuf::from),
        metadata_path.clone(),
    ];
    for path in output_paths.into_iter().flatten() {
        assert_can_write_output(&path, overwrite_output);
    }
    metadata_path
}

/// Column inputs shared by the two optimizers, parsed in their common order.
pub struct OptimizerInputs {
    pub graph_json: String,
    pub pop_col: String,
    pub assignment_col: String,
    pub sum_cols: Vec<String>,
    pub partial_cols: Vec<String>,
    pub edge_weight_keys: Vec<String>,
    pub region_weights: Option<Vec<(String, f64)>>,
}

impl OptimizerInputs {
    /// Builds the inputs from already-resolved config values, applying the
    /// same canonicalization and region-weight merging as the CLI path.
    #[allow(clippy::too_many_arguments)]
    pub fn from_config_values(
        graph_path: &str,
        pop_col: String,
        assignment_col: String,
        mut sum_cols: Vec<String>,
        partial_cols: Vec<String>,
        edge_weight_keys: Vec<String>,
        region_weights: Option<Vec<(String, f64)>>,
    ) -> Self {
        let graph_json = canonicalize_graph_path(graph_path);
        merge_region_weight_cols(&mut sum_cols, &region_weights);
        OptimizerInputs {
            graph_json,
            pop_col,
            assignment_col,
            sum_cols,
            partial_cols,
            edge_weight_keys,
            region_weights,
        }
    }
}

pub fn parse_optimizer_inputs(matches: &ArgMatches) -> OptimizerInputs {
    let graph_path = matches
        .get_one::<String>("graph_json")
        .expect("graph_json is required");
    let graph_json = canonicalize_graph_path(graph_path);
    let pop_col = matches
        .get_one::<String>("pop_col")
        .expect("pop_col is required")
        .clone();
    let assignment_col = matches
        .get_one::<String>("assignment_col")
        .expect("assignment_col is required")
        .clone();
    let mut sum_cols: Vec<String> = matches
        .get_many::<String>("sum_cols")
        .unwrap_or_default()
        .map(|c| c.to_string())
        .collect();
    let partial_cols: Vec<String> = matches
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
    let region_weights = rustrecom::config::parse_region_weights_config(region_weights_raw);
    merge_region_weight_cols(&mut sum_cols, &region_weights);
    OptimizerInputs {
        graph_json,
        pop_col,
        assignment_col,
        sum_cols,
        partial_cols,
        edge_weight_keys,
        region_weights,
    }
}

/// Resolve `--objective` and fold its required columns into the load lists.
/// Returns the raw objective config string, the constructed objective, and the
/// edge-attribute columns to load. The returned edge columns are the union of
/// the objective's requirements and `--edge-weight-keys`; the raw
/// `edge_weight_keys` list in `inputs` is left untouched because it also feeds
/// `RecomParams` (where augmenting it would change tree sampling).
pub fn prepare_objective(
    matches: &ArgMatches,
    inputs: &mut OptimizerInputs,
) -> (String, ObjectiveConfig, Vec<String>) {
    let objective_config = load_json_arg(
        matches
            .get_one::<String>("objective")
            .expect("objective is required"),
        "objective file",
    );
    prepare_objective_config(objective_config, inputs)
}

/// Core of [`prepare_objective`] for callers that already hold the objective
/// JSON string (config mode carries it as an inline object).
pub fn prepare_objective_config(
    objective_config: String,
    inputs: &mut OptimizerInputs,
) -> (String, ObjectiveConfig, Vec<String>) {
    let objective = make_objective(&objective_config);
    let mut edge_cols = required_edge_cols(&objective_config);
    for key in inputs.edge_weight_keys.iter() {
        if !edge_cols.contains(key) {
            edge_cols.push(key.clone());
        }
    }
    for col in required_node_cols(&objective_config) {
        if !inputs.sum_cols.contains(&col) {
            inputs.sum_cols.push(col);
        }
    }
    for col in partial_node_cols(&objective_config) {
        if !inputs.partial_cols.contains(&col) && !inputs.sum_cols.contains(&col) {
            inputs.partial_cols.push(col);
        }
    }
    (objective_config, objective, edge_cols)
}

/// Maps deprecated public variant names to their current spellings and warns.
pub fn normalize_variant(variant: &str) -> &str {
    match variant {
        "cut-edges-rmst" => {
            eprintln!(
                "warning: `cut-edges-rmst` is deprecated and will be removed in a future \
                 release; use `cut-edges-mst` instead"
            );
            "cut-edges-mst"
        }
        "district-pairs-rmst" => {
            eprintln!(
                "warning: `district-pairs-rmst` is deprecated and will be removed in a future \
                 release; use `district-pairs-mst` instead"
            );
            "district-pairs-mst"
        }
        _ => variant,
    }
}

/// Shared optimizer variant resolution. The optimizers only reach the
/// region-aware variants by upgrading RMST when region weights are present;
/// UST has no region-aware implementation. `reversible_error` carries each
/// command's engine message for the reversible rejection.
pub fn optimizer_variant(
    variant_str: &str,
    region_weights: &Option<Vec<(String, f64)>>,
    reversible_error: &str,
) -> RecomVariant {
    match normalize_variant(variant_str) {
        "cut-edges-mst" => match region_weights {
            None => RecomVariant::CutEdgesRMST,
            Some(_) => RecomVariant::CutEdgesRegionAware,
        },
        "district-pairs-mst" => match region_weights {
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
        "reversible" => panic!("{}", reversible_error),
        bad => panic!("Parameter error: invalid variant '{}'", bad),
    }
}

pub struct OptimizerWriters {
    pub stats: Option<Box<dyn StatsWriter>>,
    pub scores: Option<ScoresWriter>,
}

/// Write the metadata sidecar and open the stats and score writers, in the
/// optimizers' shared order: sidecar first, then the stats destination, then
/// the scores destination. The sidecar is suppressed in bendl mode
/// (`metadata_path` is `None`); the same provenance lives in the bundle's
/// Metadata asset. In config mode (`raw_config` is `Some`) the sidecar and
/// the bendl Metadata asset carry the exact raw config string instead of the
/// meta record, preserving provenance byte-for-byte.
#[allow(clippy::too_many_arguments)]
pub fn build_optimizer_writers(
    output_file: Option<&str>,
    scores_output_file: Option<&str>,
    is_bendl: bool,
    writer_str: &str,
    overwrite_output: bool,
    metadata_path: &Option<PathBuf>,
    meta: &Value,
    embed_bytes: Option<Vec<u8>>,
    raw_config: Option<&str>,
) -> OptimizerWriters {
    let mut metadata_writer: Option<Box<dyn io::Write + Send>> = metadata_path
        .as_ref()
        .map(|path| output_buffer(path.to_str().unwrap(), overwrite_output));
    if let Some(writer) = metadata_writer.as_mut() {
        use std::io::Write;
        match raw_config {
            Some(raw) => writeln!(writer, "{}", raw).unwrap(),
            None => writeln!(writer, "{}", json!({ "meta": meta })).unwrap(),
        }
        writer.flush().unwrap();
    }

    let stats: Option<Box<dyn StatsWriter>> = if is_bendl {
        let path = output_file.expect("bendl requires --output-file");
        let bundle_file = bendl_output_file(path, overwrite_output);
        let metadata = raw_config
            .map(|raw| raw.as_bytes().to_vec())
            .unwrap_or_else(|| meta.to_string().into_bytes());
        Some(Box::new(BendlBenStreamWriter::new(
            bundle_file,
            embed_bytes.expect("bendl computes the embed bytes"),
            metadata,
        )))
    } else {
        match output_file {
            Some(path) => Some(make_stats_writer(
                writer_str,
                false,
                false,
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
    let scores: Option<ScoresWriter> =
        scores_output_file.map(|path| ScoresWriter::new(output_buffer(path, overwrite_output)));
    OptimizerWriters { stats, scores }
}

// ---------------------------------------------------------------------------
// Config-mode helpers shared by all subcommands.
// ---------------------------------------------------------------------------

/// Marks an argument as required only in CLI mode; config mode supplies it
/// from the config document instead.
pub fn config_optional(arg: Arg) -> Arg {
    arg.required(false).required_unless_present("config")
}

/// The shared `--config` declaration.
pub fn config_arg() -> Arg {
    Arg::new("config")
        .long("config")
        .value_parser(value_parser!(String))
        .help(
            "Run from a versioned JSON config whose fields mirror the CLI \
            arguments. Either a JSON string (must start with '{'), a path \
            to a JSON file, or '-' to read the JSON from stdin.",
        )
}

/// Arguments that may accompany `--config`. `overwrite-output` is output-lifecycle
/// policy rather than a sampler value: it has no v1 config field, and wrappers
/// driving config mode need it to keep the clobbering behavior of the shell
/// redirects it replaces.
const CONFIG_MODE_COMPANION_ARGS: &[&str] = &["config", "overwrite-output"];

/// Reject config mode combined with any explicitly supplied CLI argument.
///
/// The argument set is read back off the parsed `ArgMatches` rather than listed here,
/// so a newly added CLI option is covered without touching this function. `value_source`
/// distinguishes a value the user typed from one clap supplied as a default, which a
/// raw argv scan or a `get_one` check cannot do.
pub fn reject_mixed_config_args(matches: &ArgMatches) {
    let mut mixed = matches
        .ids()
        .map(|id| id.as_str())
        .filter(|id| !CONFIG_MODE_COMPANION_ARGS.contains(id))
        .filter(|id| matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine))
        .collect::<Vec<_>>();
    mixed.sort_unstable();
    if !mixed.is_empty() {
        panic!(
            "--config cannot be combined with CLI arguments: {}",
            mixed.join(", ")
        );
    }
}

/// Resolves the `--config` argument to its raw string: inline JSON, stdin
/// (`-`), or a file path. Rejects mixed CLI arguments before reading anything.
pub fn resolve_config_argument(matches: &ArgMatches) -> Option<String> {
    let config_arg = matches.get_one::<String>("config")?;
    reject_mixed_config_args(matches);
    let raw = if config_arg == "-" {
        let mut buffer = String::new();
        io::Read::read_to_string(&mut io::stdin(), &mut buffer)
            .unwrap_or_else(|error| panic!("Could not read config from stdin: {error}"));
        buffer
    } else {
        load_json_arg(config_arg, "config")
    };
    Some(raw)
}

// ---------------------------------------------------------------------------
// Shared argument builders. Only arguments whose declaration (name, default,
// and help text) is byte-identical across their users live here; options that
// share a name but differ in defaults, validation, or help stay per-command.
// ---------------------------------------------------------------------------

pub fn graph_json_arg() -> Arg {
    Arg::new("graph_json")
        .long("graph-json")
        .required(true)
        .value_parser(value_parser!(String))
        .help("The path of the dual graph (in NetworkX format).")
}

pub fn tol_arg() -> Arg {
    Arg::new("tol")
        .long("tol")
        .required(true)
        .value_parser(value_parser!(f64))
        .help("The relative population tolerance.")
}

pub fn pop_col_arg() -> Arg {
    Arg::new("pop_col")
        .long("pop-col")
        .required(true)
        .value_parser(value_parser!(String))
        .help("The name of the total population column in the graph metadata.")
}

pub fn assignment_col_arg() -> Arg {
    Arg::new("assignment_col")
        .long("assignment-col")
        .required(true)
        .value_parser(value_parser!(String))
        .help("The name of the assignment column in the graph metadata.")
}

pub fn rng_seed_arg() -> Arg {
    Arg::new("rng_seed")
        .long("rng-seed")
        .required(true)
        .value_parser(value_parser!(u64))
        .help("The seed of the RNG used to draw proposals.")
}

pub fn n_threads_arg() -> Arg {
    Arg::new("n_threads")
        .long("n-threads")
        .required(false)
        .value_parser(value_parser!(usize))
        .default_value("1")
        .help("The number of threads to use.")
}

pub fn sum_cols_arg() -> Arg {
    Arg::new("sum_cols")
        .long("sum-cols")
        .value_parser(value_parser!(Option<String>))
        .num_args(1..)
        .default_value(None)
        .help("Additional columns in the graph metadata to sum over districts.")
}

pub fn partial_sum_cols_arg() -> Arg {
    Arg::new("partial_sum_cols")
        .long("partial-sum-cols")
        .value_parser(value_parser!(Option<String>))
        .num_args(1..)
        .default_value(None)
        .help(
            "Additional sum columns that may be missing on some nodes. \
            Missing entries are treated as zero instead of causing a load-time panic. \
            Use this for attributes that legitimately apply to only a subset of nodes.",
        )
}

pub fn edge_weight_keys_arg() -> Arg {
    Arg::new("edge_weight_keys")
        .long("edge-weight-keys")
        .value_parser(value_parser!(String))
        .num_args(1..)
        .help(
            "Per-edge attribute columns whose values are added to edge weights \
            in RMST / region-aware spanning-tree sampling. An edge missing a key \
            contributes 0; a key present on no edge is an error. \
            Only valid with the rmst and region-aware variants.",
        )
}

pub fn region_weights_arg() -> Arg {
    Arg::new("region_weights")
        .long("region-weights")
        .value_parser(value_parser!(String))
        .default_value("")
        .help(
            "Region columns with weights for region-aware ReCom. \
            Must be entered into the command line using the format:\n\
            \t'{\"region_col1\": weight1, \"region_col2\": weight2, ...}'",
        )
}

pub fn objective_arg() -> Arg {
    Arg::new("objective").long("objective").required(true).help(
        "Objective function configuration. Either a JSON string \
         (must start with '{') or a path to a JSON file.",
    )
}

pub fn maximize_arg() -> Arg {
    Arg::new("maximize")
        .long("maximize")
        .value_parser(value_parser!(bool))
        .default_value("true")
        .help("If true, maximize the objective. If false, minimize it.")
}

/// The optimizers' shared `--variant` declaration: default
/// `district-pairs-mst`, no reversible support.
pub fn optimizer_variant_arg() -> Arg {
    Arg::new("variant")
        .long("variant")
        .value_parser(value_parser!(String))
        .default_value("district-pairs-mst")
        .help(
            "The variant of the ReCom proposal to use.\n\
            \tcut-edges-mst (ReCom-A): sample district pairs by selecting one of the cut \
                edges of the previous plan uniformly at random. Sample using minimum \
                spanning trees.\n\
            \tdistrict-pairs-mst (ReCom-B, default): sample pairs of districts uniformly \
                at random from the space of all possible pairings. Sample using minimum \
                spanning trees.\n\
            \tcut-edges-ust (ReCom-C): sample district pairs by selecting one of the cut \
                edges of the previous plan uniformly at random. Sample using uniform \
                spanning trees. \n\
            \tdistrict-pairs-ust (ReCom-D): sample pairs of districts uniformly at random \
                from the space of all possible pairings. Sample using uniform spanning \
                trees.",
        )
}

/// The optimizers' shared `--output-file` declaration (absent means no chain
/// records; chain's version means stdout instead, so it stays in chain).
pub fn optimizer_output_file_arg() -> Arg {
    Arg::new("output-file")
        .long("output-file")
        .short('o')
        .help("Path to write chain records. If omitted, chain records are not written.")
}

pub fn bendl_graph_order_arg() -> Arg {
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
        )
}

pub fn overwrite_output_arg() -> Arg {
    Arg::new("overwrite-output")
        .long("overwrite-output")
        .action(ArgAction::SetTrue)
        .help("Overwrite existing output files instead of failing.")
}

pub fn show_progress_arg() -> Arg {
    Arg::new("show-progress")
        .long("show-progress")
        .action(ArgAction::SetTrue)
        .help("Whether to show a progress bar during execution.")
}
