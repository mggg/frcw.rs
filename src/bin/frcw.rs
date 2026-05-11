//! Main CLI for frcw.
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, ArgAction, Command};
use frcw::config::parse_region_weights_config;
use frcw::init::from_networkx;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{
    AssignmentsOnlyWriter, BenWriter, CanonicalWriter, JSONLWriter, PcompressWriter, StatsWriter,
    TSVWriter,
};
use serde_json::json;
use sha3::{Digest, Sha3_256};
use std::path::PathBuf;
use std::{fs, io};

const OUTPUT_BUFFER_CAPACITY: usize = 128 * 1024;

fn output_buffer(path: &str, overwrite_output: bool) -> Box<dyn io::Write + Send> {
    let path = std::path::Path::new(path);
    if path.exists() && !overwrite_output {
        panic!("Output file already exists. Use --overwrite-output to replace it.");
    };
    Box::new(io::BufWriter::with_capacity(
        OUTPUT_BUFFER_CAPACITY,
        fs::File::create(path).unwrap(),
    ))
}

fn main() {
    let mut cli =
        Command::new("frcw")
            .version("0.1.3")
            .author("Parker J. Rule <parker.rule@tufts.edu>")
            .about("A minimal implementation of the ReCom Markov chain")
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
                Arg::new("target_pop")
                    .long("target-pop")
                    .value_parser(value_parser!(u64))
                    .help("The target population for the districts."),
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
                Arg::new("balance_ub")
                    .long("balance-ub")
                    .short('M') // Variable used in RevReCom paper
                    .value_parser(value_parser!(u32))
                    .default_value("0") // TODO: just use unwrap_or_default() instead?
                    .help("The normalizing constant (reversible ReCom only)."),
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
            ) // other options: cut_edges, district_pairs
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
                        storing ensembles).",
                    ),
            ) // other options: jsonl-full, tsv
            .arg(
                Arg::new("sum_cols")
                    .long("sum-cols")
                    .value_parser(value_parser!(Option<String>))
                    .num_args(1..)
                    .default_value(None)
                    .help("Additional columns in the graph metadata to sum over districts."),
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
                Arg::new("cut_edges_count")
                    .long("cut-edges-count")
                    .action(ArgAction::SetTrue)
                    .help("Whether to compute and output the cut edges count at each step."),
            )
            .arg(Arg::new("output-file").long("output-file").short('o').help(
                "The path to write the output to. If not provided, ouput is printed to console.",
            ))
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

    if cfg!(feature = "linalg") {
        cli = cli.arg(
            Arg::new("spanning_tree_counts")
                .long("st-counts")
                .action(ArgAction::SetTrue)
                .help("Whether to compute and output the spanning tree counts at each step."),
        );
    }

    let matches = cli.get_matches();

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
    let graph_path_buf = PathBuf::from(graph_path);
    let graph_json = match fs::canonicalize(&graph_path_buf)
        .map_err(|e| format!("Could not canonicalize path {:?}: {e}", graph_path_buf))
        .expect(format!("Could not create fs buffer from {graph_path}").as_str())
        .into_os_string()
        .into_string()
        .map_err(|_| {
            format!(
                "path for --graph-jaon is not valid UTF-8: {:?}",
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
    let region_weights_raw = (*matches.get_one::<String>("region_weights").unwrap()).as_str();

    let variant = match variant_str {
        "reversible" => RecomVariant::Reversible,
        "cut-edges-ust" => RecomVariant::CutEdgesUST,
        "cut-edges-rmst" => RecomVariant::CutEdgesRMST,
        "cut-edges-region-aware" => RecomVariant::CutEdgesRegionAware,
        "district-pairs-ust" => RecomVariant::DistrictPairsUST,
        "district-pairs-rmst" => RecomVariant::DistrictPairsRMST,
        "district-pairs-region-aware" => RecomVariant::DistrictPairsRegionAware,
        bad => panic!("Parameter error: invalid variant '{}'", bad),
    };

    let output_buffer: Box<dyn io::Write + Send> = match matches.get_one::<String>("output-file") {
        Some(path) => output_buffer(path, overwrite_output),
        None => Box::new(io::BufWriter::with_capacity(
            OUTPUT_BUFFER_CAPACITY,
            std::io::stdout(),
        )),
    };

    let writer: Box<dyn StatsWriter> = match writer_str {
        "tsv" => Box::new(TSVWriter::new(output_buffer)),
        "jsonl" => Box::new(JSONLWriter::new(
            false,
            st_counts,
            cut_edges_count,
            output_buffer,
        )),
        "pcompress" => Box::new(PcompressWriter::new(output_buffer)),
        "jsonl-full" => Box::new(JSONLWriter::new(
            true,
            st_counts,
            cut_edges_count,
            output_buffer,
        )),
        "assignments" => Box::new(AssignmentsOnlyWriter::new(false, output_buffer)),
        "canonicalized-assignments" => Box::new(AssignmentsOnlyWriter::new(true, output_buffer)),
        "canonical" => Box::new(CanonicalWriter::new(output_buffer)),
        "ben" => Box::new(BenWriter::new(output_buffer)),
        bad => panic!("Parameter error: invalid writer '{}'", bad),
    };
    if variant == RecomVariant::Reversible && balance_ub == 0 {
        panic!("For reversible ReCom, specify M > 0.");
    }

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }

    let region_weights = parse_region_weights_config(region_weights_raw);
    // Add the keys in the region weights to sum_cols if they are not there already
    // so that the user doesn't have to
    if let Some(weight_pairs_vec) = &region_weights {
        for (key, _) in weight_pairs_vec.iter() {
            if !sum_cols.contains(&key) {
                sum_cols.push(key.clone().to_string());
            }
        }
    }

    let (graph, partition) = from_networkx(
        &graph_json,
        pop_col,
        assignment_col,
        sum_cols,
        vec![],
        vec![],
    )
        .unwrap_or_else(|e| {
            panic!(
                "Could not load graph and partition from {}: {}",
                graph_json, e
            )
        });

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
        rng_seed: rng_seed,
        balance_ub: balance_ub,
        variant: variant,
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
        "batch_size": batch_size,
        "rng_seed": rng_seed,
        "num_threads": n_threads,
        "num_steps": n_steps,
        "parallel": true,
        "graph_json": graph_json,
        "chain_variant": variant_str,
    });
    if let Some(path) = matches.get_one::<String>("output-file") {
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
    if writer_str == "jsonl" || writer_str == "jsonl-full" {
        // hotfix for pcompress writing
        // TODO: move this into init
        println!("{}", json!({ "meta": meta }).to_string());
    }

    let show_progress = matches.get_flag("show-progress");

    let output = multi_chain(
        &graph,
        &partition,
        writer,
        &params,
        n_threads,
        batch_size,
        show_progress,
    );
    match output {
        Ok(_) => {}
        Err(e) => panic!("Error during chain execution: {}", e),
    }
}
