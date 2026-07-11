//! Reversible ReCom distribution test for the frcw engine.
//!
//! Intended for use with the MGGG benchmark suite
//! (https://github.com/mggg/benchmarks).
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use clap::{value_parser, Arg, Command};
use frcw::graph::Graph;
use frcw::partition::Partition;
use frcw::recom::run::multi_chain;
use frcw::recom::{RecomParams, RecomVariant};
use frcw::stats::{AssignmentsOnlyWriter, StatsWriter};
use std::fs::read_to_string;

fn main() {
    let matches = Command::new("frcw-revrecom-dist-test")
        .version(env!("CARGO_PKG_VERSION"))
        .author("Parker J. Rule <parker.rule@tufts.edu>")
        .about("RevReCom distribution tests for frcw")
        .arg(
            Arg::new("graph_file")
                .long("graph-file")
                .value_parser(value_parser!(String))
                .required(true)
                .help("The path of the dual graph (in edge list format)."),
        )
        .arg(
            Arg::new("pop_file")
                .long("pop-file")
                .value_parser(value_parser!(String))
                .required(true)
                .help("The path of the population file for the dual graph."),
        )
        .arg(
            Arg::new("assignment_file")
                .long("assignment-file")
                .value_parser(value_parser!(String))
                .required(true)
                .help("The path of the seed plan assignment for the dual graph."),
        )
        .arg(
            Arg::new("n_steps")
                .long("n-steps")
                .value_parser(value_parser!(u64))
                .required(true)
                .help("The number of proposals to generate."),
        )
        .arg(
            Arg::new("tol")
                .long("tol")
                .value_parser(value_parser!(f64))
                .required(true)
                .help("The relative population tolerance."),
        )
        .arg(
            Arg::new("rng_seed")
                .long("rng-seed")
                .value_parser(value_parser!(u64))
                .required(true)
                .help("The seed of the RNG used to draw proposals."),
        )
        .arg(
            Arg::new("balance_ub")
                .long("balance-ub")
                .value_parser(value_parser!(u32))
                .short('M') // Variable used in RevReCom paper
                .default_value("0") // TODO: just use unwrap_or_default() instead?
                .help("The normalizing constant (reversible ReCom only)."),
        )
        .arg(
            Arg::new("n_threads")
                .long("n-threads")
                .value_parser(value_parser!(usize))
                .required(true)
                .help("The number of threads to use."),
        )
        .arg(
            Arg::new("batch_size")
                .long("batch-size")
                .value_parser(value_parser!(usize))
                .required(true)
                .help("The number of proposals per batch job."),
        )
        .get_matches();

    let n_steps = *matches
        .get_one::<u64>("n_steps")
        .expect("n_steps is required");
    let rng_seed = *matches
        .get_one::<u64>("rng_seed")
        .expect("rng_seed is required");
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
    let pop_path = matches
        .get_one::<String>("pop_file")
        .expect("pop_file is required");
    let assignments_path = matches
        .get_one::<String>("assignment_file")
        .expect("assignment_file is required");
    let graph_path = matches
        .get_one::<String>("graph_file")
        .expect("graph_file is required");

    if tol < 0.0 || tol > 1.0 {
        panic!("Parameter error: '--tol' must be between 0 and 1.");
    }

    let graph_data = read_to_string(graph_path).expect("Could not read edge list file");
    let pop_data = read_to_string(pop_path).expect("Could not read population file");
    let assignments_data =
        read_to_string(assignments_path).expect("Could not read assignment file");

    let graph = Graph::from_edge_list(&graph_data, &pop_data).unwrap();
    let partition = Partition::from_assignment_str(&graph, &assignments_data).unwrap();
    let avg_pop = (graph.total_pop as f64) / (partition.num_dists as f64);
    let params = RecomParams {
        min_pop: ((1.0 - tol) * avg_pop as f64).floor() as u32,
        max_pop: ((1.0 + tol) * avg_pop as f64).ceil() as u32,
        num_steps: n_steps,
        rng_seed: rng_seed,
        balance_ub: balance_ub,
        variant: RecomVariant::Reversible,
        region_weights: None,
        edge_weight_keys: vec![],
    };

    let output_buffer = Box::new(std::io::BufWriter::new(std::io::stdout()));
    let writer: Box<dyn StatsWriter> = Box::new(AssignmentsOnlyWriter::new(true, output_buffer));

    let output = multi_chain(
        &graph, &partition, writer, &params, n_threads, batch_size, false,
    );
    match output {
        Ok(_) => {}
        Err(e) => panic!("Error during chain execution: {}", e),
    }
}
