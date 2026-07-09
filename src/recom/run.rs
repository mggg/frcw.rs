//! Runners for ReCom.
//!
//! A runner orchestrates the various components of the ReCom algorithm
//! (spanning tree generation, etc.) and handles setup, output, the
//! collection of auxiliary statistics, and (optionally) multithreading.
//!
//! Currently, there is only one runner ([`multi_chain`]). This runner
//! is multithreaded and prints accepted proposals to `stdout` in TSV format.
//! It also collects rejection/self-loop statistics.
use super::{
    make_sampler, node_bound, random_split, sample_dist_pair, RecomParams, RecomProposal,
    RecomVariant, WorkerBuffers,
};
use crate::buffers::graph_connected_buffered;
use crate::constraints::ChainConstraint;
use crate::graph::Graph;
use crate::partition::Partition;
use crate::stats::{SelfLoopCounts, SelfLoopReason, StatsWriter};
use crossbeam::scope;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Determines how many proposals the stats thread can lag behind by
/// (compared to the head of the chain).
const STATS_CHANNEL_CAPACITY: usize = 128;

/// A unit of multithreaded work.
struct JobPacket {
    /// The number of steps to sample (*not* the number of unique plans).
    n_steps: usize,
    /// The change in the chain state since the last batch of work.
    /// If no new proposal is accepted, this may be `None`.
    diff: Option<RecomProposal>,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// The result of a unit of multithreaded work.
struct ResultPacket {
    /// Self-loop statistics.
    counts: SelfLoopCounts,
    /// ≥0 valid proposals generated within the unit of work.
    proposals: Vec<(u64, RecomProposal)>,
}

/// Information necessary to compute statistics about an accepted proposal.
#[derive(Debug)]
struct StepPacket {
    /// The current step count of the chain.
    step: u64,
    /// The accepted proposal (only `None` when the termination sentinel is set.)
    proposal: Option<RecomProposal>,
    /// The self-loop counts leading up to the proposal.
    counts: SelfLoopCounts,
    /// A sentinel used to kill the worker thread.
    terminate: bool,
}

/// Starts a thread that writes statistics from accepted plans to `stdout`.
///
/// Protocol (mirrors the tilted writer thread): a packet with `proposal:
/// Some(_)` is an accepted step and drives [`StatsWriter::step`]; a packet with
/// `proposal: None` and `terminate == false` is a pure self-loop batch and
/// drives [`StatsWriter::self_loop`] (used to flush the self-loops sampled after
/// the last accepted proposal); `terminate == true` ends the thread.
fn start_stats_thread(
    graph: &Graph,
    mut partition: Partition,
    mut writer: Box<dyn StatsWriter>,
    recv: Receiver<StepPacket>,
) {
    writer.init(graph, &partition).unwrap();
    let mut next: StepPacket = recv.recv().unwrap();
    while !next.terminate {
        if let Some(proposal) = next.proposal {
            partition.update(&proposal);
            writer
                .step(next.step, graph, &partition, &proposal, &next.counts)
                .unwrap();
        } else {
            writer
                .self_loop(next.step, graph, &partition, &next.counts)
                .unwrap();
        }
        next = recv.recv().unwrap();
    }
    writer.close().unwrap();
}

/// Stops a statistics writer thread.
fn stop_stats_thread(send: &Sender<StepPacket>) {
    send.send(StepPacket {
        step: 0,
        proposal: None,
        counts: SelfLoopCounts::default(),
        terminate: true,
    })
    .unwrap();
}

/// Starts a ReCom job thread.
/// ReCom job threads sample batches of proposals, which are then aggregated by
/// the main thread. (Thus, this function contains most of the ReCom chain logic.)
///
/// Arguments:
/// * `graph` - The graph associated with the chain.
/// * `partition` - The initial state of the chain.
/// * `params` - The chain parameters.
/// * `rng_seed` - The RNG seed for the job thread. (This should differ across threads.)
/// * `buf_size` - The buffer size for various chain buffers. This should usually be twice
///   the maximum possible district size (in nodes).
/// * `job_recv` - A Crossbeam channel for receiving batches of work from the main thread.
/// * `result_send` - A Crossbeam channel for sending completed batches to the main thread.
fn start_job_thread<C>(
    graph: &Graph,
    mut partition: Partition,
    params: RecomParams,
    rng_seed: u64,
    buf_size: usize,
    job_recv: Receiver<JobPacket>,
    result_send: Sender<ResultPacket>,
    constraint: C,
    mut constraint_state: C::State,
) where
    C: ChainConstraint + 'static,
{
    let n = graph.pops.len();
    let mut rng: SmallRng = SeedableRng::seed_from_u64(rng_seed);
    let mut buffers = WorkerBuffers::new((), n, buf_size, params.balance_ub);
    let mut st_sampler = make_sampler(&params, buf_size, &mut rng);
    let reversible = params.variant == RecomVariant::Reversible;

    let mut next: JobPacket = job_recv.recv().unwrap();
    while !next.terminate {
        if let Some(diff) = next.diff {
            partition.update(&diff);
            constraint.apply_proposal(graph, &mut constraint_state, &diff);
        }
        let mut counts = SelfLoopCounts::default();
        let mut proposals = Vec::<(u64, RecomProposal)>::new();
        for _ in 0..next.n_steps {
            // loop allows retries for non-reversible ReCom
            loop {
                // Step 1: sample a pair of adjacent districts.
                let (dist_a, dist_b) =
                    match sample_dist_pair(&graph, &mut partition, params.variant, &mut rng) {
                        Some((a, b)) => (a, b),
                        None => {
                            if reversible {
                                counts.inc(SelfLoopReason::NonAdjacent);
                                break; // success
                            } else {
                                continue; // retry
                            }
                        }
                    };
                partition.subgraph(&graph, &mut buffers.subgraph, dist_a, dist_b);

                // A disconnected merged district pair has no spanning tree.
                // Treat this as a rejection instead of panicking in the sampler.
                if !graph_connected_buffered(&buffers.subgraph.graph, &mut buffers.connectivity) {
                    if reversible {
                        counts.inc(SelfLoopReason::NoSplit);
                        break; // success
                    } else {
                        continue; // retry
                    }
                }

                // Step 2: draw a random spanning tree of the subgraph induced by the
                // two districts.
                st_sampler.random_spanning_tree_with_parent(
                    &buffers.subgraph.graph,
                    &graph,
                    &buffers.subgraph.raw_nodes,
                    &mut buffers.spanning_tree,
                    &mut rng,
                );

                // Step 3: choose a random balance edge, if possible.
                let split = random_split(
                    &buffers.subgraph.graph,
                    &graph,
                    &mut rng,
                    &buffers.spanning_tree.st,
                    dist_a,
                    dist_b,
                    &mut buffers.split,
                    &mut buffers.proposal,
                    &buffers.subgraph.raw_nodes,
                    &params,
                );
                match split {
                    Ok(n_splits) => {
                        if !constraint.proposal_valid(&graph, &constraint_state, &buffers.proposal)
                        {
                            counts.inc(SelfLoopReason::ConstraintViolation);
                            break;
                        }

                        if reversible {
                            // Step 4: accept any particular edge with probability 1 / (M * seam length)
                            let seam_length = buffers.proposal.seam_length(&graph);
                            let prob =
                                (n_splits as f64) / (seam_length as f64 * params.balance_ub as f64);
                            if prob > 1.0 {
                                panic!(
                                    "Invalid state: got {} splits, seam length {}",
                                    n_splits, seam_length
                                );
                            }
                            if rng.random::<f64>() < prob {
                                // the proposal needs to have a unique identifier so that when the
                                // packets finish, the selected plan is close to deterministic
                                // chance of a single batch getting duplicate numbers is near zero
                                // for batches of size < 1M and n_cores < 10k over a 1B run
                                proposals.push((rng.random::<u64>(), buffers.proposal.clone()));
                            } else {
                                counts.inc(SelfLoopReason::SeamLength);
                            }
                            break; // success
                        } else {
                            // Accept.
                            proposals.push((rng.random::<u64>(), buffers.proposal.clone()));
                            break; // success
                        }
                    }
                    Err(_) => {
                        if reversible {
                            counts.inc(SelfLoopReason::NoSplit); // TODO: break out errors?
                            break; // success
                        } else {
                            continue; // retry
                        }
                    }
                }
            }
        }
        result_send
            .send(ResultPacket {
                counts: counts,
                proposals: proposals,
            })
            .unwrap();
        next = job_recv.recv().unwrap();
    }
}

fn next_batch(send: &Sender<JobPacket>, diff: Option<RecomProposal>, batch_size: usize) {
    send.send(JobPacket {
        n_steps: batch_size,
        diff: diff,
        terminate: false,
    })
    .unwrap();
}

/// Stops a ReCom job thread.
fn stop_job_thread(send: &Sender<JobPacket>) {
    send.send(JobPacket {
        n_steps: 0,
        diff: None,
        terminate: true,
    })
    .unwrap();
}

/// Runs a multi-threaded ReCom chain.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The partition to start the chain run from (updated in place).
/// * `writer` - The statistics writer.
/// * `params` - The parameters of the ReCom chain run.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `batch_size` - The number of steps per unit of multithreaded work. This
///   parameter should be tuned according to the chain's average acceptance
///   probability: chains that reject most proposals (e.g. reversible ReCom
///   on large graphs) will benefit from large batches, but chains that accept
///   most or all proposals should use small batches.
pub fn multi_chain(
    graph: &Graph,
    partition: &Partition,
    writer: Box<dyn StatsWriter>,
    params: &RecomParams,
    n_threads: usize,
    batch_size: usize,
    show_progress: bool,
) -> Result<(), String> {
    multi_chain_with_constraint(
        graph,
        partition,
        writer,
        params,
        n_threads,
        batch_size,
        show_progress,
        crate::constraints::NoConstraint,
    )
}

/// Runs a multi-threaded ReCom chain.
///
/// # Arguments
///
/// * `graph` - The graph associated with `partition`.
/// * `partition` - The partition to start the chain run from (updated in place).
/// * `writer` - The statistics writer.
/// * `params` - The parameters of the ReCom chain run.
/// * `n_threads` - The number of worker threads (excluding the main thread).
/// * `batch_size` - The number of steps per unit of multithreaded work. This
/// * `constraint` - A chain constraint that can be used to restrict the proposals generated by
///     the chain.
///   parameter should be tuned according to the chain's average acceptance
///   probability: chains that reject most proposals (e.g. reversible ReCom
///   on large graphs) will benefit from large batches, but chains that accept
///   most or all proposals should use small batches.
pub fn multi_chain_with_constraint<C>(
    graph: &Graph,
    partition: &Partition,
    writer: Box<dyn StatsWriter>,
    params: &RecomParams,
    n_threads: usize,
    batch_size: usize,
    show_progress: bool,
    constraint: C,
) -> Result<(), String>
where
    C: ChainConstraint + Clone + Send + 'static,
{
    let init_constraint_state = constraint.init(graph, partition);
    if !constraint.is_valid_state(&init_constraint_state) {
        return Err("Initial partition does not satisfy the chain constraint".to_string());
    }

    let mut step = 0;
    let node_ub = node_bound(&graph.pops, params.max_pop);
    let mut job_sends = vec![]; // main thread sends work to job threads
    let mut job_recvs = vec![]; // job threads receive work from main thread
                                //
    for _ in 0..n_threads {
        let (s, r): (Sender<JobPacket>, Receiver<JobPacket>) = unbounded();
        job_sends.push(s);
        job_recvs.push(r);
    }

    // All job threads send a summary of chain results back to the main thread.
    let (result_send, result_recv): (Sender<ResultPacket>, Receiver<ResultPacket>) = unbounded();
    // The stats thread receives accepted proposals from the main thread.
    let (stats_send, stats_recv): (Sender<StepPacket>, Receiver<StepPacket>) =
        bounded(STATS_CHANNEL_CAPACITY);
    let mut rng: SmallRng = SeedableRng::seed_from_u64(params.rng_seed);

    // --- Progress bar setup ---
    let pb = if show_progress {
        let pb = ProgressBar::with_draw_target(
            Some(params.num_steps),
            indicatif::ProgressDrawTarget::stderr_with_hz(1),
        );
        pb.set_style(
            ProgressStyle::with_template(
                // {bar} with no width spec => indicatif auto-sizes to terminal width
                "[{elapsed_precise}] {bar:100.cyan/blue} {pos:>10}/{len} ({eta_precise})",
            )
            .unwrap()
            .progress_chars("##-"),
        );
        Some(pb)
    } else {
        None
    };
    let mut progress_count: u64 = 0;
    let mut last_drawn: u64 = 0;
    let progress_chunk: u64 = (params.num_steps / 1000).clamp(1, 1000);

    // Start job and stats threads.
    let scoped_result = scope(|scope| -> Result<(), String> {
        // Borrow the progress bar inside the scope.
        let pb_ref = &pb;

        // Start stats thread.
        scope.spawn({
            let partition = partition.clone();
            move |_| {
                start_stats_thread(graph, partition, writer, stats_recv);
            }
        });

        // Start job threads.
        for t_idx in 0..n_threads {
            // TODO: is this (+ t_idx) a sensible way to seed?
            let rng_seed = params.rng_seed + t_idx as u64 + 1;
            let job_recv = job_recvs[t_idx].clone();
            let result_send = result_send.clone();

            let worker_partition = partition.clone();
            let worker_constraint = constraint.clone();
            let worker_constraint_state = init_constraint_state.clone();

            scope.spawn(move |_| {
                start_job_thread(
                    graph,
                    worker_partition,
                    params.clone(),
                    rng_seed,
                    node_ub,
                    job_recv,
                    result_send,
                    worker_constraint,
                    worker_constraint_state,
                );
            });
        }

        // writer.init() writes the seed plan as record 0, so we run for one fewer
        // chain event to keep the total output record count equal to num_steps.
        let effective_steps = params.num_steps.saturating_sub(1);

        if effective_steps > 0 {
            for job in job_sends.iter() {
                job.send(JobPacket {
                    n_steps: batch_size,
                    diff: None,
                    terminate: false,
                })
                .unwrap();
            }
        }
        let mut sampled = SelfLoopCounts::default();
        let mut previously_accepted_proposal: Option<RecomProposal> = None;
        while step < effective_steps {
            let mut counts = SelfLoopCounts::default();
            let mut proposals = Vec::<(u64, RecomProposal)>::new();
            // This is where the proposals are assigned
            for _ in 0..n_threads {
                let packet: ResultPacket = result_recv.recv().unwrap();
                counts = counts + packet.counts;
                proposals.extend(packet.proposals);
            }

            let mut loops = counts.sum();
            if proposals.len() > 0 {
                // Sample events without replacement.
                proposals.sort_by(|a, b| a.0.cmp(&b.0));

                let mut total = loops + proposals.len();
                while total > 0 && step < effective_steps {
                    step += 1;

                    if let Some(pb) = pb_ref {
                        progress_count += 1;
                        if progress_count - last_drawn >= progress_chunk {
                            pb.set_position(progress_count);
                            last_drawn = progress_count;
                        }
                    }

                    let event = rng.random_range(0..total);
                    if event < loops {
                        // Case: no accepted proposal (don't need to update worker thread state).
                        sampled.inc(counts.index_and_dec(event).unwrap());
                        loops -= 1;
                    } else {
                        // Case: accepted proposal (update worker thread state).
                        if proposals.len() == 0 {
                            panic!("FATAL: Unreachable state in sampler (no proposals left).");
                        }
                        let proposal = &proposals[rng.random_range(0..proposals.len())];
                        for job in job_sends.iter() {
                            next_batch(job, Some(proposal.1.clone()), batch_size);
                        }
                        stats_send
                            .send(StepPacket {
                                step: step,
                                proposal: Some(proposal.1.clone()),
                                counts: sampled,
                                terminate: false,
                            })
                            .unwrap();
                        // Reset sampled rejection stats until the next accepted step.
                        sampled = SelfLoopCounts::default();
                        previously_accepted_proposal = Some(proposal.1.clone());
                        break;
                    }
                    total -= 1;
                }
            } else {
                // Clamp the batch to the run length still owed: a worker may
                // report more rejections than there are steps left, and counting
                // the overshoot would push the trailing self-loop frame past
                // `num_steps`. Move only the capped rejections into `sampled`
                // (the reason breakdown of the trimmed tail is irrelevant: this
                // branch fires only once no further proposal can be accepted, so
                // the capped `sampled` only ever feeds the residual self-loop
                // packet, which is consumed via `counts.sum()`).
                let remaining = effective_steps.saturating_sub(step) as usize;
                let capped = loops.min(remaining);
                for _ in 0..capped {
                    if let Some(reason) = counts.index_and_dec(0) {
                        sampled.inc(reason);
                    }
                }
                step += capped as u64;

                if let Some(pb) = pb_ref {
                    let remaining = effective_steps.saturating_sub(progress_count);
                    let inc = remaining.min(capped as u64);
                    progress_count += inc;
                    if progress_count - last_drawn >= progress_chunk {
                        pb.set_position(progress_count);
                        last_drawn = progress_count;
                    }
                }

                for job in job_sends.iter() {
                    next_batch(job, None, batch_size);
                }
            }
        }

        // Terminate worker threads.
        for job in job_sends.iter() {
            stop_job_thread(job);
        }
        // Flush self-loops sampled after the last accepted proposal: without
        // this, a chain ending on a run of rejections drops them and the output
        // record count falls below `num_steps`. `step == effective_steps` here,
        // so the packet satisfies the `self_loop` continuity contract shared by
        // the assignments/canonical writers (`step == last_step + counts.sum()`).
        // This runs even when nothing was ever accepted, so an all-rejection run
        // still emits the full requested chain length before the error below.
        if sampled.sum() > 0 {
            stats_send
                .send(StepPacket {
                    step: effective_steps,
                    proposal: None,
                    counts: sampled,
                    terminate: false,
                })
                .unwrap();
        }
        stop_stats_thread(&stats_send);
        if previously_accepted_proposal.is_none() {
            return Err("No proposals were accepted during the entire chain run. \
                This likely indicates that either the run was too short (so increase the value \
                of 'n-steps') or that the ReCom variant being used is too restrictive for the \
                graph and parameters chosen."
                .to_owned());
        }
        Ok(())
    });

    if let Some(pb) = pb {
        pb.set_position(params.num_steps);
        pb.finish_and_clear();
    }

    match scoped_result {
        Ok(inner) => inner, // inner: Result<(), String>

        // This only happens if some thread panicked.
        Err(_panic) => Err("multi_chain panicked in a worker thread".to_string()),
    }
}
