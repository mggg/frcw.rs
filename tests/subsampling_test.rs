use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Write};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use binary_ensemble::io::bundle::BendlReader;
use binary_ensemble::io::reader::BenStreamReader;
use serde_json::Value;

use rustrecom::graph::Graph;
use rustrecom::partition::Partition;
use rustrecom::recom::RecomProposal;
use rustrecom::stats::{
    AssignmentsOnlyWriter, BenWriter, BendlBenStreamWriter, CanonicalWriter, PcompressWriter,
    SelfLoopCounts, SelfLoopReason, StatsWriter,
};

#[derive(Clone)]
struct SharedBytes(Arc<Mutex<Vec<u8>>>);

impl Write for SharedBytes {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn memory_output() -> (Box<dyn Write + Send>, Arc<Mutex<Vec<u8>>>) {
    let bytes = Arc::new(Mutex::new(Vec::new()));
    (Box::new(SharedBytes(bytes.clone())), bytes)
}

fn partition(graph: &Graph, assignments: &[u32]) -> Partition {
    Partition::from_assignments(graph, &assignments.to_vec()).unwrap()
}

fn loops(count: usize) -> SelfLoopCounts {
    let mut counts = SelfLoopCounts::default();
    counts.inc_by(SelfLoopReason::NoSplit, count);
    counts
}

fn drive_writer(writer: &mut dyn StatsWriter) -> Vec<Vec<u32>> {
    let graph = Graph::rect_grid(2, 2);
    let plans = [
        vec![0, 0, 1, 1],
        vec![0, 1, 0, 1],
        vec![0, 1, 1, 0],
        vec![1, 0, 0, 1],
    ];
    let partitions = plans
        .iter()
        .map(|assignment| partition(&graph, assignment))
        .collect::<Vec<_>>();
    let proposal = RecomProposal {
        a_label: 0,
        b_label: 1,
        a_pop: 2,
        b_pop: 2,
        a_nodes: vec![],
        b_nodes: vec![],
    };

    writer.init(&graph, &partitions[0]).unwrap();
    writer
        .step(3, &graph, &partitions[1], &proposal, &loops(2))
        .unwrap();
    writer
        .step(4, &graph, &partitions[2], &proposal, &loops(0))
        .unwrap();
    writer
        .step(8, &graph, &partitions[3], &proposal, &loops(3))
        .unwrap();
    writer
        .self_loop(12, &graph, &partitions[3], &loops(4))
        .unwrap();
    writer.close().unwrap();

    vec![
        plans[0].clone(),
        plans[2].clone(),
        plans[3].clone(),
        plans[3].clone(),
    ]
}

fn decode_ben(bytes: Vec<u8>) -> Vec<Vec<u32>> {
    let mut reader = BenStreamReader::from_ben(Cursor::new(bytes)).unwrap();
    let mut assignments: Vec<Vec<u32>> = Vec::new();
    reader
        .for_each_assignment(|assignment, count| {
            for _ in 0..count {
                assignments.push(assignment.iter().map(|&label| label as u32).collect());
            }
            Ok(true)
        })
        .unwrap();
    assignments
}

#[test]
fn text_writers_keep_original_sample_positions() {
    let (output, bytes) = memory_output();
    let mut writer = AssignmentsOnlyWriter::new(false, output).with_sample_interval(4);
    let expected = drive_writer(&mut writer);
    let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    let rows = output
        .lines()
        .map(|line| {
            let (step, assignment) = line.split_once(',').unwrap();
            (
                step.parse::<u64>().unwrap(),
                serde_json::from_str::<Vec<u32>>(assignment).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows.iter().map(|(step, _)| *step).collect::<Vec<_>>(),
        vec![0, 4, 8, 12]
    );
    assert_eq!(
        rows.into_iter()
            .map(|(_, assignment)| assignment)
            .collect::<Vec<_>>(),
        expected
    );

    let (output, bytes) = memory_output();
    let mut writer = CanonicalWriter::new(output).with_sample_interval(4);
    let expected = drive_writer(&mut writer);
    let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
    let rows = output
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        rows.iter()
            .map(|row| row["sample"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 5, 9, 13]
    );
    assert_eq!(
        rows.into_iter()
            .map(|row| {
                row["assignment"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|label| label.as_u64().unwrap() as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn compressed_writers_emit_only_selected_plans() {
    let (output, bytes) = memory_output();
    let mut writer = BenWriter::new(output).with_sample_interval(4);
    let expected = drive_writer(&mut writer);
    assert_eq!(decode_ben(bytes.lock().unwrap().clone()), expected);

    let (output, bytes) = memory_output();
    let mut writer = PcompressWriter::new(output).with_sample_interval(4);
    let expected = drive_writer(&mut writer);
    let bytes = bytes.lock().unwrap().clone();
    let mut reader = BufReader::new(Cursor::new(bytes));
    let mut decoded = Vec::new();
    {
        let mut output = BufWriter::new(&mut decoded);
        pcompress::decode::decode(&mut reader, &mut output, 0, false);
    }
    let assignments = String::from_utf8(decoded)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Vec<u32>>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(assignments, expected);
}

#[test]
fn bendl_header_counts_subsampled_stream() {
    let mut path = std::env::temp_dir();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "rustrecom_subsampling_{}_{}.bendl",
        std::process::id(),
        timestamp
    ));

    let output = BufWriter::new(File::create(&path).unwrap());
    let mut writer =
        BendlBenStreamWriter::new(output, b"{}".to_vec(), b"{}".to_vec()).with_sample_interval(4);
    let expected = drive_writer(&mut writer);

    let file = File::open(&path).unwrap();
    let mut bundle = BendlReader::open(file).unwrap();
    assert_eq!(bundle.sample_count(), Some(expected.len() as i64));
    let stream = bundle.assignment_stream_reader_unverified().unwrap();
    let mut reader = BenStreamReader::from_ben(stream).unwrap();
    let mut assignments: Vec<Vec<u32>> = Vec::new();
    reader
        .for_each_assignment(|assignment, count| {
            for _ in 0..count {
                assignments.push(assignment.iter().map(|&label| label as u32).collect());
            }
            Ok(true)
        })
        .unwrap();
    assert_eq!(assignments, expected);

    std::fs::remove_file(path).unwrap();
}
