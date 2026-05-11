use approx::assert_relative_eq;
use frcw::graph::Graph;
use frcw::objectives::{make_objective_fn, required_edge_cols, required_node_cols};
use frcw::partition::Partition;

fn path_graph_with_attrs(
    assignments: Vec<u32>,
    node_attrs: Vec<(&str, Vec<&str>)>,
    edge_attrs: Vec<(&str, Vec<f64>)>,
) -> (Graph, Partition) {
    let n = assignments.len();
    assert!(n >= 2, "test helper requires at least two nodes");

    let edge_list = (0..n - 1)
        .map(|i| format!("{i} {}", i + 1))
        .collect::<Vec<String>>()
        .join("\n");
    let populations = vec!["1"; n].join(" ");

    let mut graph = Graph::from_edge_list(&edge_list, &populations).unwrap();
    for (key, values) in node_attrs {
        graph.attr.insert(
            key.to_string(),
            values.into_iter().map(|v| v.to_string()).collect(),
        );
    }
    for (key, values) in edge_attrs {
        graph.edge_attr.insert(key.to_string(), values);
    }

    let partition = Partition::from_assignments(&graph, &assignments).unwrap();
    (graph, partition)
}

#[test]
fn test_gingles_partial_scores_fractional_next_district() {
    let (graph, partition) = path_graph_with_attrs(
        vec![1, 1, 2, 2],
        vec![
            ("BVAP", vec!["60", "60", "40", "40"]),
            ("VAP", vec!["100", "100", "100", "100"]),
        ],
        vec![],
    );

    let config =
        r#"{"objective":"gingles_partial","threshold":0.5,"min_pop":"BVAP","total_pop":"VAP"}"#;
    let obj_fn = make_objective_fn(config);

    // District shares are 0.6 and 0.4, so score = 1 + (0.4 / 0.5) = 1.8.
    assert_relative_eq!(obj_fn(&graph, &partition), 1.8, epsilon = 1e-12);
}

#[test]
fn test_election_wins_respects_target_party() {
    let (graph, partition) = path_graph_with_attrs(
        vec![1, 1, 2, 2, 3, 3],
        vec![
            ("votes_a", vec!["20", "20", "30", "30", "35", "35"]),
            ("votes_b", vec!["30", "30", "20", "20", "15", "15"]),
        ],
        vec![],
    );

    let obj_a = make_objective_fn(
        r#"{"objective":"election_wins","elections":[{"votes_a":"votes_a","votes_b":"votes_b"}],"target":"a","aggregation":"mean"}"#,
    );
    let obj_b = make_objective_fn(
        r#"{"objective":"election_wins","elections":[{"votes_a":"votes_a","votes_b":"votes_b"}],"target":"b","aggregation":"mean"}"#,
    );

    // Target a: 2 wins, best loss share 0.4 => 2.8.
    // Target b: 1 win, best loss share 0.4 => 1.8.
    assert_relative_eq!(obj_a(&graph, &partition), 2.8, epsilon = 1e-12);
    assert_relative_eq!(obj_b(&graph, &partition), 1.8, epsilon = 1e-12);
}

#[test]
fn test_election_wins_aggregates_multiple_elections() {
    let (graph, partition) = path_graph_with_attrs(
        vec![1, 1, 2, 2, 3, 3],
        vec![
            ("e1_a", vec!["30", "30", "20", "25", "25", "25"]),
            ("e1_b", vec!["20", "20", "25", "30", "25", "25"]),
            ("e2_a", vec!["20", "20", "30", "30", "35", "35"]),
            ("e2_b", vec!["30", "30", "20", "20", "15", "15"]),
        ],
        vec![],
    );

    let mean_obj = make_objective_fn(
        r#"{"objective":"election_wins","elections":[{"votes_a":"e1_a","votes_b":"e1_b"},{"votes_a":"e2_a","votes_b":"e2_b"}],"target":"a","aggregation":"mean"}"#,
    );
    let min_obj = make_objective_fn(
        r#"{"objective":"election_wins","elections":[{"votes_a":"e1_a","votes_b":"e1_b"},{"votes_a":"e2_a","votes_b":"e2_b"}],"target":"a","aggregation":"min"}"#,
    );
    let sum_obj = make_objective_fn(
        r#"{"objective":"election_wins","elections":[{"votes_a":"e1_a","votes_b":"e1_b"},{"votes_a":"e2_a","votes_b":"e2_b"}],"target":"a","aggregation":"sum"}"#,
    );

    let one_below = f64::from_bits(1.0f64.to_bits() - 1);
    let e1 = 1.0 + one_below;
    let e2 = 2.8;

    assert_relative_eq!(
        mean_obj(&graph, &partition),
        (e1 + e2) / 2.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(min_obj(&graph, &partition), e1, epsilon = 1e-12);
    assert_relative_eq!(sum_obj(&graph, &partition), e1 + e2, epsilon = 1e-12);
}

#[test]
fn test_polsby_popper_uses_edge_attributes() {
    let (graph, partition) = path_graph_with_attrs(
        vec![1, 1, 2, 2],
        vec![
            ("area", vec!["1.0", "1.0", "1.0", "1.0"]),
            ("perim", vec!["4.0", "4.0", "4.0", "4.0"]),
        ],
        vec![("shared_perim", vec![1.0, 1.0, 1.0])],
    );

    let mean_obj = make_objective_fn(
        r#"{"objective":"polsby_popper","area_col":"area","perim_col":"perim","shared_perim_col":"shared_perim","aggregation":"mean"}"#,
    );
    let sum_obj = make_objective_fn(
        r#"{"objective":"polsby_popper","area_col":"area","perim_col":"perim","shared_perim_col":"shared_perim","aggregation":"sum"}"#,
    );

    // Each district has area 2 and perimeter 6, so PP = 4*pi*2/6^2 = 2*pi/9.
    let per_district = 2.0 * std::f64::consts::PI / 9.0;
    assert_relative_eq!(mean_obj(&graph, &partition), per_district, epsilon = 1e-12);
    assert_relative_eq!(
        sum_obj(&graph, &partition),
        2.0 * per_district,
        epsilon = 1e-12
    );
}

#[test]
fn test_required_columns_match_objective_configs() {
    let gingles =
        r#"{"objective":"gingles_partial","threshold":0.5,"min_pop":"BVAP","total_pop":"VAP"}"#;
    assert_eq!(required_node_cols(gingles), vec!["BVAP", "VAP"]);
    assert!(required_edge_cols(gingles).is_empty());

    let elections = r#"{"objective":"election_wins","elections":[{"votes_a":"DEM_GOV","votes_b":"REP_GOV"},{"votes_a":"DEM_SEN","votes_b":"REP_SEN"}],"target":"a","aggregation":"mean"}"#;
    assert_eq!(
        required_node_cols(elections),
        vec!["DEM_GOV", "REP_GOV", "DEM_SEN", "REP_SEN"]
    );
    assert!(required_edge_cols(elections).is_empty());

    let polsby = r#"{"objective":"polsby_popper","area_col":"area","perim_col":"perim","shared_perim_col":"shared_perim","aggregation":"mean"}"#;
    assert_eq!(required_node_cols(polsby), vec!["area", "perim"]);
    assert_eq!(required_edge_cols(polsby), vec!["shared_perim"]);
}
