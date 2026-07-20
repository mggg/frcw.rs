//! Helpers for parsing JSON configuration strings.
//!
//! Each subcommand's config format mirrors its CLI surface: field names match
//! flag names, `variant` uses the CLI spellings, only the CLI-required
//! arguments are required, and every optional field defaults exactly like its
//! flag. A `version`/`command` envelope keeps the format evolvable, and
//! unknown fields are rejected so typos (and configs written for a newer
//! version) fail loudly instead of being silently ignored.

use serde::Deserialize;
use serde_json::{from_str, Value};
use std::collections::HashMap;

const CONFIG_VERSION: u64 = 1;

fn default_one_usize() -> usize {
    1
}

fn default_writer() -> String {
    "jsonl".to_string()
}

fn default_bendl_graph_order() -> String {
    "none".to_string()
}

fn default_true() -> bool {
    true
}

fn default_optimizer_variant() -> String {
    "district-pairs-rmst".to_string()
}

fn default_optimizer_writer() -> String {
    "assignments".to_string()
}

fn default_accept_rule() -> String {
    "linear".to_string()
}

/// A versioned `rustrecom chain` run configuration. Field names and defaults
/// mirror the chain subcommand's CLI arguments one-for-one.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChainV1Config {
    pub version: u64,
    pub command: String,
    // Required, exactly like the CLI's required arguments.
    pub graph_json: String,
    pub n_steps: u64,
    pub tol: f64,
    pub pop_col: String,
    pub assignment_col: String,
    pub rng_seed: u64,
    pub variant: String,
    // Optional, defaulting exactly like the CLI.
    #[serde(default)]
    pub target_pop: Option<u64>,
    #[serde(default)]
    pub balance_ub: u32,
    #[serde(default = "default_one_usize")]
    pub n_threads: usize,
    #[serde(default = "default_one_usize")]
    pub batch_size: usize,
    #[serde(default = "default_writer")]
    pub writer: String,
    #[serde(default)]
    pub sum_cols: Vec<String>,
    #[serde(default)]
    pub region_weights: HashMap<String, f64>,
    #[serde(default)]
    pub edge_weight_keys: Vec<String>,
    #[serde(default)]
    pub cut_edges_count: bool,
    #[serde(default)]
    pub output_file: Option<String>,
    #[serde(default = "default_bendl_graph_order")]
    pub bendl_graph_order: String,
    #[serde(default)]
    pub show_progress: bool,
    #[serde(default)]
    pub constraint: Option<Value>,
}

#[derive(Debug, PartialEq)]
pub struct LoadedChainConfig {
    /// The exact input string, preserved byte-for-byte for provenance.
    pub raw: String,
    pub document: ChainV1Config,
}

/// The envelope is checked before the full parse so a config written for a
/// different version or command reports that mismatch instead of a confusing
/// unknown-field error.
#[derive(Deserialize)]
struct ConfigEnvelope {
    version: u64,
    command: String,
}

/// Parses the raw string as JSON and checks the version/command envelope for
/// `command`, returning the full document for the per-command deserialize.
fn checked_config_value(raw: &str, command: &str) -> Result<Value, String> {
    let value: Value =
        serde_json::from_str(raw).map_err(|error| format!("Invalid config JSON: {error}"))?;
    if !value.is_object() {
        return Err("config must be a JSON object".to_string());
    }
    let envelope: ConfigEnvelope = serde_json::from_value(value.clone()).map_err(|error| {
        format!("config must carry integer 'version' and string 'command' fields: {error}")
    })?;
    if envelope.version != CONFIG_VERSION {
        return Err(format!("Unsupported config version {}", envelope.version));
    }
    if envelope.command != command {
        return Err(format!(
            "Expected command '{}', got '{}'",
            command, envelope.command
        ));
    }
    Ok(value)
}

pub fn parse_chain_config(raw: &str) -> Result<LoadedChainConfig, String> {
    let value = checked_config_value(raw, "chain")?;
    let document: ChainV1Config =
        serde_json::from_value(value).map_err(|error| format!("Invalid chain config: {error}"))?;
    Ok(LoadedChainConfig {
        raw: raw.to_string(),
        document,
    })
}

/// A versioned `rustrecom short-bursts` run configuration. Field names and
/// defaults mirror the short-bursts subcommand's CLI arguments one-for-one;
/// `objective` carries the objective spec as an inline JSON object.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShortBurstsV1Config {
    pub version: u64,
    pub command: String,
    // Required, exactly like the CLI's required arguments.
    pub graph_json: String,
    pub n_steps: u64,
    pub tol: f64,
    pub pop_col: String,
    pub assignment_col: String,
    pub rng_seed: u64,
    pub burst_length: usize,
    pub objective: Value,
    // Optional, defaulting exactly like the CLI.
    #[serde(default = "default_true")]
    pub maximize: bool,
    #[serde(default = "default_one_usize")]
    pub n_threads: usize,
    #[serde(default = "default_optimizer_variant")]
    pub variant: String,
    #[serde(default = "default_optimizer_writer")]
    pub writer: String,
    #[serde(default)]
    pub sum_cols: Vec<String>,
    #[serde(default)]
    pub partial_sum_cols: Vec<String>,
    #[serde(default)]
    pub region_weights: HashMap<String, f64>,
    #[serde(default)]
    pub edge_weight_keys: Vec<String>,
    #[serde(default)]
    pub output_file: Option<String>,
    #[serde(default)]
    pub scores_output_file: Option<String>,
    #[serde(default = "default_bendl_graph_order")]
    pub bendl_graph_order: String,
    #[serde(default)]
    pub show_progress: bool,
    #[serde(default)]
    pub write_improved_scores_only: bool,
}

#[derive(Debug, PartialEq)]
pub struct LoadedShortBurstsConfig {
    /// The exact input string, preserved byte-for-byte for provenance.
    pub raw: String,
    pub document: ShortBurstsV1Config,
}

pub fn parse_short_bursts_config(raw: &str) -> Result<LoadedShortBurstsConfig, String> {
    let value = checked_config_value(raw, "short-bursts")?;
    let document: ShortBurstsV1Config = serde_json::from_value(value)
        .map_err(|error| format!("Invalid short-bursts config: {error}"))?;
    Ok(LoadedShortBurstsConfig {
        raw: raw.to_string(),
        document,
    })
}

/// A versioned `rustrecom tilted` run configuration. Field names and defaults
/// mirror the tilted subcommand's CLI arguments one-for-one; `objective`
/// carries the objective spec as an inline JSON object.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TiltedV1Config {
    pub version: u64,
    pub command: String,
    // Required, exactly like the CLI's required arguments.
    pub graph_json: String,
    pub n_steps: u64,
    pub tol: f64,
    pub pop_col: String,
    pub assignment_col: String,
    pub rng_seed: u64,
    pub objective: Value,
    // Optional, defaulting exactly like the CLI.
    #[serde(default = "default_accept_rule")]
    pub accept_rule: String,
    #[serde(default)]
    pub accept_worse_prob: Option<f64>,
    #[serde(default)]
    pub acceptance_beta: Option<f64>,
    #[serde(default = "default_true")]
    pub maximize: bool,
    #[serde(default = "default_one_usize")]
    pub n_threads: usize,
    #[serde(default = "default_optimizer_variant")]
    pub variant: String,
    #[serde(default = "default_optimizer_writer")]
    pub writer: String,
    #[serde(default)]
    pub sum_cols: Vec<String>,
    #[serde(default)]
    pub partial_sum_cols: Vec<String>,
    #[serde(default)]
    pub region_weights: HashMap<String, f64>,
    #[serde(default)]
    pub edge_weight_keys: Vec<String>,
    #[serde(default)]
    pub output_file: Option<String>,
    #[serde(default)]
    pub scores_output_file: Option<String>,
    #[serde(default = "default_bendl_graph_order")]
    pub bendl_graph_order: String,
    #[serde(default)]
    pub show_progress: bool,
    #[serde(default)]
    pub write_improved_scores_only: bool,
}

#[derive(Debug, PartialEq)]
pub struct LoadedTiltedConfig {
    /// The exact input string, preserved byte-for-byte for provenance.
    pub raw: String,
    pub document: TiltedV1Config,
}

pub fn parse_tilted_config(raw: &str) -> Result<LoadedTiltedConfig, String> {
    let value = checked_config_value(raw, "tilted")?;
    let document: TiltedV1Config =
        serde_json::from_value(value).map_err(|error| format!("Invalid tilted config: {error}"))?;
    Ok(LoadedTiltedConfig {
        raw: raw.to_string(),
        document,
    })
}

pub fn region_weights_from_map(
    region_weights: &HashMap<String, f64>,
) -> Option<Vec<(String, f64)>> {
    if region_weights.is_empty() {
        return None;
    }
    let mut weights = region_weights
        .iter()
        .map(|(key, value)| (key.clone(), *value))
        .collect::<Vec<_>>();
    weights.sort_by(|left, right| right.1.total_cmp(&left.1));
    Some(weights)
}

pub fn parse_region_weights_config(region_weights_raw: &str) -> Option<Vec<(String, f64)>> {
    match region_weights_raw {
        "" => None,
        raw => {
            let weights = from_str::<HashMap<String, f64>>(raw).unwrap();
            region_weights_from_map(&weights)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The smallest valid config: envelope plus the CLI-required arguments.
    fn minimal_config() -> Value {
        json!({
            "version": 1,
            "command": "chain",
            "graph_json": "/graph.json",
            "n_steps": 10,
            "tol": 0.05,
            "pop_col": "TOTPOP",
            "assignment_col": "district",
            "rng_seed": 42,
            "variant": "district-pairs-rmst"
        })
    }

    #[test]
    fn minimal_config_takes_cli_defaults() {
        let raw = minimal_config().to_string();
        let loaded = parse_chain_config(&raw).unwrap();
        assert_eq!(loaded.raw, raw);
        let document = loaded.document;
        assert_eq!(document.variant, "district-pairs-rmst");
        assert_eq!(document.n_threads, 1);
        assert_eq!(document.batch_size, 1);
        assert_eq!(document.writer, "jsonl");
        assert_eq!(document.bendl_graph_order, "none");
        assert_eq!(document.balance_ub, 0);
        assert_eq!(document.target_pop, None);
        assert_eq!(document.output_file, None);
        assert_eq!(document.constraint, None);
        assert!(document.sum_cols.is_empty());
        assert!(!document.cut_edges_count);
        assert!(!document.show_progress);
    }

    #[test]
    fn rejects_invalid_envelopes() {
        assert!(parse_chain_config("not json")
            .unwrap_err()
            .contains("Invalid config JSON"));
        assert!(parse_chain_config("[]")
            .unwrap_err()
            .contains("JSON object"));

        let mut config = minimal_config();
        config["version"] = json!(1.5);
        assert!(parse_chain_config(&config.to_string())
            .unwrap_err()
            .contains("'version'"));
        config["version"] = json!(2);
        assert!(parse_chain_config(&config.to_string())
            .unwrap_err()
            .contains("Unsupported config version 2"));
        config["version"] = json!(1);
        config["command"] = json!("tilted");
        assert!(parse_chain_config(&config.to_string())
            .unwrap_err()
            .contains("Expected command 'chain'"));
    }

    #[test]
    fn rejects_missing_required_and_unknown_fields() {
        let mut config = minimal_config();
        config.as_object_mut().unwrap().remove("pop_col");
        assert!(parse_chain_config(&config.to_string())
            .unwrap_err()
            .contains("missing field `pop_col`"));

        // Typos and fields from a newer format fail loudly instead of being
        // silently dropped.
        let mut config = minimal_config();
        config["pop_tol"] = json!(0.05);
        assert!(parse_chain_config(&config.to_string())
            .unwrap_err()
            .contains("unknown field `pop_tol`"));
    }

    #[test]
    fn sorts_region_weights_by_priority() {
        let weights = HashMap::from([
            ("county".to_string(), 2.0),
            ("municipality".to_string(), 5.0),
        ]);
        assert_eq!(
            region_weights_from_map(&weights),
            Some(vec![
                ("municipality".to_string(), 5.0),
                ("county".to_string(), 2.0),
            ])
        );
    }
}
