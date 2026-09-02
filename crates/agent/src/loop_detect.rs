//! Loop detection (spec §28): repeating sequences — same command, same
//! failure, same patch, same tool-call arguments, same retrieval query —
//! must stop and re-plan instead of repeating for 40 turns.

use std::collections::HashMap;

/// Tracks normalized keys of tool calls / errors. When the same key is seen
/// `threshold` times the detector trips.
#[derive(Debug, Clone)]
pub struct LoopDetector {
    threshold: usize,
    seen: HashMap<String, usize>,
    pub trips: u32,
}

impl LoopDetector {
    pub fn new(threshold: usize) -> Self {
        Self {
            threshold: threshold.max(2),
            seen: HashMap::new(),
            trips: 0,
        }
    }

    /// Normalize a tool call to a stable key: name + sorted args JSON.
    /// Whitespace/order-insensitive so "the same call" is recognized.
    pub fn tool_key(name: &str, args: &serde_json::Value) -> String {
        let normalized = normalize_value(args);
        format!("{name} {normalized}")
    }

    /// Register a tool call; returns true when the loop threshold trips.
    pub fn record_tool_call(&mut self, name: &str, args: &serde_json::Value) -> bool {
        let key = Self::tool_key(name, args);
        self.record(key)
    }

    /// Register an error; returns true when the same error repeats.
    pub fn record_error(&mut self, message: &str) -> bool {
        let key = format!("err {}", message.trim());
        self.record(key)
    }

    fn record(&mut self, key: String) -> bool {
        if key.len() > 4096 {
            return false; // hostile oversized keys never trip the detector
        }
        let count = self.seen.entry(key).or_insert(0);
        *count += 1;
        if *count >= self.threshold {
            self.trips += 1;
            self.seen.clear(); // a trip resets the window
            true
        } else {
            false
        }
    }

    pub fn threshold(&self) -> usize {
        self.threshold
    }

    pub fn count(&self, name: &str, args: &serde_json::Value) -> usize {
        self.seen
            .get(&Self::tool_key(name, args))
            .copied()
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// Canonical sort of JSON keys (recursive) for stable keys.
fn normalize_value(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut pairs: Vec<(String, String)> = map
                .iter()
                .map(|(k, val)| (k.clone(), normalize_value(val)))
                .collect();
            pairs.sort();
            let inner: Vec<String> = pairs
                .into_iter()
                .map(|(k, val)| format!("{k}:{val}"))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        serde_json::Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(normalize_value).collect();
            format!("[{}]", inner.join(","))
        }
        serde_json::Value::String(s) => format!("\"{s}\""),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_calls_trip_at_threshold() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"path": "a.rs"});
        assert!(!d.record_tool_call("read_file", &args));
        assert!(!d.record_tool_call("read_file", &args));
        assert!(
            d.record_tool_call("read_file", &args),
            "3rd identical call trips"
        );
        assert_eq!(d.trips, 1);
    }

    #[test]
    fn whitespace_and_key_order_are_insensitive() {
        let a = serde_json::json!({"path": "a.rs", "limit": 10});
        let b = serde_json::json!({"limit": 10, "path": "a.rs"});
        assert_eq!(
            LoopDetector::tool_key("read_file", &a),
            LoopDetector::tool_key("read_file", &b)
        );
        let mut d = LoopDetector::new(2);
        assert!(!d.record_tool_call("read_file", &a));
        assert!(
            d.record_tool_call("read_file", &b),
            "same call, different key order"
        );
    }

    #[test]
    fn different_args_do_not_trip() {
        let mut d = LoopDetector::new(3);
        for i in 0..20 {
            let args = serde_json::json!({"path": format!("f{i}.rs")});
            assert!(
                !d.record_tool_call("read_file", &args),
                "distinct calls must never trip"
            );
        }
        assert_eq!(d.trips, 0);
        assert_eq!(d.len(), 20);
    }

    #[test]
    fn same_call_different_tool_does_not_trip() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"path": "x"});
        d.record_tool_call("read_file", &args);
        d.record_tool_call("grep", &args);
        d.record_tool_call("write_file", &args);
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn error_repeats_trip() {
        let mut d = LoopDetector::new(2);
        assert!(!d.record_error("cargo build failed: E0308"));
        assert!(d.record_error("cargo build failed: E0308"));
        // Slightly different error text does not trip.
        let mut d = LoopDetector::new(2);
        d.record_error("E0308");
        assert!(!d.record_error("E0425"), "different error = different loop");
    }

    #[test]
    fn hostile_oversized_keys_never_trip() {
        let mut d = LoopDetector::new(2);
        let big = serde_json::json!({"payload": "x".repeat(5000)});
        assert!(
            !d.record_tool_call("t", &big),
            "oversized key must not trip"
        );
        assert_eq!(d.trips, 0);
    }

    #[test]
    fn trip_resets_window() {
        let mut d = LoopDetector::new(3);
        let args = serde_json::json!({"a": 1});
        d.record_tool_call("t", &args);
        d.record_tool_call("t", &args);
        assert!(d.record_tool_call("t", &args));
        assert_eq!(d.count("t", &args), 0, "window reset after trip");
        assert!(
            !d.record_tool_call("t", &args),
            "fresh window needs 3 again"
        );
    }

    #[test]
    fn json_normalization_is_stable() {
        let a = serde_json::json!({"z": [1, 2], "a": {"y": "s", "x": null}});
        let b = serde_json::json!({"a": {"x": null, "y": "s"}, "z": [1, 2]});
        assert_eq!(normalize_value(&a), normalize_value(&b));
    }
}
