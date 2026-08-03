//! Electrical semantics extraction for KiCad netlists.
//!
//! Provides net classification (power, ground, signal, no-connect) and
//! differential pair identification based on net names, net classes, and
//! connected pin types.

use crate::models::{ElectricalPinType, NetCategory, NetRecord};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

// Pre-compiled regexes compiled once at first use.
static GND_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(gnd|agnd|dgnd|vss|avss|dvss|0v|earth|gnd_\w+)$").unwrap()
});

static POWER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^(\+?\d+v\d*|vcc|vdd|vbat|vbus|pvcc|vref|\+3v3|\+5v|\+12v|\+1v8|vraw|v_\w+|vdd_\w+|vcc_\w+|vdda|vssa|avdd|avss|dvdd|dvss|vin|vout|pvdd|vccio|vddio|vdd_core|vcc_mcu|vcc_io)$",
    )
    .unwrap()
});

static DIFF_P_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(.*?)(?:_P|\+|\_POS|\_HIGH)$").unwrap());

static DIFF_N_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(.*?)(?:_N|\-|\_NEG|\_LOW)$").unwrap());

/// Extractor for electrical semantics: net classification and differential pair detection.
pub struct SemanticsExtractor {
    gnd_regex: &'static Regex,
    power_regex: &'static Regex,
    diff_p_regex: &'static Regex,
    diff_n_regex: &'static Regex,
}

impl SemanticsExtractor {
    /// Creates a new extractor using pre-compiled regexes.
    pub fn new() -> Self {
        Self {
            gnd_regex: &GND_REGEX,
            power_regex: &POWER_REGEX,
            diff_p_regex: &DIFF_P_REGEX,
            diff_n_regex: &DIFF_N_REGEX,
        }
    }

    /// Classifies a net into Ground, Power, Signal, or NoConnect based on net name,
    /// netclass, and connected pin types.
    ///
    /// Priority order: NoConnect → Ground → Power → Signal.
    pub fn classify_net(
        &self,
        net_name: &str,
        net_class: &str,
        pin_types: &[ElectricalPinType],
    ) -> NetCategory {
        let name_trimmed = net_name.trim();

        // 1. Unconnected / NoConnect
        if name_trimmed.is_empty()
            || name_trimmed.starts_with("unconnected-")
            || name_trimmed == "NC"
        {
            return NetCategory::NoConnect;
        }

        // 2. Ground heuristics
        if self.gnd_regex.is_match(name_trimmed)
            || net_class.eq_ignore_ascii_case("GND")
            || net_class.eq_ignore_ascii_case("GROUND")
        {
            return NetCategory::Ground;
        }

        // 3. Power heuristics
        if self.power_regex.is_match(name_trimmed)
            || net_class.eq_ignore_ascii_case("POWER")
            || net_class.eq_ignore_ascii_case("PWR")
            || pin_types.contains(&ElectricalPinType::PowerIn)
            || pin_types.contains(&ElectricalPinType::PowerOut)
        {
            return NetCategory::Power;
        }

        // 4. Default to Signal
        NetCategory::Signal
    }

    /// Identifies differential pair net partners across a list of net records.
    ///
    /// Matches nets ending in `_P`/`+`/`_POS`/`_HIGH` with nets ending in
    /// `_N`/`-`/`_NEG`/`_LOW` that share the same prefix. Sets
    /// `is_differential_pair` and `differential_pair_partner` on matched nets.
    pub fn identify_differential_pairs(&self, nets: &mut [NetRecord]) {
        // Map prefix -> Net Name for positive and negative signals
        let mut pos_map: HashMap<String, String> = HashMap::new();
        let mut neg_map: HashMap<String, String> = HashMap::new();

        for net in nets.iter() {
            if net.category != NetCategory::Signal {
                continue;
            }
            if let Some(caps) = self.diff_p_regex.captures(&net.name) {
                let prefix = caps.get(1).unwrap().as_str().to_string();
                pos_map.insert(prefix, net.name.clone());
            } else if let Some(caps) = self.diff_n_regex.captures(&net.name) {
                let prefix = caps.get(1).unwrap().as_str().to_string();
                neg_map.insert(prefix, net.name.clone());
            }
        }

        // Match pairs sharing the exact prefix
        let mut diff_pairs: HashMap<String, String> = HashMap::new();
        for (prefix, pos_net) in &pos_map {
            if let Some(neg_net) = neg_map.get(prefix) {
                diff_pairs.insert(pos_net.clone(), neg_net.clone());
                diff_pairs.insert(neg_net.clone(), pos_net.clone());
            }
        }

        for net in nets.iter_mut() {
            if let Some(partner) = diff_pairs.get(&net.name) {
                net.is_differential_pair = true;
                net.differential_pair_partner = Some(partner.clone());
            }
        }
    }
}

impl Default for SemanticsExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_net_classification() {
        let extractor = SemanticsExtractor::new();
        assert_eq!(
            extractor.classify_net("GND", "Default", &[]),
            NetCategory::Ground
        );
        assert_eq!(
            extractor.classify_net("AGND", "Default", &[]),
            NetCategory::Ground
        );
        assert_eq!(
            extractor.classify_net("DVSS", "Default", &[]),
            NetCategory::Ground
        );
        assert_eq!(
            extractor.classify_net("+3V3", "Default", &[]),
            NetCategory::Power
        );
        assert_eq!(
            extractor.classify_net("VCC_MCU", "Default", &[]),
            NetCategory::Power
        );
        assert_eq!(
            extractor.classify_net("AVDD", "Default", &[]),
            NetCategory::Power
        );
        assert_eq!(
            extractor.classify_net("VDDIO", "Default", &[]),
            NetCategory::Power
        );
        assert_eq!(
            extractor.classify_net("VIN", "Default", &[]),
            NetCategory::Power
        );
        assert_eq!(
            extractor.classify_net("SPI_MOSI", "Default", &[]),
            NetCategory::Signal
        );
        assert_eq!(
            extractor.classify_net("unconnected-(U1-Pad4)", "Default", &[]),
            NetCategory::NoConnect
        );
        assert_eq!(
            extractor.classify_net("NC", "Default", &[]),
            NetCategory::NoConnect
        );
    }

    #[test]
    fn test_net_classification_by_netclass() {
        let extractor = SemanticsExtractor::new();
        assert_eq!(
            extractor.classify_net("custom_name", "GND", &[]),
            NetCategory::Ground
        );
        assert_eq!(
            extractor.classify_net("custom_name", "POWER", &[]),
            NetCategory::Power
        );
    }

    #[test]
    fn test_net_classification_by_pin_type() {
        let extractor = SemanticsExtractor::new();
        assert_eq!(
            extractor.classify_net("custom_net", "Default", &[ElectricalPinType::PowerIn]),
            NetCategory::Power
        );
    }

    #[test]
    fn test_differential_pair_detection() {
        let extractor = SemanticsExtractor::new();
        let mut nets = vec![
            NetRecord {
                id: 1,
                name: "USB_D+".to_string(),
                net_class: "Default".to_string(),
                category: NetCategory::Signal,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout: 2,
                member_pins: vec![],
            },
            NetRecord {
                id: 2,
                name: "USB_D-".to_string(),
                net_class: "Default".to_string(),
                category: NetCategory::Signal,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout: 2,
                member_pins: vec![],
            },
            NetRecord {
                id: 3,
                name: "SPI_SCK".to_string(),
                net_class: "Default".to_string(),
                category: NetCategory::Signal,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout: 3,
                member_pins: vec![],
            },
        ];

        extractor.identify_differential_pairs(&mut nets);

        assert!(nets[0].is_differential_pair);
        assert_eq!(
            nets[0].differential_pair_partner,
            Some("USB_D-".to_string())
        );

        assert!(nets[1].is_differential_pair);
        assert_eq!(
            nets[1].differential_pair_partner,
            Some("USB_D+".to_string())
        );

        assert!(!nets[2].is_differential_pair);
    }

    #[test]
    fn test_differential_pair_underscore_notation() {
        let extractor = SemanticsExtractor::new();
        let mut nets = vec![
            NetRecord {
                id: 1,
                name: "CLK_P".to_string(),
                net_class: "Default".to_string(),
                category: NetCategory::Signal,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout: 2,
                member_pins: vec![],
            },
            NetRecord {
                id: 2,
                name: "CLK_N".to_string(),
                net_class: "Default".to_string(),
                category: NetCategory::Signal,
                is_differential_pair: false,
                differential_pair_partner: None,
                fanout: 2,
                member_pins: vec![],
            },
        ];

        extractor.identify_differential_pairs(&mut nets);

        assert!(nets[0].is_differential_pair);
        assert_eq!(
            nets[0].differential_pair_partner,
            Some("CLK_N".to_string())
        );
        assert!(nets[1].is_differential_pair);
        assert_eq!(
            nets[1].differential_pair_partner,
            Some("CLK_P".to_string())
        );
    }
}
