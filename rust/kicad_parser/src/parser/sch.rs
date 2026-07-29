use crate::error::{Error, Result};
use crate::models::ElectricalPinType;
use crate::parser::sexpr::{SexprNode, SexprParser};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Default, Serialize)]
pub struct RawSchematicData {
    /// Maps symbol lib_id (e.g. "Device:R") to a map of (pin_number -> PinInfo)
    pub lib_symbols: HashMap<String, HashMap<String, PinInfo>>,
    /// Maps ref_des (e.g. "R1") to symbol instance lib_id & properties
    pub symbol_instances: Vec<SymbolInstance>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SymbolInstance {
    pub ref_des: String,
    pub lib_id: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PinInfo {
    pub number: String,
    pub name: String,
    pub electrical_type: ElectricalPinType,
}

pub fn parse_schematic(content: &str) -> Result<RawSchematicData> {
    let mut parser = SexprParser::new(content);
    let root = parser.parse_root()?;

    if root.name() != Some("kicad_sch") {
        return Err(Error::InvalidSexpr(
            "Root node is not 'kicad_sch'".to_string(),
        ));
    }

    let mut sch_data = RawSchematicData::default();

    // 1. Extract embedded library symbols
    if let Some(lib_symbols_node) = root.get_child_by_name("lib_symbols") {
        for sym_node in lib_symbols_node.children() {
            if sym_node.name() == Some("symbol") {
                if let Some(lib_id) = sym_node.get_string_arg(0) {
                    let pin_map = extract_pins_from_lib_symbol(sym_node);
                    sch_data.lib_symbols.insert(lib_id.to_string(), pin_map);
                }
            }
        }
    }

    // 2. Extract symbol instances
    for child in root.children() {
        if child.name() == Some("symbol") {
            if let Some(inst) = parse_symbol_instance(child) {
                sch_data.symbol_instances.push(inst);
            }
        }
    }

    Ok(sch_data)
}

fn extract_pins_from_lib_symbol(node: &SexprNode) -> HashMap<String, PinInfo> {
    let mut pin_map = HashMap::new();

    // Check direct children or nested sub-units (e.g. symbol "Device:R_1_1")
    fn recurse_symbol(node: &SexprNode, pin_map: &mut HashMap<String, PinInfo>) {
        for child in node.children() {
            if child.name() == Some("symbol") {
                recurse_symbol(child, pin_map);
            } else if child.name() == Some("pin") {
                if let Some(pin_info) = parse_lib_pin(child) {
                    pin_map.insert(pin_info.number.clone(), pin_info);
                }
            }
        }
    }

    recurse_symbol(node, &mut pin_map);
    pin_map
}

fn parse_lib_pin(node: &SexprNode) -> Option<PinInfo> {
    let type_str = node.get_string_arg(0)?;
    let electrical_type = parse_electrical_type(type_str);

    let mut name = String::new();
    let mut number = String::new();

    if let Some(name_node) = node.get_child_by_name("name") {
        if let Some(n) = name_node.get_string_arg(0) {
            name = n.to_string();
        }
    }

    if let Some(num_node) = node.get_child_by_name("number") {
        if let Some(n) = num_node.get_string_arg(0) {
            number = n.to_string();
        }
    }

    if number.is_empty() {
        return None;
    }

    Some(PinInfo {
        number,
        name,
        electrical_type,
    })
}

fn parse_symbol_instance(node: &SexprNode) -> Option<SymbolInstance> {
    let mut ref_des = String::new();
    let mut lib_id = String::new();
    let mut value = String::new();

    if let Some(lib_id_node) = node.get_child_by_name("lib_id") {
        if let Some(id) = lib_id_node.get_string_arg(0) {
            lib_id = id.to_string();
        }
    }

    for child in node.children() {
        if child.name() == Some("property") {
            if let (Some(prop_name), Some(prop_val)) = (child.get_string_arg(0), child.get_string_arg(1)) {
                if prop_name.eq_ignore_ascii_case("Reference") {
                    ref_des = prop_val.to_string();
                } else if prop_name.eq_ignore_ascii_case("Value") {
                    value = prop_val.to_string();
                }
            }
        }
    }

    if ref_des.is_empty() || lib_id.is_empty() {
        return None;
    }

    Some(SymbolInstance {
        ref_des,
        lib_id,
        value,
    })
}

pub fn parse_electrical_type(s: &str) -> ElectricalPinType {
    match s.to_ascii_lowercase().as_str() {
        "input" => ElectricalPinType::Input,
        "output" => ElectricalPinType::Output,
        "bidirectional" | "bidi" => ElectricalPinType::Bidirectional,
        "tri_state" | "tristate" => ElectricalPinType::TriState,
        "passive" => ElectricalPinType::Passive,
        "free" => ElectricalPinType::Free,
        "unspecified" => ElectricalPinType::Unspecified,
        "power_in" | "powerin" => ElectricalPinType::PowerIn,
        "power_out" | "powerout" => ElectricalPinType::PowerOut,
        "open_collector" | "opencollector" => ElectricalPinType::OpenCollector,
        "open_emitter" | "openemitter" => ElectricalPinType::OpenEmitter,
        "no_connect" | "noconnect" => ElectricalPinType::NoConnect,
        _ => ElectricalPinType::Unspecified,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schematic_minimal() {
        let input = r#"
        (kicad_sch (version 20240101) (generator eeschema)
          (lib_symbols
            (symbol "Device:R"
              (pin passive line (at 0 2.54 270) (number "1") (name "~"))
              (pin passive line (at 0 -2.54 90) (number "2") (name "~"))
            )
          )
          (symbol (lib_id "Device:R") (at 100 50)
            (property "Reference" "R1")
            (property "Value" "10k")
          )
        )
        "#;
        let data = parse_schematic(input).unwrap();
        assert_eq!(data.lib_symbols.len(), 1);
        assert!(data.lib_symbols.contains_key("Device:R"));

        let pin_map = &data.lib_symbols["Device:R"];
        assert_eq!(pin_map.len(), 2);
        assert_eq!(pin_map["1"].electrical_type, ElectricalPinType::Passive);
        assert_eq!(pin_map["2"].electrical_type, ElectricalPinType::Passive);

        assert_eq!(data.symbol_instances.len(), 1);
        assert_eq!(data.symbol_instances[0].ref_des, "R1");
        assert_eq!(data.symbol_instances[0].lib_id, "Device:R");
        assert_eq!(data.symbol_instances[0].value, "10k");
    }

    #[test]
    fn test_parse_schematic_nested_symbols() {
        // KiCad uses nested symbols for multi-unit parts
        let input = r#"
        (kicad_sch (version 20240101) (generator eeschema)
          (lib_symbols
            (symbol "74xx:74LS00"
              (symbol "74xx:74LS00_1_1"
                (pin input line (at -12.7 2.54 0) (number "1") (name "A"))
                (pin input line (at -12.7 0 0) (number "2") (name "B"))
                (pin output line (at 12.7 1.27 180) (number "3") (name "Y"))
              )
            )
          )
        )
        "#;
        let data = parse_schematic(input).unwrap();
        let pin_map = &data.lib_symbols["74xx:74LS00"];
        assert_eq!(pin_map.len(), 3);
        assert_eq!(pin_map["1"].name, "A");
        assert_eq!(pin_map["1"].electrical_type, ElectricalPinType::Input);
        assert_eq!(pin_map["3"].electrical_type, ElectricalPinType::Output);
    }

    #[test]
    fn test_parse_electrical_type() {
        assert_eq!(parse_electrical_type("input"), ElectricalPinType::Input);
        assert_eq!(parse_electrical_type("passive"), ElectricalPinType::Passive);
        assert_eq!(parse_electrical_type("power_in"), ElectricalPinType::PowerIn);
        assert_eq!(parse_electrical_type("PowerIn"), ElectricalPinType::PowerIn);
        assert_eq!(parse_electrical_type("unknown_type"), ElectricalPinType::Unspecified);
    }

    #[test]
    fn test_parse_schematic_invalid_root() {
        let input = "(kicad_pcb (version 20240101))";
        let result = parse_schematic(input);
        assert!(result.is_err());
    }
}
