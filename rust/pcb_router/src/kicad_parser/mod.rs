//! KiCad PCB (.kicad_pcb) file parser using kiutils-rs.
//!
//! Parses the KiCad S-expression format into structures sufficient
//! to drive the grid-based router:
//!   - Board layers (copper layers only)
//!   - Netclasses
//!   - Nets
//!   - Modules / footprints (instances)
//!   - Pads (via / SMD / through-hole)

pub mod types;

use std::collections::HashMap;
use types::*;
use kiutils_rs::{PcbDocument, PcbFile, PcbSegment, PcbVia, WriteMode};
use std::fs;
use std::io::{self, Read};

fn parse_quoted_or_token_after_prefix(line: &str, prefix: &str) -> Option<String> {
    let rest = line.strip_prefix(prefix)?.trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    let token = rest
        .split([' ', '\t', ')'])
        .find(|s| !s.is_empty())?;
    Some(token.to_string())
}

fn parse_f64_after_prefix(line: &str, prefix: &str) -> Option<f64> {
    let rest = line.strip_prefix(prefix)?.trim_start();
    let token = rest
        .split([' ', '\t', ')'])
        .find(|s| !s.is_empty())?;
    token.parse::<f64>().ok()
}

fn parse_legacy_netclasses(content: &str) -> (Vec<Netclass>, HashMap<String, String>) {
    let mut netclasses = Vec::new();
    let mut net_to_netclass = HashMap::new();
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0usize;

    while i < lines.len() {
        let line = lines[i].trim();
        if !line.starts_with("(net_class ") {
            i += 1;
            continue;
        }

        let Some(name) = parse_quoted_or_token_after_prefix(line, "(net_class") else {
            i += 1;
            continue;
        };

        let mut nc = Netclass {
            name: name.clone(),
            ..Netclass::default()
        };

        let mut depth = line.matches('(').count() as i32 - line.matches(')').count() as i32;
        i += 1;

        while i < lines.len() && depth > 0 {
            let cur = lines[i].trim();

            if cur.starts_with("(clearance") {
                if let Some(v) = parse_f64_after_prefix(cur, "(clearance") {
                    nc.clearance = v;
                }
            } else if cur.starts_with("(trace_width") {
                if let Some(v) = parse_f64_after_prefix(cur, "(trace_width") {
                    nc.trace_width = v;
                }
            } else if cur.starts_with("(via_dia") {
                if let Some(v) = parse_f64_after_prefix(cur, "(via_dia") {
                    nc.via_dia = v;
                }
            } else if cur.starts_with("(via_drill") {
                if let Some(v) = parse_f64_after_prefix(cur, "(via_drill") {
                    nc.via_drill = v;
                }
            } else if cur.starts_with("(uvia_dia") {
                if let Some(v) = parse_f64_after_prefix(cur, "(uvia_dia") {
                    nc.uvia_dia = v;
                }
            } else if cur.starts_with("(uvia_drill") {
                if let Some(v) = parse_f64_after_prefix(cur, "(uvia_drill") {
                    nc.uvia_drill = v;
                }
            } else if cur.starts_with("(add_net") {
                if let Some(net_name) = parse_quoted_or_token_after_prefix(cur, "(add_net") {
                    net_to_netclass.insert(net_name, nc.name.clone());
                }
            }

            depth += cur.matches('(').count() as i32 - cur.matches(')').count() as i32;
            i += 1;
        }

        netclasses.push(nc);
    }

    (netclasses, net_to_netclass)
}

/// Central database built from a parsed `.kicad_pcb` file.
#[derive(Debug, Default)]
pub struct KicadPcbDatabase {
    pub layers: Vec<Layer>,
    /// Map from KiCad layer number to layer name.
    pub layer_id_to_name: HashMap<i32, String>,
    pub netclasses: Vec<Netclass>,
    pub nets: Vec<Net>,
    pub instances: Vec<Instance>,
    /// All pads from all instances, including their absolute position.
    pub pads: Vec<Pad>,
    /// Original document for native write support.
    pub doc: Option<PcbDocument>,
    new_segments: Vec<PcbSegment>,
    new_vias: Vec<PcbVia>,
}

impl KicadPcbDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let content = fs::read_to_string(path)?;
        Self::from_string(&content)
    }

    /// Parse a `.kicad_pcb` file from a reader.
    pub fn from_reader<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut content = String::new();
        reader.read_to_string(&mut content)?;
        Self::from_string(&content)
    }

    /// Internal helper to parse from a string with legacy support.
    fn from_string(content: &str) -> io::Result<Self> {
        let (parsed_netclasses, net_to_netclass) = parse_legacy_netclasses(content);

        // KiCad 5 uses '(module ...)', while KiCad 6+ (and kiutils-rs) use '(footprint ...)'
        // We do a simple replacement to bridge the gap.
        let converted = content.replace("(module ", "(footprint ");
        
        // Since kiutils-rs 0.2.0 PcbFile only reads from path, we use a temporary file.
        let temp_path = std::env::temp_dir().join(format!("pcb_router_temp_{}.kicad_pcb", 
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        
        fs::write(&temp_path, converted)?;
        
        let doc_result = PcbFile::read(&temp_path);
        
        // Clean up temp file immediately
        let _ = fs::remove_file(&temp_path);
        
        let doc = doc_result.map_err(|e| io::Error::other(e.to_string()))?;
        
        let mut db = KicadPcbDatabase::default();
        db.load_from_ast(doc.ast());
        if !parsed_netclasses.is_empty() {
            db.netclasses = parsed_netclasses;
        }
        if !net_to_netclass.is_empty() {
            for net in &mut db.nets {
                if let Some(name) = net_to_netclass.get(&net.name) {
                    net.netclass_name = name.clone();
                }
            }
        }
        db.doc = Some(doc);
        Ok(db)
    }

    fn load_from_ast(&mut self, ast: &kiutils_rs::PcbAst) {
        // Map layers
        for (idx, layer) in ast.layers.iter().enumerate() {
            let id = idx as i32;
            let name = layer.name.clone().unwrap_or_else(|| format!("Layer_{}", id));
            let layer_type = match layer.layer_type.as_deref() {
                Some("signal") | Some("power") => LayerType::Copper,
                _ if name.starts_with('F') || name.starts_with('B') || name == "In1.Cu"
                    || name.contains(".Cu") => LayerType::Copper,
                _ => LayerType::NonCopper,
            };
            self.layer_id_to_name.insert(id, name.clone());
            self.layers.push(Layer { id, name, layer_type });
        }

        // Map nets
        for net in &ast.nets {
            self.nets.push(Net {
                id: net.code.unwrap_or(0),
                name: net.name.clone().unwrap_or_default(),
                netclass_name: String::new(),
                pins: Vec::new(),
            });
        }

        // Map footprints and pads
        for fp in &ast.footprints {
            let ref_name = fp.reference.clone().unwrap_or_default();
            let pos = fp.at.unwrap_or([0.0, 0.0]);
            let rot = fp.rotation.unwrap_or(0.0);
            let layer = fp.layer.clone().unwrap_or_default();

            self.instances.push(Instance {
                reference: ref_name.clone(),
                position: (pos[0], pos[1]),
                rotation: rot,
                layer,
            });

            for pad in &fp.pads {
                let pad_pos = pad.at.unwrap_or([0.0, 0.0]);
                // Absolute position including footprint rotation.
                let rot_rad = rot.to_radians();
                let cos_r = rot_rad.cos();
                let sin_r = rot_rad.sin();
                let abs_pad_pos = (
                    pos[0] + (pad_pos[0] * cos_r - pad_pos[1] * sin_r),
                    pos[1] + (pad_pos[0] * sin_r + pad_pos[1] * cos_r),
                );

                self.pads.push(Pad {
                    reference: ref_name.clone(),
                    number: pad.number.clone().unwrap_or_default(),
                    pad_type: pad.pad_type.clone().unwrap_or_default(),
                    shape: pad.shape.clone().unwrap_or_default(),
                    position: abs_pad_pos,
                    size: {
                        let s = pad.size.unwrap_or([0.0, 0.0]);
                        (s[0], s[1])
                    },
                    net_id: pad.net.as_ref().and_then(|n| n.code).unwrap_or(-1),
                    layers: pad.layers.clone(),
                    instance_position: (pos[0], pos[1]),
                });
            }
        }
    }

    pub fn add_routing_results(&mut self, segments: Vec<PcbSegment>, vias: Vec<PcbVia>) {
        // We store them locally instead of modifying the doc directly to avoid serialization errors
        self.new_segments = segments;
        self.new_vias = vias;
    }

    pub fn write_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> io::Result<()> {
        if let Some(doc) = &self.doc {
            // 1. Write the base document to a temporary file
            let temp_path = std::env::temp_dir().join(format!("pcb_router_base_{}.kicad_pcb", 
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            
            doc.write_mode(&temp_path, WriteMode::Canonical)
                .map_err(|e| io::Error::other(e.to_string()))?;
            
            // 2. Read it back as a string
            let mut content = fs::read_to_string(&temp_path)?;
            let _ = fs::remove_file(&temp_path);
            
            // 3. Find the last ')' and insert our results before it
            if let Some(last_pos) = content.rfind(')') {
                let mut results = String::new();
                for s in &self.new_segments {
                    results.push_str(&format!("  (segment (start {} {}) (end {} {}) (width {}) (layer \"{}\") (net {}))\n",
                        s.start.unwrap_or([0.0, 0.0])[0], s.start.unwrap_or([0.0, 0.0])[1],
                        s.end.unwrap_or([0.0, 0.0])[0], s.end.unwrap_or([0.0, 0.0])[1],
                        s.width.unwrap_or(0.2),
                        s.layer.as_ref().unwrap_or(&"F.Cu".to_string()),
                        s.net.unwrap_or(0)));
                }
                for v in &self.new_vias {
                    results.push_str(&format!("  (via (at {} {}) (size {}) (drill {}) (layers \"{}\" \"{}\") (net {}))\n",
                        v.at.unwrap_or([0.0, 0.0])[0], v.at.unwrap_or([0.0, 0.0])[1],
                        v.size.unwrap_or(0.8),
                        v.drill.unwrap_or(0.4),
                        v.layers.first().cloned().unwrap_or_else(|| "F.Cu".to_string()),
                        v.layers.get(1).cloned().unwrap_or_else(|| "B.Cu".to_string()),
                        v.net.unwrap_or(0)));
                }
                
                content.insert_str(last_pos, &results);
            }
            
            // 4. Convert back to legacy 'module' tokens if needed for KiCad 5 compatibility
            let final_content = content.replace("(footprint", "(module");
            
            // 5. Write final content to destination
            fs::write(path, final_content)?;
        } else {
            return Err(io::Error::other("No original document loaded"));
        }
        Ok(())
    }

    pub fn copper_layers(&self) -> Vec<&Layer> {
        self.layers.iter().filter(|l| l.layer_type == LayerType::Copper).collect()
    }

    pub fn num_copper_layers(&self) -> usize {
        self.copper_layers().len()
    }

    pub fn print_design_statistics(&self) {
        println!("Layers: {}", self.layers.len());
        println!("Copper layers: {}", self.num_copper_layers());
        println!("Nets: {}", self.nets.len());
        println!("Netclasses: {}", self.netclasses.len());
        println!("Instances: {}", self.instances.len());
        println!("Pads: {}", self.pads.len());
    }
}

#[cfg(test)]
mod tests {
    use super::parse_legacy_netclasses;

    #[test]
    fn parse_kicad5_netclass_block() {
        let content = r#"
(net_class "Default" "Default net class."
  (clearance 0.2)
  (trace_width 0.25)
  (via_dia 0.8)
  (via_drill 0.4)
  (uvia_dia 0.3)
  (uvia_drill 0.1)
  (add_net "GND")
)
"#;
        let (netclasses, net_map) = parse_legacy_netclasses(content);
        assert_eq!(netclasses.len(), 1);
        assert_eq!(netclasses[0].name, "Default");
        assert!((netclasses[0].trace_width - 0.25).abs() < 1e-9);
        assert_eq!(net_map.get("GND").map(String::as_str), Some("Default"));
    }
}
