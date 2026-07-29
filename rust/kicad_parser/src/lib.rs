//! Fast KiCad PCB and schematic parser for geometry, connectivity,
//! and electrical semantics extraction for ML pipelines.
//!
//! # Overview
//!
//! This crate parses KiCad `.kicad_pcb` and `.kicad_sch` files (S-expression format)
//! and produces structured output suitable for machine learning and reinforcement
//! learning pipelines.
//!
//! # Modules
//!
//! - [`parser`] — S-expression tokenizer, PCB and schematic parsers
//! - [`models`] — Output data types (`BoardOutput`, `ComponentRecord`, `NetRecord`, etc.)
//! - [`geometry`] — Coordinate transforms, bounding boxes, board outline extraction, polygon ops
//! - [`semantics`] — Net classification (power/ground/signal) and differential pair detection
//! - [`error`] — Error types
//!
//! # Quick Start
//!
//! ```no_run
//! use kicad_parser::{parse_pcb_to_output, Error};
//!
//! let pcb_content = std::fs::read_to_string("board.kicad_pcb")?;
//! let output = parse_pcb_to_output(&pcb_content, "board.kicad_pcb", None, None)?;
//! println!("Found {} components", output.components.len());
//! println!("Found {} nets", output.nets.len());
//! # Ok::<(), Error>(())
//! ```

#![warn(unused_imports, unused_variables)]

pub mod error;
pub mod geometry;
pub mod models;
pub mod parser;
pub mod semantics;

// Re-export primary types for convenience
pub use error::{Error, Result};
pub use models::{
    BoardGeometry, BoardMetadata, BoardOutput, BoundingBox, ComponentRecord, ElectricalPinType,
    HoleRecord, KeepoutZoneRecord, NetCategory, NetRecord, PinRecord, PinReference, Point2D,
};
pub use parser::{build_board_output, parse_pcb, parse_schematic};
pub use semantics::SemanticsExtractor;

/// Parses a KiCad PCB file and returns a high-level `BoardOutput`.
///
/// Optionally accepts schematic content and filenames for pin enrichment.
///
/// # Arguments
///
/// * `pcb_content` — Contents of a `.kicad_pcb` file
/// * `pcb_filename` — Filename for metadata (not read, just recorded)
/// * `sch_content` — Optional contents of a `.kicad_sch` file for pin name/type enrichment
/// * `sch_filename` — Optional schematic filename for metadata
///
/// # Errors
///
/// Returns `Error::SexprParse` if the PCB content is not valid S-expressions.
/// Returns `Error::InvalidSexpr` if the root node is not `kicad_pcb`.
pub fn parse_pcb_to_output(
    pcb_content: &str,
    pcb_filename: &str,
    sch_content: Option<&str>,
    sch_filename: Option<&str>,
) -> Result<BoardOutput> {
    let raw_pcb = parse_pcb(pcb_content)?;

    let raw_sch = match sch_content {
        Some(content) => Some(parse_schematic(content)?),
        None => None,
    };

    Ok(build_board_output(
        &raw_pcb,
        raw_sch.as_ref(),
        pcb_filename,
        sch_filename,
    ))
}

/// Parses a KiCad schematic file and returns raw schematic data.
///
/// This is a convenience wrapper around [`parser::parse_schematic`].
pub fn parse_sch_to_output(sch_content: &str) -> Result<parser::RawSchematicData> {
    parse_schematic(sch_content)
}
