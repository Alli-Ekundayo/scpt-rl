//! KiCad `.kicad_pcb` parser pipeline.
//!
//! - [`lexer`] turns raw s-expression text into a [`lexer::SExpr`] tree.
//! - [`types`] (populated by Task 4) holds the KiCad-specific AST types
//!   the parser produces from the s-expression tree.

pub mod lexer;
pub mod types;
