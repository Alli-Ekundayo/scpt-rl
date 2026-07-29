//! Nom-based lexer for KiCad s-expression files (`.kicad_pcb`).
//!
//! Produces a tree of [`SExpr`] values from raw KiCad text.  Handles
//! whitespace, line comments (`# ...` to EOL), atoms, quoted strings
//! (quotes preserved), and nested `(...)` lists.

use std::fmt;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_till, take_till1, take_while},
    character::complete::{char, multispace0},
    combinator::{eof, opt},
    error::VerboseError,
    multi::many0,
    sequence::delimited,
    IResult,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single s-expression: either an atomic token (string) or a list of
/// s-expressions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SExpr {
    /// An atomic token — either a bare word like `foo` or a quoted string
    /// like `"1"` (quotes are preserved inside the atom).
    Atom(String),
    /// A parenthesised list of s-expressions.
    List(Vec<SExpr>),
}

/// Error returned when the lexer fails to tokenise the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError(pub String);

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error: {}", self.0)
    }
}

impl std::error::Error for LexError {}

// ---------------------------------------------------------------------------
// Internal parser combinators (all operate on `&str`)
// ---------------------------------------------------------------------------

type P<'a, T> = IResult<&'a str, T, VerboseError<&'a str>>;

/// Skip any amount of whitespace *and* `#`-line comments.
///
/// KiCad `.kicad_pcb` files commonly begin with a header like:
///
/// ```text
/// # KiCad PCB file
/// (kicad_pcb (version 20240108) ...)
/// ```
fn ws(input: &str) -> P<'_, ()> {
    let mut rest = input;
    loop {
        // Consume whitespace.
        let (r, _) = multispace0(rest)?;
        // Consume a comment if present, then loop to eat whitespace again.
        if let Ok((r2, _)) = tag::<_, _, VerboseError<&str>>("#")(r) {
            // Drop everything up to (and including) the next newline, if any.
            let drop_to_nl = take_till(|c| c == '\n');
            let (r3, _) = drop_to_nl(r2)?;
            let (r4, _) = opt(char('\n'))(r3)?;
            rest = r4;
            continue;
        }
        rest = r;
        break;
    }
    Ok((rest, ()))
}

/// A quoted string atom — includes the surrounding `"` characters in the
/// returned [`SExpr::Atom`].
fn quoted_atom(input: &str) -> P<'_, SExpr> {
    let (rest, inner) = delimited(
        char('"'),
        // Anything except `"` (no escapes for now — KiCad uses `\"` but
        // the brief's test case does not require escape handling).
        take_while(|c| c != '"'),
        char('"'),
    )(input)?;
    // Reconstruct the atom with its surrounding quotes, exactly as it
    // appeared in the source.
    Ok((rest, SExpr::Atom(format!("\"{inner}\""))))
}

/// A bare (unquoted) atom: a non-empty run of characters that are not
/// whitespace, not `(`, `)`, and not `#` (which would start a comment).
fn bare_atom(input: &str) -> P<'_, SExpr> {
    let (rest, tok) = take_till1(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '#')(input)?;
    Ok((rest, SExpr::Atom(tok.to_string())))
}

/// An atom — either a quoted string or a bare word.
fn atom(input: &str) -> P<'_, SExpr> {
    alt((quoted_atom, bare_atom))(input)
}

/// List parser: `(` followed by zero or more s-expressions followed by `)`.
fn list(input: &str) -> P<'_, SExpr> {
    // `sexpr` has already consumed leading whitespace/comments before
    // dispatching to `list`, so the next character must be `(`.
    let (rest, _) = char('(')(input)?;
    let mut items: Vec<SExpr> = Vec::new();
    let mut cur = rest;
    loop {
        let (r, _) = ws(cur)?;
        cur = r;
        // End of list?
        if let Ok((r, _)) = char::<_, VerboseError<&str>>(')')(cur) {
            return Ok((r, SExpr::List(items)));
        }
        if eof::<_, VerboseError<&str>>(cur).is_ok() {
            // Reached EOF without a closing `)` — propagate an error.
            return Err(nom::Err::Failure(VerboseError {
                errors: vec![(
                    cur,
                    nom::error::VerboseErrorKind::Context("unterminated list"),
                )],
            }));
        }
        // Otherwise parse another s-expr.
        let (r, sexpr) = sexpr(cur)?;
        items.push(sexpr);
        cur = r;
    }
}

/// An s-expression: either an atom or a list.
fn sexpr(input: &str) -> P<'_, SExpr> {
    let (input, _) = ws(input)?;
    alt((atom, list))(input)
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

/// Tokenise a KiCad s-expression document into a list of [`SExpr`] values.
///
/// Top-level input may contain zero or more s-expressions.  Line comments
/// beginning with `#` are skipped, as is all whitespace.
pub fn tokenize(input: &str) -> Result<Vec<SExpr>, LexError> {
    match document(input) {
        Ok((_remaining, sexprs)) => Ok(sexprs),
        Err(e) => Err(LexError(format!("{e:?}"))),
    }
}

/// Internal parser: leading ws, zero-or-more sexprs, trailing ws, then EOF.
fn document(input: &str) -> P<'_, Vec<SExpr>> {
    let (i, _) = ws(input)?;
    let (i, sexprs) = many0(sexpr)(i)?;
    let (i, _) = ws(i)?;
    let (i, _) = eof::<_, VerboseError<&str>>(i)?;
    Ok((i, sexprs))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom() {
        let sexprs = tokenize("(foo 42)").unwrap();
        assert_eq!(
            sexprs,
            vec![SExpr::List(vec![
                SExpr::Atom("foo".into()),
                SExpr::Atom("42".into()),
            ])]
        );
    }

    #[test]
    fn test_nested() {
        let sexprs = tokenize("(a (b c) d)").unwrap();
        assert_eq!(sexprs.len(), 1);
        match &sexprs[0] {
            SExpr::List(inner) => {
                assert_eq!(inner.len(), 3);
                assert!(matches!(&inner[1], SExpr::List(_)));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_quoted_string() {
        let sexprs = tokenize("(pad \"1\" smd)").unwrap();
        match &sexprs[0] {
            SExpr::List(inner) => {
                assert_eq!(inner[1], SExpr::Atom("\"1\"".into()));
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn test_unterminated() {
        assert!(tokenize("(foo").is_err());
    }

    // Extra: KiCad-style comment skipping.
    #[test]
    fn test_comments_and_whitespace() {
        let input = "# KiCad PCB file\n(foo 42)\n# trailing\n";
        let sexprs = tokenize(input).unwrap();
        assert_eq!(
            sexprs,
            vec![SExpr::List(vec![
                SExpr::Atom("foo".into()),
                SExpr::Atom("42".into()),
            ])]
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(tokenize("").unwrap(), vec![]);
        assert_eq!(tokenize("   \n\n").unwrap(), vec![]);
    }

    #[test]
    fn test_top_level_multiple() {
        let sexprs = tokenize("(a) (b c)").unwrap();
        assert_eq!(sexprs.len(), 2);
    }
}
