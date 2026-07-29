use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum SexprAtom {
    Symbol(String),
    String(String),
    Number(f64),
}

impl SexprAtom {
    pub fn as_str(&self) -> &str {
        match self {
            SexprAtom::Symbol(s) => s.as_str(),
            SexprAtom::String(s) => s.as_str(),
            SexprAtom::Number(_) => "",
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SexprAtom::Number(n) => Some(*n),
            SexprAtom::Symbol(s) => s.parse::<f64>().ok(),
            SexprAtom::String(s) => s.parse::<f64>().ok(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SexprNode {
    Atom(SexprAtom),
    List(Vec<SexprNode>),
}

impl SexprNode {
    pub fn name(&self) -> Option<&str> {
        if let SexprNode::List(list) = self {
            if let Some(SexprNode::Atom(atom)) = list.first() {
                return Some(atom.as_str());
            }
        }
        None
    }

    pub fn children(&self) -> &[SexprNode] {
        if let SexprNode::List(list) = self {
            if list.len() > 1 {
                return &list[1..];
            }
        }
        &[]
    }

    pub fn get_child_by_name(&self, name: &str) -> Option<&SexprNode> {
        if let SexprNode::List(list) = self {
            for child in list.iter().skip(1) {
                if child.name() == Some(name) {
                    return Some(child);
                }
            }
        }
        None
    }

    pub fn get_children_by_name(&self, name: &str) -> Vec<&SexprNode> {
        let mut result = Vec::new();
        if let SexprNode::List(list) = self {
            for child in list.iter().skip(1) {
                if child.name() == Some(name) {
                    result.push(child);
                }
            }
        }
        result
    }

    pub fn get_string_arg(&self, index: usize) -> Option<&str> {
        if let SexprNode::List(list) = self {
            if let Some(SexprNode::Atom(atom)) = list.get(index + 1) {
                return Some(atom.as_str());
            }
        }
        None
    }

    pub fn get_float_arg(&self, index: usize) -> Option<f64> {
        if let SexprNode::List(list) = self {
            if let Some(SexprNode::Atom(atom)) = list.get(index + 1) {
                return atom.as_f64();
            }
        }
        None
    }
}

pub struct SexprParser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> SexprParser<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    pub fn parse_root(&mut self) -> Result<SexprNode> {
        self.skip_whitespace_and_comments();
        let node = self.parse_node()?;
        Ok(node)
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn read_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn skip_whitespace_and_comments(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.read_char();
            } else if ch == ';' {
                // Comment line until newline
                while let Some(c) = self.peek_char() {
                    self.read_char();
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn parse_node(&mut self) -> Result<SexprNode> {
        self.skip_whitespace_and_comments();
        match self.peek_char() {
            Some('(') => {
                self.read_char(); // consume '('
                let mut list = Vec::new();
                loop {
                    self.skip_whitespace_and_comments();
                    match self.peek_char() {
                        Some(')') => {
                            self.read_char(); // consume ')'
                            break;
                        }
                        None => {
                            return Err(Error::SexprParse {
                                line: self.line,
                                col: self.col,
                                message: "Unmatched opening parenthesis".to_string(),
                            });
                        }
                        _ => {
                            let child = self.parse_node()?;
                            list.push(child);
                        }
                    }
                }
                Ok(SexprNode::List(list))
            }
            Some('"') => {
                let s = self.parse_quoted_string()?;
                Ok(SexprNode::Atom(SexprAtom::String(s)))
            }
            Some(_) => {
                let atom = self.parse_unquoted_atom()?;
                Ok(node_from_atom_str(&atom))
            }
            None => Err(Error::SexprParse {
                line: self.line,
                col: self.col,
                message: "Unexpected end of input".to_string(),
            }),
        }
    }

    fn parse_quoted_string(&mut self) -> Result<String> {
        self.read_char(); // consume open quote
        let mut s = String::new();
        while let Some(ch) = self.read_char() {
            if ch == '"' {
                return Ok(s);
            } else if ch == '\\' {
                if let Some(next) = self.read_char() {
                    match next {
                        'n' => s.push('\n'),
                        'r' => s.push('\r'),
                        't' => s.push('\t'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        _ => {
                            s.push('\\');
                            s.push(next);
                        }
                    }
                }
            } else {
                s.push(ch);
            }
        }
        Err(Error::SexprParse {
            line: self.line,
            col: self.col,
            message: "Unterminated string literal".to_string(),
        })
    }

    fn parse_unquoted_atom(&mut self) -> Result<String> {
        let mut s = String::new();
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() || ch == '(' || ch == ')' || ch == '"' || ch == ';' {
                break;
            }
            s.push(ch);
            self.read_char();
        }
        if s.is_empty() {
            return Err(Error::SexprParse {
                line: self.line,
                col: self.col,
                message: "Empty atom token".to_string(),
            });
        }
        Ok(s)
    }
}

fn node_from_atom_str(s: &str) -> SexprNode {
    if let Ok(val) = s.parse::<f64>() {
        SexprNode::Atom(SexprAtom::Number(val))
    } else {
        SexprNode::Atom(SexprAtom::Symbol(s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sexpr_basic_parsing() {
        let input = r#"
        (kicad_pcb (version 20240101) (generator pcbnew)
          (net 0 "")
          (net 1 "VCC")
          (footprint "Resistor_SMD:R_0805"
            (at 10.5 20.0 90)
            (pad "1" smd rect (at -1.0 0) (size 1.0 1.2) (net 1 "VCC"))
          )
        )
        "#;
        let mut parser = SexprParser::new(input);
        let root = parser.parse_root().unwrap();
        assert_eq!(root.name(), Some("kicad_pcb"));

        let version_node = root.get_child_by_name("version").unwrap();
        assert_eq!(version_node.get_float_arg(0), Some(20240101.0));

        let footprint_nodes = root.get_children_by_name("footprint");
        assert_eq!(footprint_nodes.len(), 1);
        let fp = footprint_nodes[0];
        assert_eq!(fp.get_string_arg(0), Some("Resistor_SMD:R_0805"));
    }
}
