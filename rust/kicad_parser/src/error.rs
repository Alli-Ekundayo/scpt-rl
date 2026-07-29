use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("S-expression parse error at line {line}, col {col}: {message}")]
    SexprParse {
        line: usize,
        col: usize,
        message: String,
    },

    #[error("Unexpected S-expression structure: {0}")]
    InvalidSexpr(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("File error: {0}")]
    FileError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
