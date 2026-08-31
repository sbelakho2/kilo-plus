use crate::lexer::{Lexer, TokenKind};

pub struct Parser {
    lexer: Lexer,
}

impl Parser {
    pub fn parse(&mut self) -> Result<Vec<Expr>, String> {
        let mut exprs = Vec::new();
        while let Some(token) = self.lexer.next_token() {
            match token.kind {
                TokenKind::Ident => exprs.push(Expr::Ident(token.text)),
                TokenKind::Number => exprs.push(Expr::Number(0)),
            }
        }
        Ok(exprs)
    }
}

pub enum Expr {
    Ident(String),
    Number(i64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_identifiers() {
        let mut p = Parser { lexer: Lexer::new("ab") };
        assert_eq!(p.parse().unwrap().len(), 2);
    }
}
