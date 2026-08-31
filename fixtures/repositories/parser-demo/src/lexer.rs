use std::collections::HashMap;

pub struct Lexer {
    source: String,
    position: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self { source: source.to_string(), position: 0 }
    }

    pub fn next_token(&mut self) -> Option<Token> {
        let rest = &self.source[self.position..];
        if rest.is_empty() {
            return None;
        }
        self.position += 1;
        Some(Token { kind: TokenKind::Ident, text: rest[..1].to_string() })
    }
}

pub enum TokenKind {
    Ident,
    Number,
}

pub struct Token {
    pub kind: TokenKind,
    pub text: String,
}

#[test]
fn lexer_advances() {
    let mut l = Lexer::new("ab");
    assert!(l.next_token().is_some());
    assert!(l.next_token().is_some());
    assert!(l.next_token().is_none());
}
