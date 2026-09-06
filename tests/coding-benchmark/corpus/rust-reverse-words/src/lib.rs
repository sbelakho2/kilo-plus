//! Word-order reversal: `reverse_words` must reverse the ORDER of the words
//! of the input and leave the letters of every word untouched.

/// Return the words of `s` (split on any whitespace run) in REVERSE order,
/// joined with single spaces. Whitespace-only and empty inputs produce an
/// empty string.
pub fn reverse_words(s: &str) -> String {
    s.split_whitespace()
        .map(|w| w.chars().rev().collect::<String>())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::reverse_words;

    #[test]
    fn reverses_the_order_of_words() {
        assert_eq!(reverse_words("one two three"), "three two one");
    }

    #[test]
    fn letters_inside_each_word_keep_their_order() {
        assert_eq!(reverse_words("hello world"), "world hello");
    }

    #[test]
    fn whitespace_runs_collapse_to_one_separator() {
        assert_eq!(reverse_words("a   b\t c"), "c b a");
    }

    #[test]
    fn empty_and_whitespace_only_inputs_return_empty() {
        assert_eq!(reverse_words(""), "");
        assert_eq!(reverse_words("   \t "), "");
    }
}
