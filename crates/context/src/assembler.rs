//! Context assembly with three memory classes ordered for prefix caching:
//!
//! ```text
//! STATIC PREFIX   system instructions, tools, project rules, agent behavior
//! SEMI-STABLE     task state, repository map
//! VOLATILE        recent messages, retrieved evidence, current errors
//! ```
//!
//! Never put timestamps or frequently changing values near the front. The
//! budget is enforced here, before anything is sent.

use kilop_core::error::{Error, ErrorKind};

use crate::budget::ContextBudget;
use crate::estimator::Estimator;
use crate::ledger::TaskLedger;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryClass {
    StaticPrefix,
    SemiStable,
    Volatile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextSection {
    pub class: MemoryClass,
    pub text: String,
    pub tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssembledContext {
    pub sections: Vec<ContextSection>,
    pub total_tokens: usize,
    /// StaticPrefix + SemiStable tokens: the provider-cacheable prefix.
    pub cacheable_tokens: usize,
    /// Byte offset where the volatile section begins in the render.
    pub volatile_start: usize,
}

impl AssembledContext {
    pub fn render(&self) -> String {
        self.sections.iter().map(|s| s.text.as_str()).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecentTurn {
    pub role: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Evidence {
    pub path: String,
    pub snippet: String,
    pub score: f64,
}

const ARTIFACT_REF: &str = "[evidence archived: artifact://<hash> — see artifact store]";

pub struct ContextAssembler;

impl ContextAssembler {
    /// Assemble the full context under the budget. The invariant:
    /// `total_tokens <= budget.total() - budget.output_reserve - budget.safety`
    /// Deterministic trimming order: oldest recent turns, then evidence,
    /// then oversized evidence is replaced by an artifact reference.
    pub fn assemble(
        static_prefix: &str,
        system_extra: &str,
        tool_schemas: &str,
        project_rules: &str,
        ledger: &TaskLedger,
        repo_map: &str,
        recent_turns: &[RecentTurn],
        retrieved_evidence: &[Evidence],
        errors: &str,
        budget: &ContextBudget,
    ) -> Result<AssembledContext, Error> {
        let est = Estimator;
        let context_max = budget.context_max();
        if context_max == 0 {
            return Err(Error::new(
                ErrorKind::Oversized,
                "context budget leaves no room for content",
            ));
        }

        // 1. STATIC PREFIX — byte-stable, cacheable.
        let static_text = join_sections(static_prefix, system_extra, tool_schemas, project_rules);
        let static_tokens = est.estimate_tokens(&static_text);
        let static_section = ContextSection {
            class: MemoryClass::StaticPrefix,
            text: static_text.clone(),
            tokens: static_tokens,
        };

        // 2. SEMI-STABLE — task state + repository map.
        let mut semi_text = String::new();
        semi_text.push_str("## Task state\n");
        semi_text.push_str(&ledger.compact_render());
        if !repo_map.is_empty() {
            semi_text.push_str("## Repository map\n");
            semi_text.push_str(&truncate(repo_map, 2000));
        }
        let semi_tokens = est.estimate_tokens(&semi_text);
        let semi_section = ContextSection {
            class: MemoryClass::SemiStable,
            text: semi_text.clone(),
            tokens: semi_tokens,
        };

        // Budget the volatile classes.
        let volatile_budget = context_max.saturating_sub(static_tokens).saturating_sub(semi_tokens);
        let recent_cap = budget
            .recent
            .min(volatile_budget.saturating_sub(1));
        let evidence_cap = budget.retrieved.min(volatile_budget.saturating_sub(recent_cap).saturating_sub(1));
        let error_cap = volatile_budget
            .saturating_sub(recent_cap)
            .saturating_sub(evidence_cap)
            .saturating_sub(1);

        // 3. VOLATILE — recent turns, newest kept first; trimmed oldest-first.
        let mut recent_text = String::new();
        let mut recent_tokens = 0usize;
        for turn in recent_turns.iter().rev() {
            let line = format!("\n## {}\n{}\n", turn.role, truncate(&turn.text, 4000));
            let t = est.estimate_tokens(&line);
            if recent_tokens + t > recent_cap {
                break;
            }
            recent_text.push_str(&line);
            recent_tokens += t;
        }

        // Evidence: keep high-score first, replace the rest with artifact refs.
        let mut evidence_text = String::new();
        let mut evidence_tokens = 0usize;
        let mut scored: Vec<&Evidence> = retrieved_evidence.iter().collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        for ev in scored {
            let entry = format!("\n### {}\n{}\n", ev.path, truncate(&ev.snippet, 1500));
            let t = est.estimate_tokens(&entry);
            if evidence_tokens + t > evidence_cap {
                let ref_entry = format!("\n### {}\n{}\n", ev.path, ARTIFACT_REF);
                let rt = est.estimate_tokens(&ref_entry);
                if evidence_tokens + rt <= evidence_cap {
                    evidence_text.push_str(&ref_entry);
                    evidence_tokens += rt;
                }
                continue;
            }
            evidence_text.push_str(&entry);
            evidence_tokens += t;
        }

        // Errors (current diagnostics) — bounded last.
        let error_text = if error_cap > 0 {
            truncate(errors, error_cap.saturating_mul(4))
        } else {
            String::new()
        };
        let error_tokens = est.estimate_tokens(&error_text);

        let mut volatile_text = String::new();
        if !recent_text.is_empty() {
            volatile_text.push_str("## Recent conversation\n");
            volatile_text.push_str(&recent_text);
        }
        if !evidence_text.is_empty() {
            volatile_text.push_str("\n## Retrieved evidence\n");
            volatile_text.push_str(&evidence_text);
        }
        if !error_text.is_empty() {
            volatile_text.push_str("\n## Current errors\n");
            volatile_text.push_str(&error_text);
        }
        let volatile_tokens = recent_tokens
            .saturating_add(evidence_tokens)
            .saturating_add(error_tokens)
            .saturating_add(4);

        let sections = vec![
            static_section.clone(),
            semi_section.clone(),
            ContextSection {
                class: MemoryClass::Volatile,
                text: volatile_text.clone(),
                tokens: volatile_tokens,
            },
        ];
        let total_tokens = static_tokens.saturating_add(semi_tokens).saturating_add(volatile_tokens);
        let cacheable_tokens = static_tokens.saturating_add(semi_tokens);

        if total_tokens > context_max {
            // Last-resort guard: hard cap the volatile section (drop recent
            // beyond the floor and evidence entirely) — still bounded.
            let static_len = static_text.len().saturating_add(semi_text.len());
            let allowed = context_max.saturating_mul(4).saturating_sub(static_len);
            let cut = truncate(&volatile_text, allowed);
            let cut_tokens = est.estimate_tokens(&cut);
            return Ok(AssembledContext {
                sections: vec![
                    static_section,
                    semi_section,
                    ContextSection {
                        class: MemoryClass::Volatile,
                        text: cut.clone(),
                        tokens: cut_tokens,
                    },
                ],
                total_tokens: cacheable_tokens.saturating_add(cut_tokens),
                cacheable_tokens,
                volatile_start: static_len,
            });
        }

        Ok(AssembledContext {
            sections,
            total_tokens,
            cacheable_tokens,
            volatile_start: static_text.len().saturating_add(semi_text.len()),
        })
    }
}

fn join_sections(a: &str, b: &str, c: &str, d: &str) -> String {
    let mut out = String::new();
    out.push_str(a);
    if !b.is_empty() {
        out.push_str("\n");
        out.push_str(b);
    }
    if !c.is_empty() {
        out.push_str("\n## Tools\n");
        out.push_str(c);
    }
    if !d.is_empty() {
        out.push_str("\n## Project rules\n");
        out.push_str(d);
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::ContextBudget;

    fn static_prefix() -> &'static str {
        "You are Kilo+.\nAct as a careful engineer."
    }

    fn ledger() -> TaskLedger {
        let mut l = TaskLedger::default();
        l.goal = "fix the parser".into();
        l.open_steps = vec!["reproduce crash".into()];
        l
    }

    fn turns(n: usize) -> Vec<RecentTurn> {
        (0..n)
            .map(|i| RecentTurn {
                role: if i % 2 == 0 { "user".into() } else { "assistant".into() },
                text: format!("turn {i} content {}", "x".repeat(500)),
            })
            .collect()
    }

    fn evidence(n: usize) -> Vec<Evidence> {
        (0..n)
            .map(|i| Evidence {
                path: format!("src/f{i}.rs"),
                snippet: format!("fn f{i}() {{}} // {}", "y".repeat(200)),
                score: (n - i) as f64,
            })
            .collect()
    }

    #[test]
    fn exact_32k_math_is_respected() {
        let b = ContextBudget::default();
        let budget_max = b.context_max();
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            "",
            &turns(400),
            &evidence(300),
            "err",
            &b,
        )
        .unwrap();
        assert!(
            ctx.total_tokens <= budget_max,
            "{} > {}",
            ctx.total_tokens,
            budget_max
        );
        assert!(ctx.total_tokens > 0);
    }

    #[test]
    fn oversized_volatile_trimmed_deterministically() {
        let b = ContextBudget::default();
        // 4000 turns x ~125 tokens = ~500K tokens of volatile: must come out
        // at the budget, oldest dropped first (newest retained).
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            "",
            &turns(4000),
            &[],
            "",
            &b,
        )
        .unwrap();
        assert!(ctx.total_tokens <= b.context_max());
        let render = ctx.render();
        // The newest turn survives, the oldest is gone.
        assert!(render.contains("turn 3999"));
        assert!(!render.contains("turn 0"), "oldest must be dropped first");
    }

    #[test]
    fn static_prefix_is_byte_stable_across_turns() {
        let b = ContextBudget::default();
        let a = ContextAssembler::assemble(
            static_prefix(),
            "extra",
            "tool schemas",
            "rules",
            &ledger(),
            "repo map",
            &turns(2),
            &evidence(1),
            "e1",
            &b,
        )
        .unwrap();
        let c = ContextAssembler::assemble(
            static_prefix(),
            "extra",
            "tool schemas",
            "rules",
            &ledger(),
            "repo map",
            &turns(200),
            &evidence(20),
            "e2 completely different",
            &b,
        )
        .unwrap();
        // Static sections must be byte-identical (provider prefix caching).
        assert_eq!(a.sections[0], c.sections[0], "static prefix drifted");
        assert_eq!(a.sections[1], c.sections[1], "semi-stable drifted");
        assert_ne!(a.sections[2], c.sections[2]);
        // Cacheable token count is identical too.
        assert_eq!(a.cacheable_tokens, c.cacheable_tokens);
    }

    #[test]
    fn evidence_score_ordering_and_archiving() {
        let b = ContextBudget::default();
        let mut ev = evidence(10);
        // Flip scores so the archive boundary is explicit.
        for (i, e) in ev.iter_mut().enumerate() {
            e.score = i as f64;
        }
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            "",
            &[],
            &ev,
            "",
            &b,
        )
        .unwrap();
        let render = ctx.render();
        // Highest score (9) must be inline; the lowest ones archived/replaced.
        assert!(render.contains("src/f9.rs"));
        let inline_count = render.matches("fn f").count();
        let archived_count = render.matches("archived").count();
        assert!(inline_count >= 1);
        
        assert_eq!(inline_count + archived_count, 10, "all evidence represented");
    }

    #[test]
    fn zero_budget_fails_cleanly() {
        let b = ContextBudget {
            system: 0,
            tools: 0,
            working: 0,
            retrieved: 0,
            recent: 0,
            output_reserve: 0,
            safety: 0,
        };
        let err = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
    }

    #[test]
    fn empty_inputs_produce_empty_volatile() {
        let b = ContextBudget::default();
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &TaskLedger::default(),
            "",
            &[],
            &[],
            "",
            &b,
        )
        .unwrap();
        assert!(ctx.total_tokens >= ctx.cacheable_tokens);
        let render = ctx.render();
        assert!(render.contains("You are Kilo+"));
    }

    #[test]
    fn unicode_and_huge_inputs_never_panic() {
        let b = ContextBudget::default();
        let mut hostile = Vec::new();
        for _i in 0..200 {
            hostile.push(RecentTurn {
                role: "user".into(),
                text: format!("😀{}", "汉".repeat(3000)),
            });
        }
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            &"m".repeat(100_000),
            &hostile,
            &[Evidence {
                path: "p".into(),
                snippet: "é".repeat(50_000),
                score: 1.0,
            }],
            &"e".repeat(100_000),
            &b,
        )
        .unwrap();
        assert!(ctx.total_tokens <= b.context_max());
        assert!(ctx.render().is_char_boundary(ctx.render().len()));
    }

    #[test]
    fn volatile_start_marks_cache_boundary() {
        let b = ContextBudget::default();
        let ctx = ContextAssembler::assemble(
            static_prefix(),
            "",
            "",
            "",
            &ledger(),
            "",
            &turns(1),
            &[],
            "",
            &b,
        )
        .unwrap();
        let render = ctx.render();
        assert!(ctx.volatile_start <= render.len());
        assert!(render[ctx.volatile_start..].contains("Recent conversation"));
    }
}
