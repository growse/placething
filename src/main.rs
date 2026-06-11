/// Placething — noun↔town swap server (Rust port of place_swap.py + server.py)
///
/// Noun extraction uses a stopword-filter approach instead of NLTK POS tagging:
/// we tokenise, lowercase, strip stopwords, then rank by frequency.
/// This produces "content words" which for a fun swap project is close enough to nouns.
use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Router,
};
use clap::Parser;
use once_cell::sync::Lazy;
use regex::Regex;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tower_http::limit::RequestBodyLimitLayer;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
struct Args {
    /// Path to towns.txt (TSV: rank, name, population, type)
    #[arg(default_value = "towns.txt")]
    towns: PathBuf,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8000)]
    port: u16,
}

// ---------------------------------------------------------------------------
// Stopwords  (English function words — not nouns)
// ---------------------------------------------------------------------------

static STOPWORDS: Lazy<std::collections::HashSet<&'static str>> = Lazy::new(|| {
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "can't",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "get",
        "got",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "let's",
        "like",
        "make",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "will",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
        // verb-ish forms / adverbs unlikely to be nouns
        "said",
        "went",
        "came",
        "come",
        "go",
        "know",
        "knew",
        "think",
        "thought",
        "see",
        "saw",
        "seen",
        "look",
        "looked",
        "take",
        "took",
        "taken",
        "give",
        "gave",
        "given",
        "tell",
        "told",
        "find",
        "found",
        "keep",
        "kept",
        "let",
        "put",
        "set",
        "seem",
        "seemed",
        "try",
        "tried",
        "ask",
        "asked",
        "turn",
        "turned",
        "move",
        "moved",
        "live",
        "called",
        "may",
        "might",
        "must",
        "shall",
        "used",
        "upon",
        "still",
        "even",
        "well",
        "back",
        "never",
        "always",
        "much",
        "many",
        "every",
        "ever",
        "quite",
        "rather",
        "whether",
        "however",
        "though",
        "although",
        "since",
        "while",
        "without",
        "within",
        "yes",
        "yet",
        "already",
        "almost",
        "away",
        "again",
        "together",
        "away",
        "really",
        "perhaps",
        "soon",
        "perhaps",
        "here",
        "there",
        "where",
        "then",
        "however",
        "therefore",
        "thus",
        "hence",
        "indeed",
        "else",
    ]
    .iter()
    .copied()
    .collect()
});

// ---------------------------------------------------------------------------
// Tokeniser  (same split strategy as Python: hyphenated words are one token)
// ---------------------------------------------------------------------------

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z]+(?:-[A-Za-z]+)*").unwrap());

/// Extract tokens (alphabetic + hyphenated) from text, preserving byte positions.
fn tokenise(text: &str) -> Vec<(usize, usize, &str)> {
    WORD_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end(), m.as_str()))
        .collect()
}

// ---------------------------------------------------------------------------
// Noun extraction (stopword-filter approach)
// ---------------------------------------------------------------------------

fn extract_nouns_ranked(text: &str) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for (_, _, tok) in tokenise(text) {
        if tok.len() <= 1 {
            continue;
        }
        let lower = tok.to_lowercase();
        if lower.contains('-') {
            continue; // skip hyphenated compound words
        }
        if !lower.chars().all(|c| c.is_ascii_alphabetic()) {
            continue;
        }
        if STOPWORDS.contains(lower.as_str()) {
            continue;
        }
        *counts.entry(lower).or_insert(0) += 1;
    }
    let mut ranked: Vec<(String, usize)> = counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(word, _)| word).collect()
}

// ---------------------------------------------------------------------------
// Town parsing
// ---------------------------------------------------------------------------

fn parse_towns(path: &PathBuf) -> Result<Vec<String>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading towns file {:?}", path))?;
    let towns: Vec<String> = raw
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(4, '\t').collect();
            if parts.len() >= 2 {
                Some(parts[1].trim().to_lowercase())
            } else {
                None
            }
        })
        .filter(|s| !s.is_empty())
        .collect();
    Ok(towns)
}

// ---------------------------------------------------------------------------
// Build bidirectional mapping  noun <-> town
// ---------------------------------------------------------------------------

fn build_mapping(nouns: &[String], towns: &[String]) -> HashMap<String, String> {
    let n = nouns.len().min(towns.len());
    let mut map = HashMap::with_capacity(n * 2);
    for i in 0..n {
        map.insert(nouns[i].clone(), towns[i].clone());
        map.insert(towns[i].clone(), nouns[i].clone());
    }
    map
}

// ---------------------------------------------------------------------------
// Case preservation
// ---------------------------------------------------------------------------

fn apply_case(replacement: &str, original: &str) -> String {
    if original
        .chars()
        .all(|c| c.is_uppercase() || !c.is_alphabetic())
    {
        replacement.to_uppercase()
    } else if original.starts_with(|c: char| c.is_uppercase()) {
        let mut chars = replacement.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        }
    } else {
        replacement.to_lowercase()
    }
}

// ---------------------------------------------------------------------------
// Swap logic (port of place_swap.py:swap_words)
//
// Four categories based on the key's character composition:
//   mixed_keys  – contains spaces AND non-alpha chars → regex with placeholder
//   space_keys  – contains spaces, only alpha+space → sliding window over tokens
//   hyphen_keys – contains hyphens, no spaces → O(1) dict lookup on hyphen tokens
//   word_keys   – plain alpha → O(1) dict lookup on normal tokens
// ---------------------------------------------------------------------------

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\x00N(\d+)\x00").unwrap());

fn swap_words(text: &str, mapping: &HashMap<String, String>) -> String {
    // Categorise mapping keys
    let mut mixed_keys: Vec<(&str, &str)> = Vec::new();
    let mut space_keys: Vec<(&str, &str)> = Vec::new();
    let mut hyphen_keys: HashMap<&str, &str> = HashMap::new();
    let mut word_keys: HashMap<&str, &str> = HashMap::new();

    for (k, v) in mapping {
        let has_space = k.contains(' ');
        let has_non_alpha_non_space = k.chars().any(|c| !c.is_alphabetic() && c != ' ');

        if has_space && has_non_alpha_non_space {
            mixed_keys.push((k.as_str(), v.as_str()));
        } else if has_space {
            space_keys.push((k.as_str(), v.as_str()));
        } else if k.contains('-') {
            hyphen_keys.insert(k.as_str(), v.as_str());
        } else {
            word_keys.insert(k.as_str(), v.as_str());
        }
    }

    // Phase 1: replace mixed_keys using a regex (sorted longest-first)
    mixed_keys.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

    let mut placeholders: Vec<String> = Vec::new();
    let mut current_text = text.to_string();

    if !mixed_keys.is_empty() {
        let patterns: Vec<String> = mixed_keys.iter().map(|(k, _)| regex::escape(k)).collect();
        let joined = patterns.join("|");
        let re = Regex::new(&format!("(?i){}", joined)).unwrap();

        current_text = {
            let mut result = String::with_capacity(current_text.len());
            let mut last = 0;
            for m in re.find_iter(&current_text) {
                result.push_str(&current_text[last..m.start()]);
                let original = m.as_str();
                let lower = original.to_lowercase();
                if let Some(repl) = mixed_keys
                    .iter()
                    .find(|(k, _)| *k == lower.as_str())
                    .map(|(_, v)| *v)
                {
                    let ph = format!("\x00N{}\x00", placeholders.len());
                    placeholders.push(apply_case(repl, original));
                    result.push_str(&ph);
                } else {
                    result.push_str(original);
                }
                last = m.end();
            }
            result.push_str(&current_text[last..]);
            result
        };
    }

    // Phase 2a: sliding window for space_keys
    // Pre-split keys and build a first-word index so we only check candidates
    // whose first word matches the current token — O(tokens × avg_candidates)
    // instead of O(tokens × all_space_keys).
    struct PhraseEntry<'a> {
        words: Vec<String>, // lowercase words
        val: &'a str,
    }
    let mut first_word_index: HashMap<String, Vec<PhraseEntry>> = HashMap::new();
    for (key, val) in &space_keys {
        let words: Vec<String> = key.split(' ').map(|w| w.to_lowercase()).collect();
        if let Some(first) = words.first() {
            first_word_index
                .entry(first.clone())
                .or_default()
                .push(PhraseEntry { words, val });
        }
    }
    // Sort each bucket longest-first so longer phrases win
    for bucket in first_word_index.values_mut() {
        bucket.sort_by(|a, b| b.words.len().cmp(&a.words.len()));
    }

    // Collect token positions
    let tokens: Vec<(usize, usize, &str)> = WORD_RE
        .find_iter(&current_text)
        .map(|m| (m.start(), m.end(), m.as_str()))
        .collect();

    // We'll mark which tokens have been replaced using a bool vec
    let mut replaced = vec![false; tokens.len()];

    // Collect replacements: (start_byte, end_byte, replacement_string)
    let mut replacements: Vec<(usize, usize, String)> = Vec::new();

    {
        let n = tokens.len();
        let mut i = 0;
        'outer: while i < n {
            if replaced[i] {
                i += 1;
                continue;
            }
            let lower_i = tokens[i].2.to_lowercase();
            if let Some(candidates) = first_word_index.get(&lower_i) {
                for entry in candidates {
                    let wlen = entry.words.len();
                    if i + wlen > n {
                        continue;
                    }
                    let mut matched = true;
                    for j in 0..wlen {
                        if tokens[i + j].2.to_lowercase() != entry.words[j] {
                            matched = false;
                            break;
                        }
                        if j > 0 {
                            let sep = &current_text[tokens[i + j - 1].1..tokens[i + j].0];
                            if sep != " " {
                                matched = false;
                                break;
                            }
                        }
                    }
                    if matched {
                        let span_start = tokens[i].0;
                        let span_end = tokens[i + wlen - 1].1;
                        let original = &current_text[span_start..span_end];
                        replacements.push((span_start, span_end, apply_case(entry.val, original)));
                        for j in 0..wlen {
                            replaced[i + j] = true;
                        }
                        i += wlen;
                        continue 'outer;
                    }
                }
            }
            i += 1;
        }
    }

    // Phase 2b: single word_keys and hyphen_keys
    for (idx, &(start, end, tok)) in tokens.iter().enumerate() {
        if replaced[idx] {
            continue;
        }
        let lower = tok.to_lowercase();
        let repl = if tok.contains('-') {
            hyphen_keys.get(lower.as_str()).copied()
        } else {
            word_keys.get(lower.as_str()).copied()
        };
        if let Some(r) = repl {
            replaced[idx] = true;
            replacements.push((start, end, apply_case(r, tok)));
        }
    }

    // Apply all replacements (sorted by position)
    replacements.sort_by_key(|r| r.0);
    let mut out = String::with_capacity(current_text.len());
    let mut last = 0;
    for (start, end, repl) in &replacements {
        out.push_str(&current_text[last..*start]);
        out.push_str(repl);
        last = *end;
    }
    out.push_str(&current_text[last..]);

    // Phase 3: restore placeholders
    let final_text = PLACEHOLDER_RE
        .replace_all(&out, |caps: &regex::Captures| {
            let idx: usize = caps[1].parse().unwrap();
            placeholders[idx].clone()
        })
        .into_owned();

    final_text
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct AppState {
    towns: Vec<String>,
}

// ---------------------------------------------------------------------------
// Axum handler
// ---------------------------------------------------------------------------

async fn transform(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    // Check Content-Type
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !ct.starts_with("text/plain") {
        return (StatusCode::UNSUPPORTED_MEDIA_TYPE, "Expected text/plain").into_response();
    }

    let text = match std::str::from_utf8(&body) {
        Ok(t) => t.to_string(),
        Err(_) => return (StatusCode::BAD_REQUEST, "Invalid UTF-8").into_response(),
    };

    let towns = state.towns.clone();

    // CPU-bound work off the async executor
    let result = tokio::task::spawn_blocking(move || {
        let nouns = extract_nouns_ranked(&text);
        let mapping = build_mapping(&nouns, &towns);
        swap_words(&text, &mapping)
    })
    .await;

    match result {
        Ok(output) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; charset=utf-8",
            )],
            output,
        )
            .into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {e}")).into_response(),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let towns = parse_towns(&args.towns)?;
    println!("Loaded {} towns", towns.len());

    let state = Arc::new(AppState { towns });

    let app = Router::new()
        .route("/transform", post(transform))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyLimitLayer::new(1024 * 1024)) // 1 MB
        .with_state(state);

    let addr = format!("[::]:{}", args.port);
    println!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
