# ADR 007: Three-Tier Prompt Parsing for LLM Outputs

**Date:** June 2026  
**Status:** Accepted

## Context

The LLM Judge (Mistral 7B via Ollama) processes uncertain reviews and returns a JSON classification (`negative`, `neutral`, `positive`) along with reasoning. 

While Mistral 7B is generally capable of following JSON output instructions, smaller models (like the 4-billion parameter `nemotron` variant we use by default for speed) are notoriously inconsistent. They frequently:
- Wrap JSON in markdown code blocks (````json ... ````).
- Append conversational text before or after the JSON ("Here is your classification: { ... }").
- Output malformed JSON (missing quotes, trailing commas).
- Ignore the JSON schema entirely and just output text ("The review is clearly negative").

We needed a robust way to extract the sentiment class from the LLM's raw text stream without dropping too many records due to parsing errors.

## Decision

We implemented a **three-tier fallback parsing strategy** in `src/llm_judge/prompt_builder.py`.

When attempting to parse the LLM's response, the system tries:

1. **Tier 1: Markdown Extraction.** Look for a ````json ... ```` code block. If found, extract the contents and attempt `json.loads()`.
2. **Tier 2: Raw JSON parsing.** If Tier 1 fails, attempt `json.loads()` on the raw, unaltered response string.
3. **Tier 3: Regex Heuristic Fallback.** If both JSON parsing attempts fail, abandon structured parsing. Scan the text with regex boundaries for the target words (`\bnegative\b`, `\bneutral\b`, `\bpositive\b`). If exactly one class word is found, assume that is the prediction.

If all three tiers fail, the response is marked as a parsing error and dropped.

## Rationale

- **Resilience:** Relying solely on `json.loads()` results in a high failure rate for sub-7B models. The three-tier system salvages over 90% of otherwise "failed" responses.
- **Cost:** Implementing strict structured generation (e.g., using Ollama's `format="json"` API, or grammar-based sampling) restricts the model's ability to "think out loud" before classifying (Chain-of-Thought). Allowing the model to output unstructured reasoning followed by a class word significantly improves its accuracy. The Tier 3 regex fallback is required to capture this unstructured output.

## Consequences

- **Positive:** We can use smaller, faster, cheaper models because we do not require perfect JSON syntax from them.
- **Negative:** The Tier 3 regex fallback is vulnerable to adversarial or complex outputs. If the model outputs "This is not positive, it is negative", the regex will find both words and fail, or worse, match the wrong one if the other is misspelled.
- **Negative:** If we change the classes (e.g., adding `mixed`), the regex patterns in Tier 3 must be updated alongside the prompt.
