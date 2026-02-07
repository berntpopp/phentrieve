# LLM Annotation System

Phentrieve's LLM annotation system extracts HPO terms from clinical text using large language models. It offers three annotation modes and optional post-processing refinement.

## Annotation Modes

### Mode 1: Direct Text (`direct`)

The LLM extracts HPO terms purely from its own knowledge — single API call, no tools.

```
  "Patient has recurrent seizures and no cardiac abnormalities"
       │
       │  DirectTextStrategy.annotate(text, language="en")
       ▼
 ┌──────────────────┐
 │  Prompt Assembly  │  get_prompt(mode=DIRECT, language="en")
 │                   │  → loads prompts/templates/direct_text/en.yaml
 │  Builds messages: │  → system: "You are a clinical genetics expert..."
 │    [system,       │  → few-shot: seizures/ID example pair
 │     examples,     │  → user: "Extract HPO annotations from: {text}"
 │     user]         │
 └────────┬─────────┘
          │
          │  LLMProvider.complete(messages)
          │  → single LiteLLM API call to e.g. github/gpt-4o
          ▼
 ┌──────────────────┐
 │   LLM Response    │  Raw JSON:
 │                   │  {"annotations": [
 │                   │    {"hpo_id": "HP:0001250", "assertion": "affirmed"},
 │                   │    {"hpo_id": "HP:0001627", "assertion": "negated"}
 │                   │  ]}
 └────────┬─────────┘
          │
          │  _parse_response(response) + _validate_annotations(annotations)
          │  → normalize ID formats (HP_0001250 → HP:0001250)
          │  → check each ID exists in HPODatabase
          │  → drop invalid IDs
          ▼
 ┌──────────────────┐
 │ AnnotationResult  │  .annotations = [HPOAnnotation(...), ...]
 │                   │  .token_usage = TokenUsage(prompt=800, completion=200)
 │                   │  .mode = DIRECT
 └────────┬─────────┘
          │
          │  Pipeline-level hallucination filter:
          │  → run process_clinical_text(text) to get Phentrieve candidates
          │  → remove any LLM annotations not in candidates
          ▼
   Final AnnotationResult
```

Fast and cheap. LLM may hallucinate HPO IDs from its training data, but the
pipeline filters them against Phentrieve retrieval results after the LLM call.

---

### Mode 2: Tool-Guided Term Search (`tool_term`)

The LLM identifies clinical phrases, then searches the HPO database for each one.

```
  "Patient has recurrent seizures and no cardiac abnormalities"
       │
       │  ToolGuidedStrategy.annotate(text, language="en")
       ▼
 ┌──────────────────┐
 │  Prompt Assembly  │  get_prompt(mode=TOOL_TERM, language="en")
 │                   │  → loads prompts/templates/tool_guided/en_term_search.yaml
 │  Builds messages: │  → system: "Identify phenotypes, use query_hpo_terms..."
 │    [system, user] │  → user: "Extract HPO annotations using tools: {text}"
 └────────┬─────────┘
          │
          │  LLMProvider.complete_with_tools(messages, tool_executor, max_iterations=5)
          │
          │  Each iteration = one API call to the LLM.
          │  The LLM either returns tool_calls (loop continues)
          │  or a final text response (loop ends).
          ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  API call 1:  LLM receives [system, user]                   │
 │               LLM responds with tool_calls:                  │
 │                 query_hpo_terms({query: "recurrent seizures"})│
 │                 query_hpo_terms({query: "cardiac abnormality"})
 │               (finish_reason: "tool_calls")                  │
 │                          │                                   │
 │               Both tool calls executed locally:              │
 │                                                              │
 │                 ToolExecutor.execute("query_hpo_terms",       │
 │                   {query: "recurrent seizures"})              │
 │                 → DenseRetriever.query() → ChromaDB          │
 │                 → HP:0001250 Seizure (0.92)                  │
 │                   HP:0007359 Recurrent seizure (0.95)        │
 │                                                              │
 │                 ToolExecutor.execute("query_hpo_terms",       │
 │                   {query: "cardiac abnormality"})             │
 │                 → HP:0001627 Abnormality of the cardiovascular system (0.89)
 │                   HP:0030680 Cardiac anomaly (0.85)          │
 │                                                              │
 │               Both results appended to conversation          │
 │               as tool role messages                          │
 │                          │                                   │
 │                          ▼                                   │
 │  API call 2:  LLM receives [system, user, assistant+tools,   │
 │                             tool_result_1, tool_result_2]    │
 │               LLM has all search results, produces final JSON│
 │               (finish_reason: "stop" — loop ends)            │
 │                                                              │
 │               content: {"annotations": [                     │
 │                 {"hpo_id":"HP:0007359", "assertion":"affirmed"},
 │                 {"hpo_id":"HP:0001627", "assertion":"negated"}│
 │               ]}                                             │
 └────────────────────────────┬─────────────────────────────────┘
                              │
                              │  _parse_response() + _validate_annotations()
                              │
                              │  Hallucination filter:
                              │  → collect candidate IDs from all query_hpo_terms results
                              │  → remove any LLM annotations not in candidates
                              ▼
                       AnnotationResult
                         .annotations = [...]
                         .tool_calls = [ToolCall("query_hpo_terms", ...), ...]
                         .token_usage = TokenUsage(prompt=2400, completion=600, api_calls=2)
```

LLM controls which phrases to search. A single API response can request multiple
tool calls at once (both queries above happen in one round-trip). IDs come from
the database, and any IDs the LLM invents beyond the search results are filtered out.

---

### Mode 3: Tool-Guided Text Processing (`tool_text`)

Phentrieve processes the full text first, then the LLM selects from the candidates.

```
  "Patient has recurrent seizures and no cardiac abnormalities"
       │
       │  ToolGuidedStrategy.annotate(text, language="en")
       ▼
 ┌──────────────────┐
 │  Prompt Assembly  │  get_prompt(mode=TOOL_TEXT, language="en")
 │                   │  → loads prompts/templates/tool_guided/en_text_process.yaml
 │  Builds messages: │  → system: "Select ONLY from returned candidates..."
 │    [system, user] │  → user: "Review and select HPO annotations: {text}"
 └────────┬─────────┘
          │
          │  Pipeline runs process_clinical_text directly
          │  (no initial API call — results injected into prompt)
          ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  ToolExecutor.execute("process_clinical_text",               │
 │    {text: "Patient has...", language: "en"})                  │
 │                          │                                   │
 │                          ▼                                   │
 │              ┌──────────────────────────┐                    │
 │              │   Phentrieve Pipeline     │                    │
 │              │                          │                    │
 │              │  TextProcessingPipeline   │                    │
 │              │  .process(text, "en")     │                    │
 │              │    │                      │                    │
 │              │    ├─► Chunk: "recurrent seizures"            │
 │              │    │   assertion: AFFIRMED │                    │
 │              │    │                      │                    │
 │              │    └─► Chunk: "no cardiac abnormalities"      │
 │              │        assertion: NEGATED  │                    │
 │              │                          │                    │
 │              │  DenseRetriever.query()   │                    │
 │              │  per chunk → ChromaDB     │                    │
 │              │                          │                    │
 │              │  orchestrate_hpo_extraction()                 │
 │              │  → aggregate & deduplicate │                    │
 │              └────────────┬─────────────┘                    │
 │                           │                                  │
 │                           ▼                                  │
 │              candidates injected into prompt:                │
 │                HP:0001250 Seizure (affirmed, 0.92)           │
 │                HP:0007359 Recurrent seizure (affirmed, 0.95) │
 │                HP:0001627 Cardiac abnormality (negated, 0.89)│
 │                HP:0030680 Cardiac anomaly (negated, 0.72)    │
 │                           │                                  │
 │                           ▼                                  │
 │  Single API call: LLM receives candidates, selects & annotates
 │    content: {"annotations": [                                │
 │      {"hpo_id":"HP:0007359", "assertion":"affirmed"},        │
 │      {"hpo_id":"HP:0001627", "assertion":"negated"},         │
 │      {"hpo_id":"HP:9999999", "assertion":"affirmed"}  ← hallucinated!
 │    ]}                                                        │
 └────────────────────────────┬─────────────────────────────────┘
                              │
                              │  Hallucination filter (TOOL_TEXT only):
                              │  → collect candidate IDs from tool results:
                              │    {HP:0001250, HP:0007359, HP:0001627, HP:0030680}
                              │  → HP:0007359 ✓ in candidates → keep
                              │  → HP:0001627 ✓ in candidates → keep
                              │  → HP:9999999 ✗ NOT in candidates → remove + log warning
                              ▼
                       AnnotationResult
                         .annotations = [HP:0007359 affirmed, HP:0001627 negated]
                         .tool_calls = [ToolCall("process_clinical_text", ...)]
                         .token_usage = TokenUsage(prompt=1800, completion=400)
```

Most accurate mode. LLM acts as validator/selector, constrained to retrieved candidates.
The Phentrieve pipeline always runs directly (no overhead API call) — results are
injected into the prompt, so only a single LLM API call is made.

---

## Post-Processing (Optional)

After any mode, one or more refinement steps can be chained. Each step receives the annotations + original text and returns refined annotations.

### Individual Steps

When run individually, each step makes a separate LLM API call:

```
  Primary Annotations + original text
       │
       │  PostProcessor.process(annotations, original_text, language)
       ▼
 ┌──────────────────┐
 │   Validation      │  LLM re-reads source text with each annotation
 │                   │  → validated: keep as-is
 │ LLMProvider       │  → corrected: adjust assertion/confidence
 │ .complete(        │  → removed:   drop with logged reason
 │   validation.yaml │     e.g. "no evidence for HP:0001234 in text"
 │   + annotations   │
 │   + text)         │
 └────────┬─────────┘
          │
          │  PostProcessor.process(annotations, original_text, language)
          ▼
 ┌──────────────────┐
 │   Refinement      │  LLM checks if more specific HPO terms apply
 │                   │  optionally calls query_hpo_terms tool
 │ LLMProvider       │  → kept:    HP:0001250 already optimal
 │ .complete(        │  → refined: HP:0001250 Seizure
 │   refinement.yaml │             → HP:0007359 Recurrent seizure
 │   + annotations   │             (text says "recurrent")
 │   + text)         │
 └────────┬─────────┘
          │
          │  PostProcessor.process(annotations, original_text, language)
          ▼
 ┌──────────────────┐
 │  Assertion Review │  LLM focuses on negation/uncertainty markers
 │                   │  built-in prompt (no yaml template)
 │ LLMProvider       │
 │ .complete(        │  Checks for: "no", "without", "denies", "absent",
 │   review prompt   │  "possible", "suspected", "may have", "history of"
 │   + annotations   │
 │   + text)         │  → original: affirmed → correct: negated
 │                   │    reason: "preceded by 'no'"
 └────────┬─────────┘
          │
          ▼
  Final Annotations
```

Each step is independent and failure-tolerant — if any step errors, the previous annotations carry forward unchanged.

### Combined Post-Processing (`combined`)

When all three individual steps are requested, the pipeline **automatically routes to a single combined LLM call** instead of making 3 separate calls. This eliminates 2 API round-trips (~4–30s saved per document).

```
  Primary Annotations + original text
       │
       │  CombinedPostProcessor.process(annotations, original_text, language)
       │  (single LLM call covers all 3 tasks)
       ▼
 ┌──────────────────────────────────────────────────────────────┐
 │   Combined Review (1 API call)                               │
 │                                                              │
 │   LLM performs in a single pass:                             │
 │     1. Validation  — remove false positives                  │
 │     2. Assertion   — correct negation/uncertainty errors     │
 │     3. Refinement  — upgrade to more specific HPO terms      │
 │                                                              │
 │   Output:                                                    │
 │     validated_annotations: [{hpo_id, assertion, ...}]        │
 │     removed_annotations:   [{hpo_id, reason}]                │
 │     refined_annotations:   [{original_hpo_id,                │
 │                               refined_hpo_id, ...}]          │
 └────────────────────────────┬─────────────────────────────────┘
                              │
                              ▼
                       PostProcessingStats
                         .removed = N
                         .assertions_changed = M
                         .terms_refined = K
```

**Auto-routing**: When the pipeline detects all 3 individual steps (`validation`, `assertion_review`, `refinement`) are requested, it automatically uses the combined post-processor. Individual steps remain available when only 1–2 are selected.

**Direct use**: You can also request `combined` explicitly via `--postprocess combined`.

---

## CLI Examples

### Annotate text

```bash
# Basic annotation (default: tool_text mode, github/gpt-4o)
phentrieve llm annotate "Patient has recurrent seizures and no cardiac abnormalities"

# Choose mode and model
phentrieve llm annotate "Patient has seizures and hypotonia" \
    --mode direct --model gemini/gemini-2.0-flash

# German text with auto-detection
phentrieve llm annotate "Krampfanfälle und Muskelhypotonie" --language auto

# From file, with post-processing, output as JSON
phentrieve llm annotate --input clinical_note.txt \
    --postprocess validation,assertion_review \
    --format json --output result.json

# Combined post-processing (all 3 steps in a single LLM call)
phentrieve llm annotate --input clinical_note.txt \
    --postprocess combined --format json

# All 3 individual steps → auto-routed to combined (same result, 1 API call)
phentrieve llm annotate --input clinical_note.txt \
    --postprocess validation,assertion_review,refinement

# Include HPO definitions/synonyms in output
phentrieve llm annotate "seizures and intellectual disability" --include-details

# Debug mode — shows tool calls, filtering steps, token counts
phentrieve llm annotate "no seizures, possible cardiac defect" --debug

# Pipe from stdin
echo "recurrent febrile seizures" | phentrieve llm annotate
```

### Benchmark against gold standard

```bash
# Benchmark one mode against PhenoBERT data
phentrieve llm benchmark tests/data/en/phenobert/ \
    --dataset GeneReviews --mode tool_text

# Compare all three modes
phentrieve llm benchmark tests/data/en/phenobert/ \
    --dataset GeneReviews --mode all

# Quick test with 5 documents and debug output
phentrieve llm benchmark tests/data/en/phenobert/ --limit 5 --debug

# With confidence intervals
phentrieve llm benchmark tests/data/en/phenobert/ --bootstrap-ci

# Simple JSON benchmark file
phentrieve llm benchmark tests/data/benchmarks/german/tiny_v1.json
```

### Compare configurations

```bash
# Compare modes for one model
phentrieve llm compare -t tests/data/benchmarks/german/tiny_v1.json \
    --modes direct,tool_term,tool_text

# Compare models
phentrieve llm compare -t tests/data/benchmarks/german/tiny_v1.json \
    --models github/gpt-4o,gemini/gemini-2.0-flash

# Full grid: models x modes x post-processing
phentrieve llm compare -t tests/data/benchmarks/german/tiny_v1.json \
    --models github/gpt-4o,gemini/gemini-2.0-flash \
    --modes direct,tool_text \
    --postprocess none,validation,combined
```

### List models and auth status

```bash
phentrieve llm models
```

---

## Internal Optimizations

### Prompt Template Caching

Prompt templates are loaded from YAML files on disk. The `load_prompt_template()` function uses `@lru_cache` to cache templates per `(mode, language, variant)` combination, avoiding redundant disk reads when annotating multiple documents.

```python
# Cache is transparent — no API change
template = load_prompt_template(PostProcessingStep.VALIDATION, "en")

# Clear cache if needed (e.g., after editing templates during development)
load_prompt_template.cache_clear()
```

### Shared Utilities (`phentrieve/llm/utils.py`)

Common helper functions used across annotation strategies and post-processors are consolidated in `phentrieve/llm/utils.py`:

| Function | Purpose |
|----------|---------|
| `extract_json(text)` | Extract JSON object from LLM responses (handles markdown code blocks) |
| `normalize_hpo_id(hpo_id)` | Normalize HPO IDs to `HP:XXXXXXX` format (accepts various separators) |
| `parse_assertion(assertion_str)` | Parse assertion strings to `AssertionStatus` enum (accepts synonyms like "absent", "denied", "probable") |

These functions are imported by `tool_guided.py`, `direct_text.py`, `validation.py`, `assertion_review.py`, `refinement.py`, and `combined.py`.
