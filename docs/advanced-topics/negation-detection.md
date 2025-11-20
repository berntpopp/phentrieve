# Clinical Negation Detection with ConText

Phentrieve implements the **ConText algorithm** (originally developed by the medspaCy team) for detecting negation, uncertainty, and other assertion attributes in clinical text. This document provides technical details about the implementation, file structure, and detection logic.

## Overview

The ConText algorithm identifies clinical assertions by detecting trigger phrases and determining their scope and direction in text. This enables Phentrieve to distinguish between:

- **Affirmed** phenotypes: "Patient has seizures"
- **Negated** phenotypes: "No evidence of seizures"
- **Normal** findings: "Heart sounds are normal"
- **Uncertain** conditions: "Possible developmental delay"

**Key Innovation:** Direction-aware scope detection with TERMINATE boundaries prevents negation from incorrectly affecting unrelated text.

## Implementation Architecture

### Detection Strategy Hierarchy

Phentrieve uses a **two-tier detection strategy** with automatic fallback:

```
┌─────────────────────────────────────┐
│  1. DependencyAssertionDetector    │ ← Primary (if spaCy available)
│     Uses: Dependency parsing        │
│     Accuracy: Highest               │
│     Speed: Moderate                 │
└─────────────────────────────────────┘
              ↓ (fallback)
┌─────────────────────────────────────┐
│  2. KeywordAssertionDetector        │ ← Fallback
│     Uses: ConText rules + patterns  │
│     Accuracy: High                  │
│     Speed: Fast                     │
└─────────────────────────────────────┘
```

**Priority Logic:** The `CombinedAssertionDetector` tries dependency parsing first. If inconclusive or spaCy unavailable, it falls back to keyword-based ConText detection.

### File Structure

```
phentrieve/text_processing/
├── assertion_detection.py          # Main implementation
│   ├── Direction (enum)             # FORWARD, BACKWARD, BIDIRECTIONAL, TERMINATE, PSEUDO
│   ├── TriggerCategory (enum)       # NEGATED_EXISTENCE, POSSIBLE_EXISTENCE, etc.
│   ├── ConTextRule (dataclass)      # Frozen dataclass for thread-safe rules
│   ├── parse_context_rules()        # JSON parser for medspaCy format
│   ├── KeywordAssertionDetector     # ConText-based detection
│   ├── DependencyAssertionDetector  # spaCy dependency-based detection
│   └── CombinedAssertionDetector    # Orchestrator with fallback logic
│
└── default_lang_resources/         # Language-specific ConText rules
    ├── context_rules_en.json        # English (26 rules)
    ├── context_rules_de.json        # German (26 rules) - resolves issue #79
    ├── context_rules_es.json        # Spanish (24 rules)
    ├── context_rules_fr.json        # French (24 rules, includes BIDIRECTIONAL)
    ├── context_rules_nl.json        # Dutch (22 rules)
    └── normality_cues.json          # Phentrieve-specific (not in ConText standard)
```

**Total ConText Rules:** 122 rules across 5 languages

## ConText Rule Format

### medspaCy-Compatible JSON Structure

```json
{
  "context_rules": [
    {
      "literal": "no",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {
        "source": "medspaCy",
        "language": "en"
      }
    },
    {
      "literal": "ruled out",
      "category": "NEGATED_EXISTENCE",
      "direction": "BACKWARD",
      "metadata": {
        "source": "context-additions",
        "language": "en"
      }
    },
    {
      "literal": "but",
      "category": "TERMINATE",
      "direction": "TERMINATE",
      "metadata": {
        "source": "context-conjunctions",
        "language": "en"
      }
    }
  ]
}
```

### Rule Categories

| Category | Purpose | Example Triggers |
|----------|---------|------------------|
| `NEGATED_EXISTENCE` | Negates concept existence | "no", "without", "denies", "absence of" |
| `POSSIBLE_EXISTENCE` | Indicates uncertainty | "possible", "maybe", "suspected" |
| `HYPOTHETICAL` | Future/conditional | "if", "whether", "should" |
| `HISTORICAL` | Past condition | "history of", "previously had" |
| `FAMILY` | Family history | "family history", "mother has" |
| `TERMINATE` | Scope boundary | "but", "however", "although" |
| `PSEUDO` | False positive prevention | "not only", "no increase", "not excluded" |

**Current Implementation:** Phentrieve v0.1.0 fully implements `NEGATED_EXISTENCE`, `TERMINATE`, and `PSEUDO`. Other categories reserved for future expansion.

### Direction Types

```python
class Direction(Enum):
    FORWARD = "FORWARD"          # Trigger affects text AFTER it
    BACKWARD = "BACKWARD"        # Trigger affects text BEFORE it
    BIDIRECTIONAL = "BIDIRECTIONAL"  # Trigger affects text on BOTH sides
    TERMINATE = "TERMINATE"      # Marks scope boundary
    PSEUDO = "PSEUDO"            # Prevents false positives
```

## Detection Algorithm

### Three-Pass Detection Logic

```python
def _detect_with_context_rules(text, rules):
    """ConText detection with direction awareness and TERMINATE handling."""

    # PASS 1: Find PSEUDO rules (false positives)
    # - Identifies phrases like "not only" that LOOK like negation but aren't
    # - Records text spans to prevent substring matches later
    pseudo_spans = []
    for rule in rules where direction == PSEUDO:
        if rule.literal found in text with word boundaries:
            pseudo_spans.append((start_pos, end_pos))

    # PASS 2: Find TERMINATE triggers (scope boundaries)
    # - Identifies conjunctions like "but", "however" that limit scope
    # - Used in Pass 3 to stop scope expansion at boundaries
    terminate_positions = []
    for rule in rules where direction == TERMINATE:
        if rule.literal found in text with word boundaries:
            terminate_positions.append((start_pos, end_pos))

    # PASS 3: Process NEGATED_EXISTENCE and other categories
    for rule in rules where category == NEGATED_EXISTENCE:
        if rule.literal found in text with word boundaries:
            # Skip if overlaps with PSEUDO span
            if overlaps_with_pseudo_spans:
                continue

            # Extract scope based on direction, respecting TERMINATE boundaries
            scope = extract_scope(
                text,
                trigger_position,
                rule.direction,
                terminate_positions
            )

            if scope contains clinical concepts:
                mark_as_negated()
```

### Direction-Aware Scope Extraction

#### FORWARD Direction

```
Text: "No fever or cough"
      ^   ^^^^^^^^^^^^^
      |   scope (AFTER trigger)
      trigger

Extraction:
1. Find trigger: "no" at position 0
2. Extract N words AFTER trigger (default: 7 words)
3. Stop at TERMINATE if encountered
4. Result: "fever or cough"
```

#### BACKWARD Direction

```
Text: "Pneumonia was ruled out"
      ^^^^^^^^^     ^^^^^^^^^^
      scope         trigger
      (BEFORE)

Extraction:
1. Find trigger: "ruled out" at position 14
2. Extract N words BEFORE trigger (default: 7 words)
3. Stop at TERMINATE if encountered
4. Result: "Pneumonia was"
```

#### BIDIRECTIONAL Direction

```
Text: "Patient ne présente pas de fièvre"
      ^^^^^^^ ^^^^^^^^^^^^^^^^ ^^^^^^^^
      before   trigger (ne...pas) after

Extraction (French discontinuous negation):
1. Find trigger: "ne ... pas"
2. Extract N words BEFORE and AFTER
3. Combine scopes with trigger in middle
4. Result: "Patient [ne...pas] fièvre"
```

### TERMINATE Scope Boundaries

**Problem:** Without boundaries, negation can leak across sentence boundaries.

```
❌ WITHOUT TERMINATE:
"No fever but has persistent cough"
 ^   ^^^^     ^^^ ^^^^^^^^^^^ ^^^^^
 |   negated? |   incorrectly negated!
 trigger      boundary

✅ WITH TERMINATE:
"No fever but has persistent cough"
 ^   ^^^^  X   ^^^ ^^^^^^^^^^^ ^^^^^
 |   ✓     |   ✓ correctly affirmed
 trigger   stop  scope boundary
```

**Implementation:**
1. Detect all TERMINATE triggers in text
2. When extracting FORWARD scope, stop at first TERMINATE after trigger
3. When extracting BACKWARD scope, stop at last TERMINATE before trigger
4. This prevents negation scope from crossing conjunctions

**TERMINATE Rules by Language:**

| Language | Conjunctions |
|----------|-------------|
| English | "but", "however", "although" |
| German | "aber", "jedoch", "obwohl" |
| Spanish | "pero", "sin embargo", "aunque" |
| French | "mais", "cependant", "bien que" |
| Dutch | "maar", "echter", "hoewel" |

### PSEUDO False Positive Prevention

**Problem:** Some phrases contain negation words but don't negate concepts.

```
❌ WITHOUT PSEUDO:
"Not only fever but also cough"
 ^^^ ^^^^ ^^^^^
 |   incorrectly detected as negation

✅ WITH PSEUDO:
"Not only fever but also cough"
 ^^^^^^^^
 |
 PSEUDO rule matches → prevents "not" from triggering
```

**Overlap Detection:** When a PSEUDO rule matches "not only", any shorter overlapping trigger like "not" is prevented from matching that same text span.

**PSEUDO Rules by Language:**

| Language | Phrases |
|----------|---------|
| English | "not only", "not excluded", "no increase" |
| German | "nicht nur", "nicht ausgeschlossen", "keine Zunahme" |
| Spanish | "no solo", "no excluido", "sin aumento" |
| French | "non seulement", "pas exclu", "pas d'augmentation" |
| Dutch | "niet alleen", "niet uitgesloten", "geen toename" |

## Examples

### Example 1: Simple Negation (FORWARD)

```python
from phentrieve.text_processing.assertion_detection import KeywordAssertionDetector

detector = KeywordAssertionDetector(language="en")
text = "Patient denies fever"

status, details = detector.detect(text)
# status: AssertionStatus.NEGATED
# details: {'keyword_negated_scopes': ['denies: fever']}
```

**Analysis:**
- Trigger: "denies" (FORWARD direction)
- Scope: "fever" (extracted AFTER trigger)
- Result: NEGATED

### Example 2: BACKWARD Negation

```python
detector = KeywordAssertionDetector(language="en")
text = "Seizures were ruled out"

status, details = detector.detect(text)
# status: AssertionStatus.NEGATED
# details: {'keyword_negated_scopes': ['ruled out: seizures were']}
```

**Analysis:**
- Trigger: "ruled out" (BACKWARD direction)
- Scope: "Seizures were" (extracted BEFORE trigger)
- Result: NEGATED

### Example 3: TERMINATE Boundary

```python
detector = KeywordAssertionDetector(language="en")
text = "No fever but has persistent cough"

status, details = detector.detect(text)
# status: AssertionStatus.NEGATED
# details: {'keyword_negated_scopes': ['no: fever']}
# Note: "cough" is NOT in scope due to TERMINATE
```

**Analysis:**
- Trigger: "no" (FORWARD direction)
- TERMINATE detected: "but" at position 9
- Scope: "fever" (stops at "but", excludes "has persistent cough")
- Result: Only "fever" is negated; "cough" remains affirmed

### Example 4: PSEUDO Prevention

```python
detector = KeywordAssertionDetector(language="en")
text = "Not only fever but also cough"

status, details = detector.detect(text)
# status: AssertionStatus.AFFIRMED
# details: {'keyword_negated_scopes': []}
```

**Analysis:**
- PSEUDO match: "not only" detected
- Prevents: "not" from triggering negation
- Result: No negation detected, both concepts AFFIRMED

### Example 5: Multilingual (German, issue #79)

```python
detector = KeywordAssertionDetector(language="de")
text = "Ausschluss von Krampfanfällen"

status, details = detector.detect(text)
# status: AssertionStatus.NEGATED
# details: {'keyword_negated_scopes': ['Ausschluss von: Krampfanfällen']}
```

**Analysis:**
- Trigger: "Ausschluss von" (FORWARD direction)
- Scope: "Krampfanfällen" (seizures)
- Result: NEGATED
- **Resolves:** Issue #79 - missing German negation term

## Configuration

### Keyword Detection Window

The detection window size controls how many words are examined for scope extraction:

```python
# In assertion_detection.py
KEYWORD_WINDOW = 7  # Extract up to 7 words for scope
```

**Impact:**
- Larger window = more context captured but potential for false scope expansion
- Smaller window = more precise but may miss relevant context

### Word Boundary Matching

ConText rules use strict word boundary matching to prevent substring false positives:

```python
def _is_cue_match(text_lower: str, cue_lower: str, index: int) -> bool:
    """Check if cue matches with proper word boundaries."""
    # Character BEFORE cue must be non-alphanumeric or start of string
    # Character AFTER cue must be non-alphanumeric or end of string
```

**Example:**
- ✅ "no fever" → "no" matches (space boundary)
- ❌ "Normal" → "no" does NOT match (substring)

## Language Support

### Supported Languages (122 Rules Total)

| Language | Rules | Special Features |
|----------|-------|------------------|
| **English** | 26 | Full ConText support |
| **German** | 26 | Issue #79 resolution, compound phrases |
| **Spanish** | 24 | Gender variations |
| **French** | 24 | BIDIRECTIONAL "ne...pas" |
| **Dutch** | 22 | Germanic structure |

### Adding New Languages

1. **Create ConText rules file:**
```bash
phentrieve/text_processing/default_lang_resources/context_rules_{lang}.json
```

2. **Follow medspaCy format:**
```json
{
  "context_rules": [
    {
      "literal": "trigger phrase",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {
        "source": "your-source",
        "language": "{lang}"
      }
    }
  ]
}
```

3. **Test with unit tests:**
```python
def test_new_language_negation():
    detector = KeywordAssertionDetector(language="new_lang")
    text = "negation phrase concept"
    status, details = detector.detect(text)
    assert status == AssertionStatus.NEGATED
```

### Language Fallback

If ConText rules are not available for a requested language, the detector falls back to English rules:

```python
# Automatic fallback chain:
{requested_language} → English → Disabled (warning logged)
```

## Implementation Details

### Thread Safety

ConText rules are loaded as **frozen dataclasses**, ensuring thread-safe concurrent access:

```python
@dataclass(frozen=True)
class ConTextRule:
    """Immutable ConText rule for thread safety."""
    literal: str
    category: TriggerCategory
    direction: Direction
    metadata: dict[str, Any] | None = None
```

### Caching Strategy

Rules are loaded once per language and cached for the lifetime of the detector instance:

```python
# In _load_context_rules()
# Rules loaded on first detection call
# Cached in detector instance for subsequent calls
```

### Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| Rule loading | O(n) rules | Once per detector instance |
| PSEUDO detection | O(n) rules × O(m) text length | Pass 1 |
| TERMINATE detection | O(n) rules × O(m) text length | Pass 2 |
| Negation detection | O(n) rules × O(m) text length | Pass 3 |
| Overlap checking | O(k) PSEUDO spans | Typically k < 5 |

**Overall:** Linear in number of rules and text length, optimized for clinical text (typically <500 words per chunk).

## Testing

### Test Coverage

- **28 unit tests** covering all features (16 original + 12 new direction/TERMINATE tests)
- **82% coverage** for assertion_detection.py module
- Test suites:
  - `TestKeywordAssertionDetector` - Basic keyword detection
  - `TestDependencyAssertionDetector` - spaCy dependency parsing
  - `TestConTextRule` - Rule validation
  - `TestParseContextRules` - JSON parsing
  - `TestDirectionAwareDetection` - FORWARD/BACKWARD/BIDIRECTIONAL/PSEUDO
  - `TestTerminateScopeHandling` - TERMINATE boundaries
  - `TestConTextIntegration` - Multilingual integration tests

### Running Tests

```bash
# All assertion detection tests
pytest tests/unit/core/test_assertion_detection.py -v

# Specific test class
pytest tests/unit/core/test_assertion_detection.py::TestDirectionAwareDetection -v

# With coverage report
pytest tests/unit/core/test_assertion_detection.py --cov=phentrieve.text_processing.assertion_detection
```

## References

### Academic & Technical

- **medspaCy ConText Algorithm:** [medspaCy Documentation](https://github.com/medspacy/medspacy)
- **Original NegEx Algorithm:** Harkema et al. (2009), "ConText: An algorithm for determining negation, experiencer, and temporal status in clinical reports"
- **Phentrieve Implementation:** Based on medspaCy format with direction-aware TERMINATE handling

### Issue Resolution

- **Issue #79:** "Missing German negation terms (Ausschluss variants)"
  - **Status:** ✅ Resolved
  - **Resolution:** Added 5 missing terms via ConText rules:
    - "Ausschluss"
    - "Ausschluss von"
    - "kann ausgeschlossen"
    - "wird ausgeschlossen"
    - "werden ausgeschlossen"
  - **Commit:** `feat: Add German ConText rules (26 rules) - fixes issue #79`

### Code Review

See `docs/refactoring/NEGATION-DETECTION-CODE-REVIEW.md` for detailed analysis of:
- Performance optimization (100-1000x speedup achieved)
- Code quality improvements
- Antipattern elimination
- Testing strategy

## Future Enhancements

### Planned Features

1. **Additional ConText Categories:**
   - `POSSIBLE_EXISTENCE`: Uncertainty detection
   - `HYPOTHETICAL`: Conditional statements
   - `HISTORICAL`: Past medical history
   - `FAMILY`: Family history attribution

2. **Cross-Sentence Scope:**
   - Detect negation scope spanning multiple sentences
   - Anaphora resolution ("The patient denies these symptoms")

3. **Nested Negations:**
   - Double negatives: "not ruled out" → affirmed
   - Complex nesting: "no evidence of absence of"

4. **Machine Learning Enhancement:**
   - Train scope boundary classifier on clinical corpora
   - Learn language-specific patterns

### Contributing

To add ConText rules for a new language or improve existing rules:

1. Fork the repository
2. Add/modify rules in `phentrieve/text_processing/default_lang_resources/`
3. Add corresponding unit tests
4. Submit pull request with examples demonstrating improvement

**Rule Quality Guidelines:**
- Include metadata with source attribution
- Test with real clinical text examples
- Document edge cases and limitations
- Ensure word boundary correctness

---

**Document Version:** 1.0 (2025-11-20)
**Phentrieve Version:** 0.1.0
**ConText Rules:** 122 (EN: 26, DE: 26, ES: 24, FR: 24, NL: 22)
