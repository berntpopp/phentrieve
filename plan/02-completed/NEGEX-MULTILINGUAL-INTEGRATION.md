# NegEx/ConText Multilingual Integration Plan

**Goal**: Integrate industry-standard NegEx/ConText trigger sets for all supported languages
**Approach**: Elegant, efficient, data-driven integration without major refactor
**Status**: Ready for Implementation
**Priority**: High (comprehensive multilingual support)
**Estimated Effort**: 4.5 hours (Phase 1: 2h, Phase 2: 2h, Phase 3: 1.5h, Testing: 1h)
**Coverage**: ~270+ triggers across 5 languages (16x improvement over current 17 triggers)

---

## Executive Summary

**Current State**: Phentrieve has custom German negation cues (17 terms) and partial implementation.

**Opportunity**: Leverage **medspaCy's ConText** framework with **validated multilingual trigger sets**:
- ✅ **English**: 53 ConText rules
- ✅ **French**: 44 ConText rules (validated)
- ✅ **Dutch**: 40+ ConText rules (validated, 2024)
- ✅ **German**: 86 NegEx-DE triggers (F1 > 0.9)
- ✅ **Spanish**: ~40 ConText rules (available)
- ⚠️ **Swedish**: Research available but needs integration

**Key Insight**: medspaCy uses **JSON format** for ConText rules that we can adopt!

**Integration Strategy**:
1. Convert our current `negation_cues.json` to medspaCy-compatible ConText JSON format
2. Import validated multilingual trigger sets from academic sources
3. Keep our existing code logic, just enhance the data
4. **No refactoring needed** - data-driven approach!

---

## Research Findings

### 1. medspaCy: Modern NegEx/ConText Framework

**What it is**: Clinical NLP library extending spaCy with ConText algorithm (NegEx generalization)

**Multilingual Support** (as of v1.3.1, November 2024):

| Language | ConText Rules | Status | Source |
|----------|---------------|--------|--------|
| **English (en)** | 53 rules | ✅ Production | Built-in |
| **French (fr)** | 44 rules | ✅ Validated | [FastConText 2016](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01656834/document) |
| **Dutch (nl)** | 40+ rules | ✅ Validated | [MedRxiv 2024](https://github.com/mi-erasmusmc/medspacy_dutch) |
| **German (de)** | 0 rules | ⚠️ Empty | Need to add NegEx-DE |
| **Spanish (es)** | ~20 rules | ⚠️ Minimal | QuickUMLS samples only |
| Portuguese (pt) | ~10 rules | ⚠️ Minimal | QuickUMLS samples |
| Italian (it) | ~10 rules | ⚠️ Minimal | QuickUMLS samples |
| Polish (pl) | 0 rules | ⚠️ Empty | No implementation |

**Architecture**: JSON-based rule files, language-agnostic framework

### 2. NegEx-DE: German Clinical Negation

**Source**: [MACSS DFKI Project](http://macss.dfki.de/german_trigger_set.html) (BioTxtM 2016)

**Performance**: F1-score > 0.9 on German clinical notes and discharge summaries

**Key Features**:
- 86 German negation triggers
- Handles inflections (kein, keine, keiner, keines, keinem, keinen)
- Handles discontinuous triggers ("kann ... ausgeschlossen werden")
- Categories: PRE (before), POST (after), PSEUDO (false alarms), POSP (pseudo-post), PREP (pseudo-pre)

**Advantages over our current system**:
- ✅ 86 triggers vs our 17 triggers (5x more coverage)
- ✅ Pseudo-negation handling (e.g., "nicht nur" = not negation)
- ✅ Position-aware rules (before vs after concept)
- ✅ Validated on real clinical data (F1 > 0.9)

### 3. Multilingual NegEx Lexicon (2013)

**Source**: Chapman et al., "Extending the NegEx Lexicon for Multiple Languages"

**Languages**: English, German, French, Swedish

**Format**: OWL/RDF (can be converted to JSON)

**Download**: GitHub repositories (rafaharo/negex, chapmanbe/negex)

**Status**: ⚠️ Files not directly accessible, may need extraction from papers

### 4. French ConText (FastConText 2016)

**Source**: [HAL-LIRMM](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01656834/document)

**Rules**: 44 ConText triggers for French clinical text

**Categories**:
- Negation
- Temporality (historical, current)
- Uncertainty
- Experiencer (patient, family)

**Integration**: Already converted to medspaCy JSON format

### 5. Dutch ConText (2024)

**Source**: [medspacy_dutch](https://github.com/mi-erasmusmc/medspacy_dutch)

**Rules**: 40+ ConText triggers for Dutch clinical text

**Performance**: High recall on Dutch clinical NLP tasks

**Format**: medspaCy-compatible JSON

---

## medspaCy ConText JSON Format

### Structure

```json
{
  "context_rules": [
    {
      "literal": "no evidence of",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "pattern": null,
      "metadata": {"comment": "Common negation phrase"}
    },
    {
      "literal": ";",
      "category": "TERMINATE",
      "direction": "TERMINATE"
    }
  ]
}
```

### Field Definitions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `literal` | string | Yes | Text to match (case-insensitive if no pattern) |
| `category` | string | Yes | Semantic category (see below) |
| `direction` | string | Yes | FORWARD, BACKWARD, BIDIRECTIONAL, TERMINATE, PSEUDO |
| `pattern` | list/null | No | spaCy token pattern for complex matching |
| `allowed_types` | list | No | Entity types that can be modified |
| `excluded_types` | list | No | Entity types that cannot be modified |
| `max_targets` | int | No | Maximum number of targets to modify |
| `max_scope` | int | No | Maximum scope in tokens (default: sentence) |
| `metadata` | dict | No | Extra info (comments, source, etc.) |

### Standard Categories

| Category | Meaning | Example |
|----------|---------|---------|
| `NEGATED_EXISTENCE` | Concept is negated | "no fever", "denies pain" |
| `POSSIBLE_EXISTENCE` | Concept is uncertain | "possible pneumonia", "maybe seizure" |
| `HISTORICAL` | Past condition | "history of diabetes", "previous MI" |
| `HYPOTHETICAL` | Future/conditional | "if symptoms worsen", "should develop" |
| `FAMILY` | Family member | "mother had cancer", "father with HTN" |
| `TERMINATE` | Ends scope | ";", "however", "but" |
| `PSEUDO` | False trigger | "not only", "no increase" |

### Direction Types

- **FORWARD**: Modifies concepts AFTER the trigger ("no [fever]")
- **BACKWARD**: Modifies concepts BEFORE the trigger ("[fever] is absent")
- **BIDIRECTIONAL**: Modifies concepts before AND after
- **TERMINATE**: Ends the scope of preceding modifiers (conjunctions)
- **PSEUDO**: False alarm, should NOT trigger negation

---

## Proposed Integration Approach

### Option A: Adopt medspaCy Format (RECOMMENDED) ✅

**Why**: Industry-standard, validated, extensible, community support

**Changes Needed**:
1. Convert `negation_cues.json` → `context_rules.json` (30 minutes)
2. Update `assertion_detection.py` to parse medspaCy format (1 hour)
3. Import validated trigger sets for each language (30 minutes)
4. Test and validate (1 hour)

**Benefits**:
- ✅ Access to validated multilingual trigger sets
- ✅ Industry-standard format (easier for contributors)
- ✅ Future-proof (can use medspaCy directly later if needed)
- ✅ Handles pseudo-negations, scope termination, directionality
- ✅ Easy to extend (just add JSON entries)

**Risks**:
- ⚠️ Need to update parsing logic (but simple JSON parsing)
- ⚠️ Need to test existing functionality (regression tests)

### Option B: Keep Current Format, Enhance Data

**Why**: Minimal code changes, just add more triggers

**Changes Needed**:
1. Add NegEx-DE triggers to `negation_cues.json` (30 minutes)
2. Add French/Dutch triggers to separate JSON files (30 minutes)
3. No code changes needed

**Benefits**:
- ✅ Zero refactoring
- ✅ Immediate improvement

**Drawbacks**:
- ❌ Misses pseudo-negation handling
- ❌ Doesn't handle directionality (pre vs post)
- ❌ Doesn't handle scope termination
- ❌ Non-standard format (harder for contributors)
- ❌ Won't benefit from future medspaCy improvements

### Option C: Hybrid Approach

**Why**: Get benefits of both - standard format + no major refactor

**Changes Needed**:
1. Create `context_rules.json` in medspaCy format (for new languages)
2. Keep `negation_cues.json` for backward compatibility
3. Update parser to handle BOTH formats
4. Gradually migrate existing rules

**Benefits**:
- ✅ No breaking changes
- ✅ Can adopt standard format incrementally
- ✅ Keeps existing code working

**Drawbacks**:
- ❌ Maintains two formats (complexity)
- ❌ Technical debt

---

## Recommended Implementation Plan

### Phase 1: Convert to medspaCy Format (2 hours)

**Goal**: Adopt industry-standard ConText JSON format

**Steps**:

1. **Create converter script** (30 minutes)
   ```python
   # scripts/convert_to_context_rules.py
   def convert_negation_cues_to_context_rules(negation_cues_json):
       """Convert old format to medspaCy ConText format."""
       context_rules = []
       for lang, cues in negation_cues_json.items():
           for cue in cues:
               rule = {
                   "literal": cue.strip(),
                   "category": "NEGATED_EXISTENCE",
                   "direction": "FORWARD",  # Default
                   "metadata": {"source": "legacy", "language": lang}
               }
               context_rules.append(rule)
       return {"context_rules": context_rules}
   ```

2. **Create new context rule files** (30 minutes)
   ```
   phentrieve/text_processing/default_lang_resources/
   ├── context_rules_en.json    # English ConText rules
   ├── context_rules_de.json    # German (NegEx-DE 86 triggers)
   ├── context_rules_fr.json    # French (FastConText 44 rules)
   ├── context_rules_nl.json    # Dutch (40+ rules)
   └── context_rules_es.json    # Spanish (expandable)
   ```

3. **Update assertion_detection.py parser** (1 hour)
   ```python
   def load_context_rules(language: str) -> list[ConTextRule]:
       """Load ConText rules for a language."""
       filename = f"context_rules_{language}.json"
       rules_data = load_language_resource(
           default_resource_filename=filename,
           config_key_for_custom_file="context_rules_file",
           language_resources_config_section=config.get("language_resources", {})
       )

       rules = []
       for rule_dict in rules_data.get("context_rules", []):
           rules.append(ConTextRule(**rule_dict))
       return rules

   @dataclass
   class ConTextRule:
       literal: str
       category: str
       direction: str
       pattern: Optional[list] = None
       allowed_types: Optional[list] = None
       excluded_types: Optional[list] = None
       max_targets: Optional[int] = None
       max_scope: Optional[int] = None
       metadata: Optional[dict] = None
   ```

4. **Test with existing functionality** (30 minutes)
   - Convert current negation_cues.json → context_rules_{lang}.json
   - Run all existing tests
   - Verify no regressions

### Phase 2: Import Validated Trigger Sets (2 hours)

**Goal**: Add comprehensive multilingual triggers for all supported languages

**Sources**:

1. **German (NegEx-DE)** - 86 triggers
   - Download from MACSS DFKI (if accessible) or extract from paper
   - Categories: PRE, POST, PSEUDO, POSP, PREP
   - Convert to medspaCy format

2. **Spanish (NegEx-MES + CONICET)** - 50-70 triggers
   - NegEx-MES: https://github.com/PlanTL-GOB-ES/NegEx-MES
   - CONICET CSV: https://ri.conicet.gov.ar/handle/11336/256470
   - Gender-aware triggers (masculino/feminino)
   - Categories: negPhrases, postNegPhrases, pseNegPhrases, conjunctions
   - Merge both sources for comprehensive coverage

3. **French (FastConText)** - 44 triggers
   - Available from medspaCy community discussions
   - Already in medspaCy format
   - Copy directly to `context_rules_fr.json`

4. **Dutch** - 40+ triggers
   - GitHub: https://github.com/mi-erasmusmc/medspacy_dutch
   - Already in medspaCy format
   - Copy directly to `context_rules_nl.json`

5. **English** - 53 triggers (baseline)
   - medspaCy built-in rules
   - Enhance our current English rules

**Example German NegEx-DE Rule**:
```json
{
  "literal": "kann ausgeschlossen werden",
  "category": "NEGATED_EXISTENCE",
  "direction": "BACKWARD",
  "metadata": {
    "source": "NegEx-DE",
    "category_original": "POST",
    "comment": "Discontinuous trigger"
  }
}
```

**Example Pseudo-Negation**:
```json
{
  "literal": "nicht nur",
  "category": "PSEUDO",
  "direction": "FORWARD",
  "metadata": {
    "comment": "False negation - means 'not only', affirms concept"
  }
}
```

### Phase 3: Enhance Detection Logic (1.5 hours)

**Goal**: Support directionality, pseudo-negations, scope termination

**Changes to `assertion_detection.py`**:

1. **Add direction support**
   ```python
   def apply_context_rule(rule: ConTextRule, text: str, concept_position: int):
       """Apply ConText rule based on direction."""
       if rule.direction == "FORWARD":
           # Check if concept is AFTER the trigger
           return concept_position > trigger_position
       elif rule.direction == "BACKWARD":
           # Check if concept is BEFORE the trigger
           return concept_position < trigger_position
       elif rule.direction == "BIDIRECTIONAL":
           # Concept can be anywhere in scope
           return True
       elif rule.direction == "TERMINATE":
           # End scope of previous modifiers
           return False
       elif rule.direction == "PSEUDO":
           # False alarm, don't negate
           return False
   ```

2. **Add scope termination**
   ```python
   # Detect terminators (conjunctions, punctuation)
   terminators = [rule for rule in context_rules if rule.category == "TERMINATE"]

   # When terminator found, stop applying previous modifiers
   if terminator_found:
       break  # Don't apply negation beyond terminator
   ```

3. **Add pseudo-negation handling**
   ```python
   # Check for pseudo-negations FIRST
   pseudo_rules = [rule for rule in context_rules if rule.category == "PSEUDO"]
   if any(pseudo_match(text, rule) for rule in pseudo_rules):
       return AssertionStatus.AFFIRMED  # Not actually negated!
   ```

---

## Implementation Checklist

### Phase 1: Convert to ConText Format (2 hours)

- [ ] Create `scripts/convert_to_context_rules.py` converter
- [ ] Convert `negation_cues.json` to `context_rules_{lang}.json` format
- [ ] Create `ConTextRule` dataclass in `assertion_detection.py`
- [ ] Update parser to load ConText rules from JSON
- [ ] Run existing tests to verify no regressions
- [ ] Update documentation

### Phase 2: Import Validated Trigger Sets (1 hour)

- [ ] **German**: Extract NegEx-DE 86 triggers and convert to JSON
  - [ ] PRE triggers (before concept)
  - [ ] POST triggers (after concept)
  - [ ] PSEUDO triggers (false alarms)
- [ ] **French**: Copy FastConText 44 rules from medspaCy community
- [ ] **Dutch**: Copy rules from medspacy_dutch GitHub repo
- [ ] **English**: Import medspaCy English ConText rules (53 rules)
- [ ] **Spanish**: Start with basic rules, mark for expansion

### Phase 3: Enhance Logic (1 hour)

- [ ] Add direction support (FORWARD, BACKWARD, BIDIRECTIONAL)
- [ ] Add scope termination (TERMINATE category)
- [ ] Add pseudo-negation handling (PSEUDO category)
- [ ] Add tests for new categories
- [ ] Benchmark performance (should be <10ms per chunk)

### Testing & Validation

- [ ] Unit tests for ConText rule parsing
- [ ] Unit tests for each direction type
- [ ] Unit tests for scope termination
- [ ] Unit tests for pseudo-negations
- [ ] Integration tests with real German/French/Dutch text
- [ ] Regression tests (all existing tests pass)
- [ ] Performance benchmarks

### Documentation

- [ ] Update README with ConText format explanation
- [ ] Update `text-processing-pipeline.md` with ConText details
- [ ] Add examples for each language
- [ ] Document how to add new triggers
- [ ] Document how to contribute language-specific rules

---

## Benefits of This Approach

### Immediate Benefits

1. ✅ **5x more German triggers** (17 → 86) with validation (F1 > 0.9)
2. ✅ **French support** with 44 validated triggers
3. ✅ **Dutch support** with 40+ validated triggers
4. ✅ **English enhancement** with 53 ConText rules
5. ✅ **Pseudo-negation handling** ("nicht nur" no longer incorrectly negates)
6. ✅ **Scope termination** (conjunctions properly end negation scope)
7. ✅ **Directionality** (pre vs post concept positioning)

### Long-term Benefits

1. ✅ **Industry-standard format** (easier for contributors)
2. ✅ **Community support** (leverage medspaCy ecosystem)
3. ✅ **Future-proof** (can integrate directly with medspaCy later)
4. ✅ **Extensible** (easy to add new languages)
5. ✅ **Maintainable** (clear structure, well-documented)
6. ✅ **Academic credibility** (using validated, published trigger sets)

---

## Performance Considerations

### Memory

- **Current**: ~2 KB per language (17 triggers)
- **After**: ~20 KB per language (86+ triggers)
- **Impact**: Negligible (total < 200 KB for all languages)

### Speed

- **Parsing JSON**: One-time cost at startup (~10ms)
- **Rule matching**: O(n * m) where n = triggers, m = text length
- **With 86 triggers**: Still < 5ms per chunk (acceptable)
- **Optimization**: Cache compiled regex patterns if needed

### Disk Space

- **Current**: 7 KB total (all language negation files)
- **After**: 50-100 KB total (all ConText rule files)
- **Impact**: Negligible

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Breaking existing functionality | High | Low | Comprehensive regression tests |
| Performance degradation | Medium | Low | Benchmark, optimize if needed |
| Format incompatibility | Medium | Low | Thorough testing with all languages |
| Missing trigger sets | Low | Medium | Start with Phase 1, add incrementally |
| Contributor confusion | Low | Medium | Clear documentation, examples |

---

## Alternative: Using medspaCy Directly

### Why NOT do this (for now)

**Pros**:
- ✅ Battle-tested library
- ✅ Full ConText implementation
- ✅ Active community

**Cons**:
- ❌ **Major refactor** (complete rewrite of assertion_detection.py)
- ❌ **Heavy dependency** (adds sciSpaCy, medspaCy, their dependencies)
- ❌ **Less control** (harder to customize)
- ❌ **Higher effort** (4-8 hours vs 2-4 hours)
- ❌ **May not integrate well** with our chunking strategy

**Decision**: **Adopt format, not library** (for now)
- Get 80% of benefits with 20% of effort
- Can migrate to library later if needed
- Keeps our codebase lightweight and customizable

---

## Success Criteria

### Must Have
- ✅ ConText JSON format adopted
- ✅ German: 86 NegEx-DE triggers integrated
- ✅ French: 44 FastConText rules integrated
- ✅ Dutch: 40+ rules integrated
- ✅ All existing tests pass (0 regressions)
- ✅ Documentation updated

### Should Have
- ✅ Pseudo-negation handling working
- ✅ Scope termination working
- ✅ Directionality support working
- ✅ Performance < 10ms per chunk

### Nice to Have
- 🔄 Direct medspaCy integration (future)
- 🔄 Spanish trigger set expansion
- 🔄 Swedish trigger set integration

---

## References

### Academic Papers
- Chapman et al. (2013) - "Extending the NegEx Lexicon for Multiple Languages"
- Cotik et al. (2016) - "Negation Detection in Clinical Reports Written in German" (NegEx-DE)
- FastConText (2016) - French ConText rules
- Dutch ConText (2024) - medRxiv paper on Dutch clinical NLP

### GitHub Repositories
- https://github.com/medspacy/medspacy - medspaCy main repository
- https://github.com/mi-erasmusmc/medspacy_dutch - Dutch ConText rules
- https://github.com/chapmanbe/negex - Original NegEx
- https://github.com/rafaharo/negex - NegEx multilingual lexicon

### Online Resources
- http://macss.dfki.de/german_trigger_set.html - NegEx-DE triggers (may require extraction)
- https://hal-lirmm.ccsd.cnrs.fr/lirmm-01656834/document - French FastConText paper
- medspaCy documentation - Context component guide

---

## Appendix A: Example ConText Rule Files

### `context_rules_de.json` (German, excerpt)

```json
{
  "context_rules": [
    {
      "literal": "kein",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {"source": "NegEx-DE", "type": "PRE"}
    },
    {
      "literal": "Ausschluss",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {"source": "NegEx-DE", "type": "PRE"}
    },
    {
      "literal": "kann ausgeschlossen werden",
      "category": "NEGATED_EXISTENCE",
      "direction": "BACKWARD",
      "metadata": {"source": "NegEx-DE", "type": "POST", "comment": "Discontinuous"}
    },
    {
      "literal": "nicht nur",
      "category": "PSEUDO",
      "direction": "FORWARD",
      "metadata": {"source": "NegEx-DE", "type": "PSEUDO", "comment": "False negation"}
    },
    {
      "literal": "aber",
      "category": "TERMINATE",
      "direction": "TERMINATE",
      "metadata": {"source": "NegEx-DE", "comment": "Conjunction terminator"}
    }
  ]
}
```

### `context_rules_fr.json` (French, excerpt)

```json
{
  "context_rules": [
    {
      "literal": "pas de",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {"source": "FastConText"}
    },
    {
      "literal": "sans",
      "category": "NEGATED_EXISTENCE",
      "direction": "FORWARD",
      "metadata": {"source": "FastConText"}
    },
    {
      "literal": "antécédent de",
      "category": "HISTORICAL",
      "direction": "FORWARD",
      "metadata": {"source": "FastConText", "comment": "History of"}
    },
    {
      "literal": "mais",
      "category": "TERMINATE",
      "direction": "TERMINATE",
      "metadata": {"source": "FastConText", "comment": "But"}
    }
  ]
}
```

---

## Appendix B: ConTextRule Class Implementation

```python
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class Direction(Enum):
    """Direction of modifier scope."""
    FORWARD = "FORWARD"          # Modifies concepts after trigger
    BACKWARD = "BACKWARD"        # Modifies concepts before trigger
    BIDIRECTIONAL = "BIDIRECTIONAL"  # Modifies concepts in both directions
    TERMINATE = "TERMINATE"      # Ends scope of previous modifiers
    PSEUDO = "PSEUDO"            # False alarm, not a real trigger

class Category(Enum):
    """Semantic category of modifier."""
    NEGATED_EXISTENCE = "NEGATED_EXISTENCE"
    POSSIBLE_EXISTENCE = "POSSIBLE_EXISTENCE"
    HISTORICAL = "HISTORICAL"
    HYPOTHETICAL = "HYPOTHETICAL"
    FAMILY = "FAMILY"
    TERMINATE = "TERMINATE"
    PSEUDO = "PSEUDO"

@dataclass
class ConTextRule:
    """
    ConText rule for assertion detection.

    Compatible with medspaCy ConText JSON format.
    """
    literal: str
    category: str
    direction: str
    pattern: Optional[List[Dict[str, Any]]] = None
    allowed_types: Optional[List[str]] = None
    excluded_types: Optional[List[str]] = None
    max_targets: Optional[int] = None
    max_scope: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize fields."""
        # Validate direction
        if self.direction not in [d.value for d in Direction]:
            raise ValueError(f"Invalid direction: {self.direction}")

        # Validate category
        if self.category not in [c.value for c in Category]:
            raise ValueError(f"Invalid category: {self.category}")

        # Normalize literal
        if not self.pattern:
            self.literal = self.literal.strip()

    def matches(self, text: str, position: int) -> bool:
        """Check if rule matches at given position in text."""
        if self.pattern:
            # Use spaCy pattern matching (future enhancement)
            raise NotImplementedError("Pattern matching not yet implemented")
        else:
            # Simple string matching
            text_lower = text.lower()
            literal_lower = self.literal.lower()

            # Check if literal appears at position
            if text_lower[position:position+len(literal_lower)] == literal_lower:
                # Verify word boundary
                if position == 0 or not text_lower[position-1].isalnum():
                    if (position + len(literal_lower) >= len(text) or
                        not text_lower[position+len(literal_lower)].isalnum()):
                        return True
            return False

    def applies_to_concept(self,
                           concept_position: int,
                           trigger_position: int,
                           text_length: int) -> bool:
        """Check if this rule applies to a concept at given position."""
        if self.direction == Direction.FORWARD.value:
            return concept_position > trigger_position
        elif self.direction == Direction.BACKWARD.value:
            return concept_position < trigger_position
        elif self.direction == Direction.BIDIRECTIONAL.value:
            return True
        elif self.direction in (Direction.TERMINATE.value, Direction.PSEUDO.value):
            return False
        return False

def load_context_rules_from_json(json_data: Dict[str, Any]) -> List[ConTextRule]:
    """Load ConText rules from JSON data."""
    rules = []
    for rule_dict in json_data.get("context_rules", []):
        rules.append(ConTextRule(**rule_dict))
    return rules
```

---

**Plan Status**: ✅ Ready for Implementation
**Next Step**: Get approval, then implement Phase 1
**Created**: 2025-11-20
**Author**: Claude Code
