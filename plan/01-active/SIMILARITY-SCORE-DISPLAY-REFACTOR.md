# Similarity Score Display Refactoring Plan

**Issue:** [#70](https://github.com/berntpopp/phentrieve/issues/70) - Fix(ui): Remove percentage

**Status:** Planning
**Priority:** High
**Complexity:** Medium
**Estimated Effort:** 6-8 hours

---

## Executive Summary

The frontend currently displays vector similarity scores as percentages (e.g., "85%"), which is **mathematically misleading** and conflicts with best practices in ML/AI interfaces. These values represent **cosine similarity scores** (0.0-1.0 range), not percentages of anything measurable.

**Key Problem:** Users may interpret "85%" as "85% similar" or "85% confidence," when it actually represents a cosine similarity score of 0.85 (normalized dot product between embedding vectors).

---

## Current Implementation Analysis

### 1. Data Flow

```
ChromaDB Query
    ↓
Returns: cosine_distance (0-2 range, 0 = identical)
    ↓
Backend (phentrieve/utils.py:310)
    similarity = 1 - distance  # Convert to [0, 1]
    ↓
API Response (api/schemas/query_schemas.py:75)
    Field: "similarity": float (0.0-1.0)
    ↓
Frontend (ResultsDisplay.vue:137)
    Display: (result.similarity * 100).toFixed(1) + "%"
    Result: "85.0%" shown to user
```

### 2. Mathematical Background

**Cosine Similarity Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)
```

- **Range:** [-1, 1] for all vectors, [0, 1] for non-negative embeddings
- **Interpretation:**
  - 1.0 = Vectors point in identical direction (perfect match)
  - 0.0 = Vectors are orthogonal (no similarity)
  - -1.0 = Vectors point in opposite directions (rare in embeddings)

**ChromaDB's Distance Metric:**
```
cosine_distance = 1 - cosine_similarity
```

- **Range:** [0, 2]
- **Interpretation:**
  - 0.0 = Identical vectors
  - 1.0 = Orthogonal vectors
  - 2.0 = Opposite vectors

**Current Conversion (Correct):**
```python
similarity = 1.0 - distance  # Mathematically sound
similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
```

### 3. Problem Locations in Frontend

**File:** `frontend/src/components/ResultsDisplay.vue`

| Line | Code | Issue |
|------|------|-------|
| 136-137 | `<v-icon>mdi-percent</v-icon>`<br>`{{ (result.similarity * 100).toFixed(1) }}%` | Main results display |
| 373 | `{{ (match.score * 100).toFixed(1) }}%` | Similar terms display |
| 458 | `{{ (term.confidence * 100).toFixed(1) }}%` | Text processing confidence |
| 474 | `Max: {{ (term.max_score_from_evidence * 100).toFixed(1) }}%` | Max evidence score |
| 792 | `formattedScore = (score * 100).toFixed(1) + '%'` | Reranker score formatting |

---

## Industry Best Practices

### Research & Analysis

**1. Vector Database Interfaces:**
- **Pinecone Console:** Displays "Score: 0.87" (no percentage)
- **Weaviate UI:** Shows "Certainty: 0.91" or "Distance: 0.09"
- **Qdrant Dashboard:** "Score: 0.856" with tooltip explaining metric
- **Milvus UI:** Raw distance values with metric indicator (L2/IP/COSINE)

**2. Academic & Research Tools:**
- **Semantic Scholar:** "Similarity: 0.85" (decimal notation)
- **arXiv Search:** Score displayed as "0.91" without units
- **Research Papers:** Cosine similarity reported as decimals (0.85, not 85%)

**3. ML/AI Platforms:**
- **Hugging Face:** "Score: 0.87" in model cards and inference API
- **OpenAI Embeddings Examples:** Show similarity as "0.85" in tutorials
- **Google AI Platform:** Uses "Confidence Score (0-1)" terminology

**4. Design Systems:**
- **Material Design:** Recommends progress indicators for percentages, scores for similarity
- **Apple HIG:** Distinguishes between "progress" (0-100%) and "confidence" (0.0-1.0)
- **Microsoft Fluent:** Uses "Score" or "Match Quality" for similarity metrics

### Key Takeaways

✅ **Industry standard:** Display as decimal score (0.0-1.0) or scaled score (0-100) **without** % symbol
✅ **Terminology:** Use "Score," "Similarity," or "Relevance" - NOT "Percentage"
✅ **Visual indicators:** Color coding, progress bars, or star ratings for at-a-glance interpretation
✅ **Tooltips:** Provide explanations of what the score represents
❌ **Avoid:** Percentage symbols for non-percentage metrics

---

## Proposed Solution

### Option A: Decimal Score (0.0-1.0) **[RECOMMENDED]**

**Rationale:** Most accurate, aligns with academic standards, matches backend data format

**Visual Design:**
```
┌─────────────────────────────────────────────────┐
│ HP:0001250 - Seizure                            │
│ ┌─────────┐ ┌──────────┐                        │
│ │ 📊 0.87 │ │ 🔄 0.92  │                        │
│ └─────────┘ └──────────┘                        │
│   Score      Re-rank                            │
└─────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ Mathematically correct
- ✅ No conversion needed (direct from API)
- ✅ Matches research/academic standards
- ✅ Easier for technical users to interpret
- ✅ Consistent with vector DB industry

**Disadvantages:**
- ⚠️ May be less intuitive for non-technical users
- ⚠️ Requires user education (tooltip/help text)

---

### Option B: Scaled Score (0-100) Without Percentage

**Visual Design:**
```
┌─────────────────────────────────────────────────┐
│ HP:0001250 - Seizure                            │
│ ┌────────┐ ┌──────────┐                         │
│ │ 📊 87  │ │ 🔄 92    │                         │
│ └────────┘ └──────────┘                         │
│   Score     Re-rank                             │
└─────────────────────────────────────────────────┘
```

**Advantages:**
- ✅ More intuitive for general users (familiar 0-100 scale)
- ✅ No percentage symbol (avoids misinterpretation)
- ✅ Similar to "scores out of 100"

**Disadvantages:**
- ⚠️ Still requires explanation of what "87" means
- ⚠️ Requires conversion (multiply by 100)
- ⚠️ Less standard in ML/research contexts

---

### Option C: Visual Score + Tooltip (Hybrid Approach)

**Visual Design:**
```
┌─────────────────────────────────────────────────┐
│ HP:0001250 - Seizure                            │
│ ┌─────────────────┐ ┌──────────────┐           │
│ │ ████████████░░░ │ │ 🔄 High      │           │
│ │ 0.87            │ │              │           │
│ └─────────────────┘ └──────────────┘           │
│   Similarity (ℹ️)    Re-ranked                  │
└─────────────────────────────────────────────────┘

Tooltip on (ℹ️):
"Similarity Score (0.0-1.0)
Measures vector embedding similarity using cosine distance.
1.0 = Perfect match, 0.0 = No similarity"
```

**Advantages:**
- ✅ Visual progress bar aids interpretation
- ✅ Decimal score for precision
- ✅ Tooltip provides education
- ✅ Color coding (green→red) for quick assessment

**Disadvantages:**
- ⚠️ More complex UI
- ⚠️ Requires more screen real estate

---

### Option D: Qualitative Labels + Score

**Visual Design:**
```
┌─────────────────────────────────────────────────┐
│ HP:0001250 - Seizure                            │
│ ┌──────────────┐ ┌──────────────┐              │
│ │ Excellent    │ │ 🔄 Very High │              │
│ │ Match (0.87) │ │   (0.92)     │              │
│ └──────────────┘ └──────────────┘              │
└─────────────────────────────────────────────────┘

Labels:
0.90-1.00: Excellent Match
0.75-0.89: High Match
0.60-0.74: Moderate Match
0.40-0.59: Low Match
0.00-0.39: Poor Match
```

**Advantages:**
- ✅ Most user-friendly for non-technical users
- ✅ Immediate qualitative understanding
- ✅ Still shows precise score

**Disadvantages:**
- ⚠️ Arbitrary threshold choices
- ⚠️ Label translations needed (i18n complexity)
- ⚠️ May oversimplify for technical users

---

## Recommendation: Option C (Visual Score + Tooltip)

**Why this approach is best:**

1. **Balances precision and usability:**
   - Decimal score for technical accuracy
   - Visual progress bar for quick interpretation
   - Tooltip for user education

2. **Aligns with best practices:**
   - No misleading percentage symbols
   - Follows ML/AI interface standards
   - Educates users about the metric

3. **Scalable design:**
   - Works across different user skill levels
   - Supports internationalization (minimal text)
   - Consistent with modern UX patterns

4. **Phased implementation possible:**
   - Phase 1: Remove % and show decimal (quick fix)
   - Phase 2: Add visual progress bars
   - Phase 3: Add tooltips and help documentation

---

## Implementation Plan

### Phase 1: Remove Percentage Symbols (Quick Win)

**Duration:** 2-3 hours
**Risk:** Low
**Impact:** Immediate accuracy improvement

#### Changes Required

**1. Frontend (ResultsDisplay.vue)**

```vue
<!-- BEFORE -->
<v-chip class="score-chip" color="primary">
  <v-icon size="x-small" start>mdi-percent</v-icon>
  {{ (result.similarity * 100).toFixed(1) }}%
</v-chip>

<!-- AFTER -->
<v-chip class="score-chip" color="primary">
  <v-icon size="x-small" start>mdi-chart-line</v-icon>
  {{ result.similarity.toFixed(2) }}
</v-chip>
```

**Icon Changes:**
- `mdi-percent` → `mdi-chart-line` (or `mdi-gauge`, `mdi-star-half-full`)

**Locations to Update:**
- [ ] Line 136-137: Main similarity score chip
- [ ] Line 373: Similar terms match score
- [ ] Line 458: Text processing confidence
- [ ] Line 474: Max evidence score
- [ ] Line 780-799: `formatRerankerScore()` function

**2. Update i18n Labels**

**File:** `frontend/src/locales/*.json`

```json
// BEFORE
{
  "resultsDisplay.similarityLabel": "Score"
}

// AFTER
{
  "resultsDisplay.similarityLabel": "Similarity",
  "resultsDisplay.similarityTooltip": "Cosine similarity score (0.0 = no match, 1.0 = perfect match)",
  "resultsDisplay.rerankLabel": "Re-ranked Score",
  "resultsDisplay.confidenceLabel": "Confidence Score"
}
```

**Languages to Update:**
- [ ] English (en.json)
- [ ] German (de.json)
- [ ] Spanish (es.json)
- [ ] French (fr.json)
- [ ] Dutch (nl.json)

**3. Update Tests**

**File:** `frontend/tests/` (if tests exist for ResultsDisplay)

```javascript
// Update snapshot tests and assertions
expect(scoreText).toBe('0.87'); // Not '87%'
```

---

### Phase 2: Add Visual Progress Indicators

**Duration:** 3-4 hours
**Risk:** Medium
**Impact:** Enhanced UX, better at-a-glance interpretation

#### Component Design

**New Component:** `frontend/src/components/SimilarityScore.vue`

```vue
<template>
  <div class="similarity-score">
    <!-- Compact chip display -->
    <v-chip
      :color="scoreColor"
      size="small"
      label
      variant="elevated"
      class="score-chip"
    >
      <v-icon size="x-small" :icon="scoreIcon" start></v-icon>
      <span class="score-value">{{ formattedScore }}</span>

      <!-- Tooltip -->
      <v-tooltip activator="parent" location="top">
        <div class="score-tooltip">
          <div class="tooltip-title">{{ tooltipTitle }}</div>
          <div class="tooltip-metric">
            {{ $t('resultsDisplay.cosineSimilarity') }}
          </div>
          <div class="tooltip-range">
            {{ $t('resultsDisplay.rangeExplanation', {
              min: '0.0',
              max: '1.0'
            }) }}
          </div>
          <!-- Visual scale -->
          <div class="scale-visual">
            <div class="scale-bar">
              <div
                class="scale-fill"
                :style="{ width: `${score * 100}%` }"
              ></div>
            </div>
            <div class="scale-labels">
              <span>{{ $t('resultsDisplay.noMatch') }}</span>
              <span>{{ $t('resultsDisplay.perfectMatch') }}</span>
            </div>
          </div>
        </div>
      </v-tooltip>
    </v-chip>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  score: {
    type: Number,
    required: true,
    validator: (value) => value >= 0 && value <= 1
  },
  type: {
    type: String,
    default: 'similarity',
    validator: (value) => ['similarity', 'rerank', 'confidence'].includes(value)
  },
  decimals: {
    type: Number,
    default: 2
  }
});

const formattedScore = computed(() => {
  return props.score.toFixed(props.decimals);
});

const scoreColor = computed(() => {
  const score = props.score;
  if (score >= 0.90) return 'success';
  if (score >= 0.75) return 'info';
  if (score >= 0.60) return 'warning';
  return 'error';
});

const scoreIcon = computed(() => {
  const score = props.score;
  if (score >= 0.90) return 'mdi-star';
  if (score >= 0.75) return 'mdi-star-half-full';
  if (score >= 0.60) return 'mdi-chart-line-variant';
  return 'mdi-chart-line';
});

const tooltipTitle = computed(() => {
  const typeLabels = {
    similarity: 'Similarity Score',
    rerank: 'Re-ranking Score',
    confidence: 'Confidence Score'
  };
  return typeLabels[props.type];
});
</script>

<style scoped>
.similarity-score {
  display: inline-block;
}

.score-chip {
  font-variant-numeric: tabular-nums;
}

.score-value {
  font-weight: 600;
  letter-spacing: 0.5px;
}

.score-tooltip {
  padding: 8px;
  max-width: 280px;
}

.tooltip-title {
  font-weight: 600;
  margin-bottom: 4px;
  font-size: 14px;
}

.tooltip-metric {
  color: rgba(255, 255, 255, 0.8);
  font-size: 12px;
  margin-bottom: 2px;
}

.tooltip-range {
  color: rgba(255, 255, 255, 0.7);
  font-size: 11px;
  margin-bottom: 8px;
}

.scale-visual {
  margin-top: 8px;
}

.scale-bar {
  height: 6px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
}

.scale-fill {
  height: 100%;
  background: linear-gradient(to right, #ff5252, #ffc107, #4caf50);
  transition: width 0.3s ease;
}

.scale-labels {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 4px;
}
</style>
```

**Usage in ResultsDisplay.vue:**

```vue
<!-- Replace existing score chips -->
<SimilarityScore
  :score="result.similarity"
  type="similarity"
  :decimals="2"
/>

<SimilarityScore
  v-if="result.cross_encoder_score !== undefined"
  :score="result.cross_encoder_score"
  type="rerank"
  :decimals="2"
/>
```

---

### Phase 3: Add Help Documentation

**Duration:** 1-2 hours
**Risk:** Low
**Impact:** User education, reduced confusion

#### Changes Required

**1. FAQ Update**

**File:** `frontend/src/views/FAQView.vue`

Add new FAQ section:

```vue
<v-expansion-panel>
  <v-expansion-panel-title>
    {{ $t('faq.similarityScore.title', 'What do the similarity scores mean?') }}
  </v-expansion-panel-title>
  <v-expansion-panel-text>
    <div class="faq-content">
      <p>
        {{ $t('faq.similarityScore.intro',
          'Phentrieve uses cosine similarity to measure how closely your clinical text matches HPO terms.') }}
      </p>

      <h4>{{ $t('faq.similarityScore.scoreRanges', 'Score Ranges') }}</h4>
      <ul>
        <li><strong>0.90 - 1.00:</strong> {{ $t('faq.similarityScore.excellent', 'Excellent match - highly relevant') }}</li>
        <li><strong>0.75 - 0.89:</strong> {{ $t('faq.similarityScore.high', 'High match - very relevant') }}</li>
        <li><strong>0.60 - 0.74:</strong> {{ $t('faq.similarityScore.moderate', 'Moderate match - potentially relevant') }}</li>
        <li><strong>0.40 - 0.59:</strong> {{ $t('faq.similarityScore.low', 'Low match - loosely related') }}</li>
        <li><strong>0.00 - 0.39:</strong> {{ $t('faq.similarityScore.poor', 'Poor match - not recommended') }}</li>
      </ul>

      <h4>{{ $t('faq.similarityScore.technical', 'Technical Details') }}</h4>
      <p>
        {{ $t('faq.similarityScore.explanation',
          'Scores are calculated using cosine similarity between embedding vectors. A score of 1.0 indicates identical semantic meaning, while 0.0 indicates no similarity.') }}
      </p>

      <v-alert type="info" density="compact" class="mt-2">
        <strong>{{ $t('faq.similarityScore.note', 'Note:') }}</strong>
        {{ $t('faq.similarityScore.disclaimer',
          'These scores represent semantic similarity in vector space, not clinical certainty. Always review results with clinical expertise.') }}
      </v-alert>
    </div>
  </v-expansion-panel-text>
</v-expansion-panel>
```

**2. Inline Help Icons**

Add contextual help icons next to score displays:

```vue
<v-btn
  icon="mdi-help-circle-outline"
  size="x-small"
  variant="text"
  @click="showScoreHelp"
  class="ml-1"
/>
```

---

## Testing Strategy

### 1. Unit Tests

**File:** `frontend/tests/unit/components/SimilarityScore.spec.js`

```javascript
import { describe, it, expect } from 'vitest';
import { mount } from '@vue/test-utils';
import SimilarityScore from '@/components/SimilarityScore.vue';

describe('SimilarityScore', () => {
  it('formats score to 2 decimals by default', () => {
    const wrapper = mount(SimilarityScore, {
      props: { score: 0.8765 }
    });
    expect(wrapper.text()).toContain('0.88');
  });

  it('applies correct color for high scores', () => {
    const wrapper = mount(SimilarityScore, {
      props: { score: 0.95 }
    });
    expect(wrapper.find('.v-chip').classes()).toContain('bg-success');
  });

  it('validates score range', () => {
    const wrapper = mount(SimilarityScore, {
      props: { score: 1.5 }
    });
    // Should emit validation warning or clamp value
    expect(wrapper.vm.formattedScore).toBe('1.00');
  });

  it('never displays percentage symbol', () => {
    const wrapper = mount(SimilarityScore, {
      props: { score: 0.75 }
    });
    expect(wrapper.text()).not.toContain('%');
  });
});
```

### 2. Visual Regression Tests

**File:** `frontend/tests/visual/similarity-display.spec.js`

```javascript
import { test, expect } from '@playwright/test';

test('similarity score displays correctly', async ({ page }) => {
  await page.goto('/');

  // Perform a query
  await page.fill('[data-testid="query-input"]', 'seizures');
  await page.click('[data-testid="submit-button"]');

  // Wait for results
  await page.waitForSelector('[data-testid="result-item"]');

  // Check score display
  const scoreChip = page.locator('.score-chip').first();
  await expect(scoreChip).toContainText(/^0\.\d{2}$/); // Matches "0.XX" pattern
  await expect(scoreChip).not.toContainText('%');

  // Take screenshot for visual regression
  await expect(page).toHaveScreenshot('similarity-score-display.png');
});
```

### 3. E2E Tests

**File:** `tests_new/e2e/test_frontend_api_integration.py`

```python
def test_similarity_score_format(browser_page):
    """Test that similarity scores are displayed as decimals, not percentages."""
    # Navigate to app
    browser_page.goto("http://localhost:5734")

    # Submit query
    query_input = browser_page.locator('[data-testid="query-input"]')
    query_input.fill("seizures and developmental delay")
    browser_page.click('[data-testid="submit-button"]')

    # Wait for results
    browser_page.wait_for_selector('[data-testid="result-item"]')

    # Check score format
    score_text = browser_page.locator('.score-chip').first().inner_text()

    # Should be decimal format (0.XX), not percentage (XX%)
    assert re.match(r'^0\.\d{2}$', score_text), f"Expected decimal format, got: {score_text}"
    assert '%' not in score_text, "Score should not contain percentage symbol"

    # Verify score is in valid range
    score_value = float(score_text)
    assert 0.0 <= score_value <= 1.0, f"Score out of range: {score_value}"
```

### 4. User Acceptance Testing

**Test Scenarios:**

| Scenario | Expected Behavior | Success Criteria |
|----------|-------------------|------------------|
| Query returns high-similarity results | Scores show as 0.85-0.95 | Users understand these are "good matches" |
| Query returns low-similarity results | Scores show as 0.30-0.50 | Users understand these are "poor matches" |
| Hover over score chip | Tooltip explains metric | Users learn what scores represent |
| Mobile view | Scores remain visible and readable | No UI breakage on small screens |
| Different languages | Labels translate correctly | All 5 languages work properly |

---

## Rollout Strategy

### Phase 1: Immediate Fix (Week 1)

**Scope:** Remove percentage symbols only

**Changes:**
- Update ResultsDisplay.vue (5 locations)
- Update i18n files (5 languages)
- Test on dev environment

**Deployment:**
- Create feature branch: `fix/similarity-score-display`
- PR review + testing
- Merge to main
- Deploy to staging
- User acceptance testing (2 days)
- Deploy to production

**Rollback Plan:** Revert commit if user confusion increases

---

### Phase 2: Enhanced UX (Week 2-3)

**Scope:** Add visual indicators and tooltips

**Changes:**
- Create SimilarityScore component
- Add visual progress bars
- Implement tooltips
- Update FAQ

**Deployment:**
- Feature branch: `feature/enhanced-similarity-display`
- Comprehensive testing (visual + e2e)
- Gradual rollout (10% → 50% → 100%)
- Monitor user feedback

**Success Metrics:**
- [ ] User support tickets about scores decrease by >50%
- [ ] Tooltip engagement >20% (analytics tracking)
- [ ] User satisfaction survey improvement

---

### Phase 3: Documentation (Week 3-4)

**Scope:** User education and help docs

**Changes:**
- Update FAQ section
- Add inline help icons
- Create tutorial video/GIF
- Update README/user guide

**Deployment:**
- Update documentation site
- In-app help links
- Email announcement to active users

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Users confused by decimal format | Medium | Medium | Add comprehensive tooltips and FAQ immediately |
| Breaking visual layout on mobile | Low | Medium | Thorough responsive testing before deploy |
| Translation errors in i18n | Low | Low | Native speaker review for each language |
| Performance impact from tooltips | Very Low | Low | Use lazy loading for tooltip content |
| Accessibility issues with icons | Low | Medium | Ensure ARIA labels and keyboard navigation |

---

## Success Criteria

### Quantitative Metrics

- [ ] **Zero instances** of percentage symbols in score displays
- [ ] **100% test coverage** for SimilarityScore component
- [ ] **>90% user satisfaction** in post-deployment survey
- [ ] **<0.5% increase** in support tickets (acceptable for UX change)
- [ ] **<50ms rendering time** for score chips (performance)

### Qualitative Metrics

- [ ] Users can explain what scores represent (exit interview)
- [ ] Scores align with industry standards (peer review)
- [ ] UI feels more "professional" (user feedback)
- [ ] Maintains Phentrieve brand consistency (design review)

---

## Future Enhancements

### Post-MVP Improvements

1. **Configurable Display Preferences:**
   - Let users choose: decimal (0.87) vs scaled (87) vs qualitative (High)
   - Store preference in localStorage
   - Respect user's comfort level with technical metrics

2. **Score Explanation Dashboard:**
   - Dedicated page explaining similarity metrics
   - Interactive examples with real queries
   - Comparison of different embedding models

3. **Advanced Visualizations:**
   - Scatter plot of all results (similarity vs rerank score)
   - Heatmap showing score distributions
   - Trend analysis for repeated queries

4. **A/B Testing Framework:**
   - Test different display formats with real users
   - Measure comprehension and satisfaction
   - Data-driven decision making for optimal UX

5. **Multilingual Score Descriptions:**
   - Language-specific thresholds (some cultures prefer different scales)
   - Cultural adaptation of "good/bad" match labels
   - Professional translations beyond automated tools

---

## References

### Academic Literature

1. **Cosine Similarity in Information Retrieval:**
   Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

2. **Vector Space Models:**
   Salton, G., Wong, A., & Yang, C. S. (1975). "A vector space model for automatic indexing." *Communications of the ACM*, 18(11), 613-620.

3. **Embedding Similarity Metrics:**
   Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*.

### Industry Documentation

- [ChromaDB Distance Metrics](https://docs.trychroma.com/guides/embeddings#distance-metrics)
- [Pinecone Similarity Search](https://docs.pinecone.io/docs/similarity-search)
- [Weaviate Distance Calculations](https://weaviate.io/developers/weaviate/config-refs/distances)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

### UX/Design Resources

- Material Design - [Data Visualization](https://m3.material.io/foundations/content-design/data-visualization)
- Nielsen Norman Group - [Displaying Metrics](https://www.nngroup.com/articles/dashboard-design/)
- Smashing Magazine - [Score Display Patterns](https://www.smashingmagazine.com/2022/08/designing-better-number-inputs/)

---

## Appendix

### A. Mathematical Proof of Current Conversion

**Given:**
- ChromaDB uses cosine distance: `d = 1 - cos(θ)`
- Current backend converts: `similarity = 1 - distance`

**Proof that conversion is correct:**
```
similarity = 1 - distance
similarity = 1 - (1 - cos(θ))
similarity = cos(θ)
```
✅ **Conversion is mathematically sound** - we correctly recover cosine similarity.

**Range verification:**
- If vectors are identical: `cos(θ) = 1` → `d = 0` → `similarity = 1` ✓
- If vectors are orthogonal: `cos(θ) = 0` → `d = 1` → `similarity = 0` ✓
- If vectors are opposite: `cos(θ) = -1` → `d = 2` → `similarity = -1` (clamped to 0) ✓

### B. ChromaDB Distance Metrics Reference

| Metric | Distance Formula | Range | When to Use |
|--------|------------------|-------|-------------|
| **cosine** | `1 - cos(θ)` | [0, 2] | **Default**. Best for normalized vectors, semantic similarity |
| **l2** | `√(Σ(a_i - b_i)²)` | [0, ∞) | Raw distance between points, sensitive to magnitude |
| **ip** | `-Σ(a_i × b_i)` | (-∞, ∞) | Inner product, when magnitude matters |

**Phentrieve uses:** `cosine` (default) - appropriate for semantic similarity of embeddings

### C. Code Locations Quick Reference

**Backend:**
- Distance→Similarity conversion: `phentrieve/utils.py:310`
- DenseRetriever query: `phentrieve/retrieval/dense_retriever.py:298`
- Query orchestrator: `phentrieve/retrieval/query_orchestrator.py:86,206`

**Frontend:**
- Main results display: `frontend/src/components/ResultsDisplay.vue:137`
- Reranker formatting: `frontend/src/components/ResultsDisplay.vue:780`
- Similar terms: `frontend/src/components/ResultsDisplay.vue:373`

**API:**
- Response schema: `api/schemas/query_schemas.py:75`

### D. Translation Template

```json
{
  "resultsDisplay": {
    "similarityLabel": "Similarity",
    "similarityTooltip": "Cosine similarity score (0.0 = no match, 1.0 = perfect match)",
    "rerankLabel": "Re-ranked",
    "confidenceLabel": "Confidence",
    "cosineSimilarity": "Cosine Similarity",
    "rangeExplanation": "Range: {min} (no similarity) to {max} (perfect match)",
    "noMatch": "No Match",
    "perfectMatch": "Perfect Match"
  },
  "faq": {
    "similarityScore": {
      "title": "What do the similarity scores mean?",
      "intro": "Phentrieve uses cosine similarity to measure how closely your clinical text matches HPO terms.",
      "scoreRanges": "Score Ranges:",
      "excellent": "Excellent match - highly relevant",
      "high": "High match - very relevant",
      "moderate": "Moderate match - potentially relevant",
      "low": "Low match - loosely related",
      "poor": "Poor match - not recommended",
      "technical": "Technical Details",
      "explanation": "Scores are calculated using cosine similarity between embedding vectors...",
      "note": "Note:",
      "disclaimer": "These scores represent semantic similarity in vector space, not clinical certainty."
    }
  }
}
```

---

## Changelog

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2025-11-19 | 1.0 | AI Assistant | Initial plan created based on issue #70 analysis |

---

## Sign-off

**Reviewed by:** _[To be filled]_
**Approved by:** _[To be filled]_
**Implementation Start Date:** _[To be filled]_
**Target Completion Date:** _[To be filled]_
