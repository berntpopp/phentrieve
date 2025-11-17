# query_orchestrator.py Testing Plan

## Module Analysis (279 statements, 8% coverage)

### Functions Overview
1. **`convert_results_to_candidates()`** (lines 52-104, 53 statements)
   - Pure function: ChromaDB results → candidate format
   - ~50% uncovered (lines 68-104)
   
2. **`segment_text()`** (lines 107-127, 21 statements)
   - Pure function: Text → sentences
   - ~90% uncovered (lines 119-127)
   
3. **`format_results()`** (lines 130-308, 179 statements)
   - Pure function: Raw results → structured format
   - ~90% uncovered (lines 152-291)
   
4. **`_format_structured_results_to_text_display()`** (lines 309-352)
   - Private function: Structured → text display
   - 100% uncovered (lines 319-350)
   
5. **`process_query()`** (lines 353-643, 291 statements)
   - Complex: Query processing pipeline
   - ~90% uncovered (lines 387-641)
   
6. **`orchestrate_query()`** (lines 644-892, 249 statements)
   - Main orchestration: Model loading, interactive mode
   - ~90% uncovered (lines 701-892)

## Testing Strategy (Bottom-Up Approach)

### Phase 1: Test Pure Helper Functions (HIGH VALUE, LOW EFFORT) ⭐
**Estimated: 1 hour, +40% coverage**

#### 1.1: Test `convert_results_to_candidates()` (20 min)
- ✅ Valid ChromaDB results → candidates
- ✅ Cross-lingual mode (English docs)
- ✅ Monolingual mode (load translations)
- ✅ Empty results
- ✅ Missing translation fallback

#### 1.2: Test `segment_text()` (15 min)
- ✅ Basic sentence segmentation
- ✅ Multi-sentence text
- ✅ Language detection (ASCII → English)
- ✅ Non-ASCII text handling
- ✅ Edge cases (empty, single char)

#### 1.3: Test `format_results()` (25 min)
- ✅ Format valid results
- ✅ Filter by threshold
- ✅ Limit by max_results
- ✅ Handle reranked results
- ✅ Empty results
- ✅ Type coercion (max_results as string)
- ✅ Missing distances

**Expected Coverage: 8% → 50%**

### Phase 2: Test Complex Functions (MEDIUM VALUE, HIGH EFFORT)
**Estimated: 2 hours, +30% coverage**

#### 2.1: Test `process_query()` (1 hour)
- Mock: DenseRetriever, load_embedding_model
- Test query processing flow
- Test reranking integration
- Test assertion detection

#### 2.2: Test `orchestrate_query()` (1 hour)  
- Mock: Model loading, ChromaDB
- Test orchestration flow
- Test interactive mode setup
- Test error handling

**Expected Coverage: 50% → 80%**

### Phase 3: Private Function (Optional)
- `_format_structured_results_to_text_display()` is private
- Tested indirectly through public functions
- Skip for now unless needed for coverage

## Implementation Priority

**Start with Phase 1** - Pure functions are:
- Easy to test (no complex mocking)
- High value (cover 40% of module)
- Fast to implement (~1 hour)
- Foundation for Phase 2 tests

After Phase 1, reassess if Phase 2 is worth the effort or if we should move to other modules.
