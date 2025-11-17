# CLI Testing Implementation Plan

## Analysis Summary

### utils.py (8% coverage - 90/98 uncovered)
**2 functions, both virtually untested:**

1. **`load_text_from_input()`** (lines 15-54)
   - Input sources: CLI arg, file, stdin
   - Error cases: missing file, empty text, no input
   - **Coverage: ~10%** (only basic path tested)

2. **`resolve_chunking_pipeline_config()`** (lines 57-227)
   - Config sources: JSON file, YAML file, strategy name
   - 7 strategy types to handle
   - **Coverage: 0%** (completely untested!)

### query_commands.py (52% coverage - 43/90 uncovered)
**Existing tests: 10 tests that mock orchestrate_query ✅**
**Missing coverage (lines 169-170, 184, 189-274, 321-324):**
- Interactive mode logic (lines 189-274)
- Help text generation (lines 169-170, 184, 321-324)

### benchmark_commands.py (20% coverage - 35/44 uncovered)
**Missing: Most benchmarking logic**

## Implementation Order (High Impact First)

### Phase 1: utils.py (HIGH IMPACT) ⭐
**Why first?** Used by ALL other CLI commands - foundational

#### 1.1: Test `load_text_from_input()` (30 min)
- ✅ Text from CLI argument
- ✅ Text from file
- ✅ Text from stdin
- ✅ Error: File not found
- ✅ Error: Empty text
- ✅ Error: No input provided

#### 1.2: Test `resolve_chunking_pipeline_config()` (1 hour)
- ✅ Load from JSON config file
- ✅ Load from YAML config file
- ✅ Strategy: simple, detailed, semantic
- ✅ Strategy: sliding_window variants (4 types)
- ✅ Error: Config file not found
- ✅ Error: Invalid config format
- ✅ Fallback to default config

**Expected Coverage Improvement: 8% → 80%**

### Phase 2: query_commands.py (MEDIUM IMPACT)
**Why second?** Finish what's started, highest user visibility

#### 2.1: Test interactive mode logic (45 min)
- Lines 189-274 (interactive setup, prompt loop)
- Mock user input, test exit conditions

**Expected Coverage Improvement: 52% → 75%**

### Phase 3: benchmark_commands.py (MEDIUM IMPACT)
**Why third?** Important for evaluation, less frequently used

#### 3.1: Test benchmark CLI commands (45 min)
- Mock benchmark orchestrator
- Test file I/O, result formatting

**Expected Coverage Improvement: 20% → 70%**

## Total Estimated Time: 3.5 hours
## Expected Overall CLI Coverage: 28% → 65%+
