# Test Priority Analysis (Refocused on Core/CLI)

## Current Coverage Status (ACTUAL)

### ✅ Excellent Core Modules (80-100%)
- `dense_retriever.py`: **98%**
- `embeddings.py`: **100%**
- `output_formatters.py`: **100%**
- `reranker.py`: **100%**
- `assertion_detection.py`: **84%**
- `chunkers.py`: **87%**
- `resource_loader.py`: **87%**
- `pipeline.py`: **77%**

### ⚠️ CLI Modules (Decent but needs work)
- `similarity_commands.py`: **64%**
- `query_commands.py`: **52%**
- `data_commands.py`: **36%**
- `index_commands.py`: **33%**
- `text_commands.py`: **9%** ← CRITICAL GAP (187/205 uncovered)
- `cli/utils.py`: **8%** ← CRITICAL GAP

### ❌ Critical Gaps in Core Orchestrators
- `query_orchestrator.py`: **8%** ← HIGHEST PRIORITY (258/279 uncovered)
- `hpo_extraction_orchestrator.py`: **10%** ← HIGH PRIORITY
- `text_attribution.py`: **14%**

### ❌ API Helpers (Low Priority)
- `api_helpers.py`: **0%**
- `output_formatters_new.py`: **0%**

## Recommended Priority Order

1. **query_orchestrator.py** (8% → 80%) - Core retrieval logic, 279 statements
2. **text_commands.py** (9% → 70%) - Largest CLI module, 205 statements
3. **hpo_extraction_orchestrator.py** (10% → 70%) - Core extraction logic
4. **cli/utils.py** (8% → 60%) - Shared CLI utilities
5. **text_attribution.py** (14% → 70%) - Attribution tracking

## Why This Order?

- **query_orchestrator** = Core retrieval workflow (all queries go through this)
- **text_commands** = Largest CLI module, most user-facing
- **hpo_extraction** = Core extraction pipeline
- **cli/utils** = Shared by all CLI commands
- **text_attribution** = Improves result quality tracking
