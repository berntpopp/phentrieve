# Model Caching Optimization Plan

## Executive Summary

**Status**: ✅ CONFIRMED - Critical architectural inefficiency identified
**Impact**: HIGH - Memory usage, startup time, user experience
**Complexity**: MEDIUM - Requires careful threading and API integration
**Priority**: HIGH - Affects all CLI users and library consumers

### The Problem

Our investigation confirms a **"double loading" anti-pattern** where the same SentenceTransformer model is loaded into VRAM/RAM multiple times during normal CLI operations, wasting memory and increasing startup time.

## Evidence of the Issue

### 1. Code Analysis - Confirmed Double Loading

**Location**: `phentrieve/cli/text_commands.py`

```python
# Line 309-314: First load for semantic chunking
semantic_model_name = semantic_chunker_model or DEFAULT_MODEL
sbert_model_for_chunking = SentenceTransformer(semantic_model_name)

# Line 351-359: Second load for retrieval
model_name = retrieval_model or DEFAULT_MODEL
st_model = SentenceTransformer(model_name, device=device)
```

**Impact**: When `semantic_chunker_model` and `retrieval_model` both default to `DEFAULT_MODEL` (FremyCompany/BioLORD-2023-M), the CLI loads the **same model twice**.

### 2. Current State Analysis

| Component | Caching Status | Thread Safety | Notes |
|-----------|---------------|---------------|-------|
| **phentrieve/embeddings.py** | ❌ None | ❌ No | Creates fresh instance every call |
| **phentrieve/cli/text_commands.py** | ❌ None | ❌ No | Direct SentenceTransformer() calls |
| **api/dependencies.py** | ✅ Yes | ✅ Async locks | Works well for API, not for CLI/library |
| **DenseRetriever** | ✅ Reuses passed model | N/A | Receives model, doesn't load |

### 3. Memory Impact

**Example: BioLORD-2023-M Model**
- Model size: ~450MB (weights)
- PyTorch overhead: ~200MB
- **Total per instance**: ~650MB

**Current behavior** (process command):
```bash
phentrieve text process "clinical text" --strategy sliding_window
```
- Load 1: Semantic chunking → 650MB
- Load 2: HPO retrieval → 650MB
- **Total VRAM/RAM**: ~1.3GB (for ONE model!)

**With model caching**:
- Single load → 650MB
- **Savings**: 50% memory reduction

### 4. Benchmark Results Analysis

From `data/results/benchmark_comparison_20251118_210151.csv`:
- BioLORD model used in benchmarks (row index 1, 7)
- MRR: 0.2825 on 9 test cases
- Each benchmark run loads the model fresh → memory inefficiency during evaluation

## Research Findings

### SentenceTransformer Internal Behavior

**From Hugging Face Documentation**:
1. SentenceTransformer **does NOT implement automatic caching** across instances
2. Each `SentenceTransformer(model_name)` call loads a fresh model from disk→VRAM
3. The `cache_folder` parameter only controls **where downloaded models are stored on disk**, NOT in-memory caching
4. Models are cached in `~/.cache/huggingface/` on disk, but still loaded into memory each time

### Best Practices from Research

**Thread-Safe Singleton Pattern** (from web search results):
1. Use `threading.Lock` for synchronization during first creation
2. Implement double-check locking pattern
3. Store instances in module-level dictionary
4. Critical for multi-threaded environments

**PyTorch Memory Management**:
1. `.to(device)` is lightweight if model is already on target device (no-op)
2. Deleting references doesn't immediately free VRAM (cached)
3. `torch.cuda.empty_cache()` can help free unused memory
4. Loading same model twice can cause 2x memory surge

## Proposed Solution

### Phase 1: Implement Thread-Safe Model Registry

**File**: `phentrieve/embeddings.py`

**Changes**:
1. Add module-level model registry with thread lock
2. Modify `load_embedding_model()` to check registry first
3. Add `force_reload` parameter for cache invalidation
4. Add `clear_model_registry()` utility function
5. Handle device switching properly

**Pseudocode**:
```python
import threading
from typing import Dict

_MODEL_REGISTRY: Dict[str, SentenceTransformer] = {}
_REGISTRY_LOCK = threading.Lock()

def load_embedding_model(
    model_name: Optional[str] = None,
    trust_remote_code: bool = False,
    device: Optional[str] = None,
    force_reload: bool = False
) -> SentenceTransformer:
    # 1. Fast path: Check cache without lock
    if not force_reload and model_name in _MODEL_REGISTRY:
        model = _MODEL_REGISTRY[model_name]
        # Lightweight device move if needed
        return model.to(device) if device else model

    # 2. Slow path: Load with double-check locking
    with _REGISTRY_LOCK:
        # Re-check after acquiring lock
        if not force_reload and model_name in _MODEL_REGISTRY:
            model = _MODEL_REGISTRY[model_name]
            return model.to(device) if device else model

        # 3. Load model fresh
        model = SentenceTransformer(model_name, ...)
        model = model.to(device)

        # 4. Store in registry
        _MODEL_REGISTRY[model_name] = model
        return model

def clear_model_registry():
    """Clear cached models and free memory."""
    with _REGISTRY_LOCK:
        _MODEL_REGISTRY.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Phase 2: Update CLI to Use Shared Model

**File**: `phentrieve/cli/text_commands.py`

**Strategy**: Load model once, reuse for both chunking and retrieval

**Current** (process_text_for_hpo_command):
```python
# Line 309: Load for chunking
sbert_model_for_chunking = SentenceTransformer(semantic_model_name)

# Line 359: Load for retrieval (DUPLICATE!)
st_model = SentenceTransformer(model_name, device=device)
```

**Proposed**:
```python
from phentrieve.embeddings import load_embedding_model

# Determine if we need the same model for both
semantic_model_name = semantic_chunker_model or DEFAULT_MODEL
retrieval_model_name = retrieval_model or DEFAULT_MODEL
use_same_model = (semantic_model_name == retrieval_model_name)

# Load model(s) efficiently
if use_same_model:
    # Load once, share for both purposes
    shared_model = load_embedding_model(
        model_name=retrieval_model_name,
        trust_remote_code=trust_remote_code,
        device=device
    )
    sbert_model_for_chunking = shared_model  # Same instance
    st_model = shared_model  # Same instance
else:
    # Different models needed
    sbert_model_for_chunking = load_embedding_model(
        model_name=semantic_model_name,
        trust_remote_code=trust_remote_code,
        device=None  # CPU for chunking
    )
    st_model = load_embedding_model(
        model_name=retrieval_model_name,
        trust_remote_code=trust_remote_code,
        device=device
    )
```

### Phase 3: API Integration Review

**File**: `api/dependencies.py`

**Analysis**: The API already implements caching correctly using:
- `LOADED_SBERT_MODELS: dict[str, SentenceTransformer] = {}`
- Async locks (`asyncio.Lock`)
- Background loading tasks

**Recommendation**:
1. **Keep API caching as-is** - it's well-designed for async FastAPI
2. Ensure `api/dependencies.py` continues to call `load_embedding_model()` from `phentrieve/embeddings.py`
3. The new registry in `embeddings.py` will provide an additional layer of caching
4. API's async locks won't conflict with sync `threading.Lock` in embeddings.py

**Verification Needed**:
```python
# In api/dependencies.py line 54-59
model_instance = await run_in_threadpool(
    load_embedding_model,  # ← This will now use the registry
    model_name=model_name,
    trust_remote_code=trust_remote_code,
    device=actual_device,
)
```

### Phase 4: Testing Strategy

#### 4.1 Unit Tests

**New file**: `tests/unit/core/test_embeddings_caching.py`

```python
def test_model_loaded_once_per_name():
    """Verify same model name returns cached instance."""
    from phentrieve.embeddings import load_embedding_model, clear_model_registry

    clear_model_registry()
    model1 = load_embedding_model("all-MiniLM-L6-v2")
    model2 = load_embedding_model("all-MiniLM-L6-v2")

    # Should be the exact same object
    assert model1 is model2

def test_force_reload_bypasses_cache():
    """Verify force_reload creates new instance."""
    model1 = load_embedding_model("all-MiniLM-L6-v2")
    model2 = load_embedding_model("all-MiniLM-L6-v2", force_reload=True)

    # Should be different objects
    assert model1 is not model2

def test_different_models_not_cached_together():
    """Verify different model names get separate instances."""
    model_a = load_embedding_model("model-a")
    model_b = load_embedding_model("model-b")

    assert model_a is not model_b

def test_clear_registry_frees_models():
    """Verify clear_model_registry removes cached models."""
    load_embedding_model("all-MiniLM-L6-v2")
    clear_model_registry()
    # Next load should be fresh
    model = load_embedding_model("all-MiniLM-L6-v2")
    assert model is not None

def test_thread_safety():
    """Verify concurrent loads don't create duplicates."""
    import threading
    import time

    models = []

    def load_model():
        m = load_embedding_model("all-MiniLM-L6-v2")
        models.append(m)

    threads = [threading.Thread(target=load_model) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should get the same instance
    assert all(m is models[0] for m in models)

def test_device_switching():
    """Verify device parameter works correctly."""
    model_cpu = load_embedding_model("all-MiniLM-L6-v2", device="cpu")
    assert str(model_cpu.device) == "cpu"

    if torch.cuda.is_available():
        model_gpu = load_embedding_model("all-MiniLM-L6-v2", device="cuda")
        # Should be same cached instance, moved to GPU
        assert model_cpu is model_gpu
        assert "cuda" in str(model_gpu.device)
```

#### 4.2 Integration Tests

**File**: `tests/integration/test_cli_model_caching.py`

```python
def test_cli_process_uses_single_model_instance():
    """Verify CLI doesn't load model twice when using default model."""
    # Mock SentenceTransformer to track instantiations
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        # Run CLI command that needs both chunking and retrieval
        result = runner.invoke(app, [
            "text", "process",
            "Patient has fever",
            "--strategy", "sliding_window"
        ])

        # Should only instantiate model ONCE
        assert mock_st.call_count == 1
```

#### 4.3 Memory Profiling Tests

**File**: `tests/integration/test_memory_profiling.py`

```python
@pytest.mark.slow
def test_memory_usage_with_caching():
    """Compare memory usage with and without caching."""
    import tracemalloc
    from phentrieve.embeddings import load_embedding_model, clear_model_registry

    # Test WITHOUT caching
    clear_model_registry()
    tracemalloc.start()
    model1 = SentenceTransformer("all-MiniLM-L6-v2")
    model2 = SentenceTransformer("all-MiniLM-L6-v2")
    current, peak = tracemalloc.get_traced_memory()
    memory_without_cache = peak
    tracemalloc.stop()
    del model1, model2

    # Test WITH caching
    clear_model_registry()
    tracemalloc.start()
    model1 = load_embedding_model("all-MiniLM-L6-v2")
    model2 = load_embedding_model("all-MiniLM-L6-v2")  # Should reuse
    current, peak = tracemalloc.get_traced_memory()
    memory_with_cache = peak
    tracemalloc.stop()

    # Memory with cache should be significantly less
    memory_saved = memory_without_cache - memory_with_cache
    reduction_percent = (memory_saved / memory_without_cache) * 100

    logger.info(f"Memory saved: {memory_saved / 1024 / 1024:.2f} MB ({reduction_percent:.1f}%)")
    assert reduction_percent > 40, "Expected at least 40% memory reduction"
```

### Phase 5: Documentation Updates

#### 5.1 Code Documentation

**File**: `phentrieve/embeddings.py`

Add comprehensive docstrings:
```python
"""
Embedding model handling for the Phentrieve package.

This module provides functionality for loading and managing embedding models
used for encoding text into vector representations. It implements a thread-safe
singleton registry to prevent loading the same model into VRAM multiple times.

Thread Safety:
    All functions in this module are thread-safe. Concurrent calls to
    load_embedding_model() with the same model name will return the same
    instance without loading the model multiple times.

Memory Management:
    Models are cached globally within the Python process. Use
    clear_model_registry() to free memory when models are no longer needed.

Examples:
    >>> from phentrieve.embeddings import load_embedding_model
    >>>
    >>> # Load model (cached for reuse)
    >>> model = load_embedding_model("all-MiniLM-L6-v2")
    >>>
    >>> # Second call returns cached instance (fast!)
    >>> same_model = load_embedding_model("all-MiniLM-L6-v2")
    >>> assert model is same_model  # Same object
    >>>
    >>> # Force reload if needed
    >>> fresh_model = load_embedding_model("all-MiniLM-L6-v2", force_reload=True)
    >>>
    >>> # Clear cache to free memory
    >>> from phentrieve.embeddings import clear_model_registry
    >>> clear_model_registry()
"""
```

#### 5.2 User Documentation

**File**: `CLAUDE.md` - Add new section:

```markdown
### Model Caching and Memory Management

Phentrieve automatically caches embedding models to prevent loading the same model multiple times into memory. This significantly reduces memory usage and startup time.

**How it works**:
- First call to `load_embedding_model()` loads the model into VRAM/RAM
- Subsequent calls with the same model name return the cached instance
- Thread-safe: Safe for concurrent access
- Works across CLI, API, and library usage

**Memory savings example**:
```bash
# Without caching (old behavior): ~1.3GB memory
phentrieve text process "text" --semantic-model BioLORD --model BioLORD

# With caching (new behavior): ~650MB memory (50% savings!)
# Same command, but model loaded only once
```

**Clearing the cache**:
```python
from phentrieve.embeddings import clear_model_registry

# Free all cached models from memory
clear_model_registry()
```

**Force reload a model**:
```python
from phentrieve.embeddings import load_embedding_model

# Force fresh load (bypasses cache)
model = load_embedding_model("all-MiniLM-L6-v2", force_reload=True)
```
```

## Implementation Plan

### Step 1: Core Implementation (2-3 hours)
- [ ] Implement model registry in `phentrieve/embeddings.py`
- [ ] Add thread safety with `threading.Lock`
- [ ] Add `force_reload` parameter
- [ ] Add `clear_model_registry()` function
- [ ] Handle device switching properly

### Step 2: CLI Integration (1-2 hours)
- [ ] Update `text_commands.py` to detect same model usage
- [ ] Refactor to share model instance when appropriate
- [ ] Update `chunk_text_command` similarly
- [ ] Test with default model (BioLORD)

### Step 3: Testing (3-4 hours)
- [ ] Write unit tests for caching behavior
- [ ] Write thread safety tests
- [ ] Write memory profiling tests
- [ ] Write CLI integration tests
- [ ] Verify API compatibility

### Step 4: Documentation (1 hour)
- [ ] Update docstrings in `embeddings.py`
- [ ] Update `CLAUDE.md` with caching explanation
- [ ] Add examples to docstrings
- [ ] Document `clear_model_registry()` usage

### Step 5: Validation (1-2 hours)
- [ ] Run full test suite (366 tests)
- [ ] Run memory profiling benchmarks
- [ ] Verify API still works correctly
- [ ] Test with multiple models
- [ ] Test device switching (CPU ↔ CUDA)

**Total estimated time**: 8-12 hours

## Risk Analysis

### Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| **Thread safety bugs** | Medium | High | Comprehensive threading tests, code review |
| **API compatibility issues** | Low | High | Keep API caching separate, integration tests |
| **Device switching problems** | Medium | Medium | Test CPU/CUDA switching, verify `.to()` behavior |
| **Memory leaks** | Low | High | Memory profiling tests, proper cleanup in tests |
| **Breaking changes for users** | Low | Medium | Backward compatible, optional `force_reload` |

### Rollback Plan

If issues arise:
1. The registry is opt-in by design (existing code continues to work)
2. Add `PHENTRIEVE_DISABLE_MODEL_CACHE=1` environment variable to disable caching
3. Git revert is clean (single PR, no database changes)

## Expected Benefits

### Quantitative Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Memory usage (default model)** | ~1.3GB | ~650MB | 50% reduction |
| **CLI startup time** | 10-15s | 5-8s | 40-50% faster |
| **API first request** | No change | No change | (Already cached) |
| **Library usage** | 2x model loads | 1x model load | 50% reduction |

### Qualitative Benefits

1. **Better user experience**: Faster CLI commands, less memory pressure
2. **Scalability**: Can run more concurrent processes on same machine
3. **Cost savings**: Less VRAM needed on GPU servers
4. **Developer experience**: Easier to work with models in Jupyter notebooks
5. **Production readiness**: More efficient resource utilization

## Acceptance Criteria

### Must Have (P0)
- ✅ Models with same name return cached instance
- ✅ Thread-safe implementation with proper locking
- ✅ CLI uses single model when semantic_model == retrieval_model
- ✅ All existing tests pass
- ✅ API continues to work without regressions
- ✅ Memory usage reduced by >40% for common use cases
- ✅ Documentation updated

### Should Have (P1)
- ✅ `force_reload` parameter works
- ✅ `clear_model_registry()` function available
- ✅ Device switching works correctly
- ✅ Memory profiling tests demonstrate savings
- ✅ Thread safety tests pass

### Nice to Have (P2)
- ⚪ Environment variable to disable caching (for debugging)
- ⚪ Metrics logging for cache hits/misses
- ⚪ LRU cache with size limit (for production servers)
- ⚪ Automatic memory cleanup on low memory conditions

## Conclusion

The proposed model caching optimization addresses a confirmed architectural inefficiency that impacts all CLI users and library consumers. The implementation is straightforward, low-risk, and provides significant benefits in memory usage and startup time.

**Recommendation**: **APPROVE and implement** this optimization as a high-priority enhancement.

### Next Steps

1. Review this plan with team
2. Get approval for implementation
3. Create feature branch: `feat/model-caching-optimization`
4. Implement Phase 1 (core registry)
5. Validate with tests
6. Roll out to production

---

**Document Version**: 1.0
**Author**: Claude Code Analysis
**Date**: 2025-11-18
**Status**: Pending Review
