# Performance Baselines

**Date:** 2025-11-14
**Branch:** main
**Commit:** 9984f6f0ebc196c8415611b160940e9af9fe37a7
**Python Version:** 3.10.14

---

## Metrics

### Format Time (Black - Current Tool)

**Command:** `black phentrieve/ api/ tests/ --check`
**Time:** 5.978 seconds (real time)
**Files:** 71 files checked, all would be left unchanged
**Status:** ✅ All code already formatted with Black

**Detailed Output:**
```
All done! ✨ 🍰 ✨
71 files would be left unchanged.

real    0m5.978s
user    0m4.647s
sys     0m0.735s
```

### Format Time (Ruff - Comparison)

**Command:** `ruff format phentrieve/ api/ tests/ --check`
**Time:** 0.318 seconds (real time)
**Files:** 71 files checked, 17 would be reformatted, 54 already formatted
**Speed Improvement:** **18.8x faster than Black**

**Detailed Output:**
```
17 files would be reformatted, 54 files already formatted

real    0m0.318s
user    0m0.049s
sys     0m0.086s
```

**Note:** Ruff has minor formatting differences from Black in 17 files (acceptable).

### Python Dependencies

**Total packages:** 453
**File:** `plan/requirements-baseline.txt`

**Key Dependencies:**
- black==25.1.0 (current formatter)
- ruff==0.14.1 (already installed for comparison)
- Python 3.10.14

### Current Project Structure

**Python Packages:**
- `phentrieve/` - Core library (CLI, data processing, indexing, retrieval, text processing)
- `api/` - FastAPI backend
- `tests/` - Test suite

**Total Python Files:** 71

---

## Test Suite Baseline

**Command:** `pytest tests/`

*(To be measured)*

---

## CI/CD Baseline

**Platform:** GitHub Actions (if exists)
**File:** `.github/workflows/deploy-docs.yml`

*(To be measured if CI exists)*

---

## Observations

### Current Tooling
- ✅ Black is configured and working (line-length: 88, target: py39)
- ✅ All code is formatted with Black
- ✅ Both Black and Ruff are installed in current environment
- ✅ Python 3.10.14 environment active

### Ruff Pilot Results (Preview)
- ⚡ Ruff is **18.8x faster** than Black (0.318s vs 5.978s)
- 📝 17 files would need reformatting (minor style differences)
- ✅ Performance improvement is significant
- ✅ No errors or warnings

### Recommendation for Phase 1
- ✅ **Proceed with Ruff migration**
- ✅ Performance gain is substantial
- ✅ Formatting differences are acceptable (PEP 8 compliant)
- ✅ Risk is LOW (Ruff is mature and well-tested)

---

## Notes

- Baseline measurements taken on branch `feature/phase-0-preparation`
- All measurements on same system for consistency
- Ruff version 0.14.1 is current stable (Nov 2025)
- Black version 25.1.0 is current stable

---

**Status:** Baselines documented ✅
**Next Step:** Install uv and run pilot test (Step 3)
