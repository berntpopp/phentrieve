# Scripts and Setup Files Audit

**Date:** 2025-11-14
**Purpose:** Review legacy scripts for relevance after tooling modernization
**Branch:** feature/phase-0-preparation

---

## 📋 Executive Summary

**Files Analyzed:** 4 (setup.py, lint.sh, docker-login.sh, setup_phentrieve.sh)
**Recommendation:**
- ❌ **REMOVE:** 2 files (setup.py, lint.sh) - Obsolete
- ✅ **KEEP:** 1 file (setup_phentrieve.sh) - Still useful
- ⚠️ **OPTIONAL:** 1 file (docker-login.sh) - Low priority

---

## 📊 File-by-File Analysis

### 1. setup.py ❌ **REMOVE - OBSOLETE**

**File:** `setup.py` (41 bytes)
**Last Modified:** May 11, 2025

**Content:**
```python
from setuptools import setup

setup()
```

**Purpose:**
- Legacy setuptools entry point
- Defers all configuration to setup.cfg or pyproject.toml

**Current System:**
- All metadata now in `pyproject.toml` under `[project]`
- Build backend: `setuptools.build_meta` (configured in pyproject.toml)
- No longer needed with modern PEP 621 configuration

**Why Obsolete:**
- PEP 621 (2020) introduced declarative metadata in pyproject.toml
- setuptools 61.0+ supports pyproject.toml natively
- This file serves NO purpose when all config is in pyproject.toml

**Impact of Removal:**
- ✅ No impact - all build/install functionality in pyproject.toml
- ✅ Cleaner project root
- ✅ Follows modern Python packaging best practices

**Replacement:**
- Already replaced by `pyproject.toml`
- No action needed beyond deletion

**Recommendation:** **DELETE**

**Command:**
```bash
rm setup.py
```

---

### 2. lint.sh ❌ **REMOVE - OBSOLETE**

**File:** `lint.sh` (471 bytes)
**Last Modified:** May 12, 2025

**Content:**
```bash
#!/bin/bash
# Helper script for linting Python code with Black in WSL environment
black --config "$SCRIPT_DIR/pyproject.toml" ./phentrieve ./api
```

**Purpose:**
- Format Python code with Black
- WSL-specific path handling

**Current System:**
- **Black → Ruff migration complete (Phase 1)** ✅
- Black completely removed from project
- Makefile provides: `make format` (uses Ruff)
- CI/CD uses Ruff in workflows

**Why Obsolete:**
- Black is no longer installed or configured
- Ruff replaced Black with 18.8x performance improvement
- Makefile targets are more maintainable than scattered shell scripts

**Makefile Replacement:**
```makefile
format: ## Format Python code with Ruff
	ruff format phentrieve/ api/ tests/

lint: ## Lint Python code with Ruff
	ruff check phentrieve/ api/ tests/

lint-fix: ## Lint and auto-fix Python code
	ruff check --fix phentrieve/ api/ tests/
```

**Impact of Removal:**
- ✅ No impact - functionality replaced
- ✅ Removes confusion about which tool to use
- ✅ Eliminates obsolete Black references

**Recommendation:** **DELETE**

**Command:**
```bash
rm lint.sh
```

---

### 3. docker-login.sh ⚠️ **OPTIONAL - LOW PRIORITY**

**File:** `docker-login.sh` (637 bytes)
**Last Modified:** May 12, 2025

**Content:**
```bash
#!/bin/bash
# Script to attempt Docker Hub login if needed
# This can help with image pull rate limits
if ! docker info 2>/dev/null | grep -q "Username"; then
  echo "You're not logged into Docker Hub. Pull rate limits may apply."
  # ... interactive login prompt
fi
```

**Purpose:**
- Interactive Docker Hub login
- Avoid Docker Hub rate limits (100 pulls/6h anonymous, 200/6h authenticated)

**Current System:**
- **GHCR migration complete (Phase 7)** ✅
- Primary registry: `ghcr.io/berntpopp/phentrieve`
- Docker Hub: Retained as fallback/backup

**Analysis:**

**Pros (Keep):**
- Still useful for local development if pulling public images
- Docker Hub rate limits still apply to base images (e.g., python:3.10)
- Simple, non-invasive helper script
- No dependencies or conflicts

**Cons (Remove):**
- Not referenced anywhere in CI/CD or documentation
- Not critical for project functionality
- GitHub Actions uses its own registry authentication
- Most developers know `docker login` command

**Usage Pattern:**
- Local development only
- Manual invocation (not automated)
- Helps with rate limits for:
  - Base image pulls (python:3.10, node:18, etc.)
  - Public Docker Hub images

**Current Relevance:** LOW
- GHCR is primary
- Only matters for base images
- Easy to recreate if needed

**Recommendation:** **KEEP for now, LOW PRIORITY**
- Provides user convenience
- Minimal maintenance burden
- Can remove in future cleanup if never used

**Alternative:** Add to documentation instead:
```markdown
## Avoiding Docker Hub Rate Limits
If you encounter rate limits when pulling base images:
```bash
docker login
```
```

**Decision:** **KEEP** (but consider documenting in README and removing later)

---

### 4. setup_phentrieve.sh ✅ **KEEP - STILL USEFUL**

**File:** `setup_phentrieve.sh` (11,329 bytes, 232 lines)
**Last Modified:** May 13, 2025

**Content:**
Comprehensive Docker deployment setup script with:
1. Dependency checks (Docker, Docker Compose)
2. Environment configuration (.env.docker)
3. Host directory structure creation
4. Docker network setup for Nginx Proxy Manager integration
5. HPO data preparation
6. Default model indexing
7. Detailed deployment instructions

**Purpose:**
- Automated first-time deployment setup
- Nginx Proxy Manager (NPM) integration
- Production environment preparation

**Current System:**
- Docker deployment still supported
- GHCR images available
- Makefile provides some Docker commands

**Analysis:**

**Pros (Keep):**
- ✅ Handles complex first-time setup
- ✅ NPM integration is non-trivial
- ✅ Data directory structure creation
- ✅ Useful for production deployments
- ✅ Comprehensive error handling
- ✅ No equivalent in Makefile

**Cons:**
- ⚠️ Not using Makefile commands (duplication)
- ⚠️ Could be simplified
- ⚠️ Not documented in CLAUDE.md or README

**Functionality NOT in Makefile:**
- NPM network integration
- Environment file creation from template
- Data directory structure setup
- Interactive configuration prompts
- Pre-flight checks and validation

**Functionality DUPLICATED:**
- Image building (`docker-compose build`)
- Could use `make docker-build` instead

**Recommendation:** **KEEP but MODERNIZE**

**Suggested Improvements:**

1. **Use Makefile where possible:**
   ```bash
   # Instead of:
   $COMPOSE_COMMAND -f docker-compose.yml --env-file .env.docker build phentrieve_api

   # Use:
   make docker-build
   ```

2. **Add to documentation:**
   - Add to CLAUDE.md under Docker section
   - Create deployment guide in docs/

3. **Future Enhancement:**
   - Consider creating `make setup-production` that calls this script
   - Or migrate functionality into Makefile

**Current Status:** **KEEP AS-IS**
- Still provides value
- No conflicts with modernized tooling
- Useful for deployment scenarios

**Future Action:**
- Document in CLAUDE.md
- Consider modernization in future (not urgent)

---

## 📦 Dependencies & References

### Where Scripts Were Referenced

**Checked Locations:**
- ✅ `.github/workflows/` - None reference these scripts
- ✅ `Makefile` - No references
- ✅ `CLAUDE.md` - No references
- ✅ `README.md` - No references
- ✅ `docker-compose*.yml` - No direct references

**Result:** None of the obsolete scripts are actively used

---

## 🎯 Summary of Actions

### Immediate Actions (No Risk)

1. **DELETE setup.py**
   ```bash
   rm setup.py
   ```
   - **Impact:** NONE
   - **Reason:** Obsolete, replaced by pyproject.toml
   - **Risk:** Zero

2. **DELETE lint.sh**
   ```bash
   rm lint.sh
   ```
   - **Impact:** NONE
   - **Reason:** Black removed, Ruff in Makefile
   - **Risk:** Zero

### Optional Actions

3. **KEEP docker-login.sh** (for now)
   - Add note to documentation
   - Consider removing in future if unused
   - **Risk:** Very low (harmless helper)

4. **KEEP setup_phentrieve.sh** (valuable)
   - Document in CLAUDE.md
   - Consider modernization later
   - **Risk:** Low (production deployment tool)

---

## 📋 Modernization Checklist

### What's Already Modern ✅
- ✅ Makefile with comprehensive targets
- ✅ pyproject.toml for all configuration
- ✅ Ruff for linting/formatting
- ✅ uv for package management
- ✅ GHCR for container images
- ✅ CI/CD with GitHub Actions

### What Could Be Better ⚠️
- ⚠️ setup_phentrieve.sh could use Makefile commands
- ⚠️ docker-login.sh not documented
- ⚠️ No deployment documentation in CLAUDE.md

---

## 🔄 Comparison: Old vs New

| Task | Old Method | New Method | Status |
|------|-----------|------------|--------|
| Format code | `./lint.sh` | `make format` | ✅ Use Makefile |
| Lint code | N/A | `make lint` | ✅ Use Makefile |
| Type check | N/A | `make typecheck-fast` | ✅ Use Makefile |
| Install deps | `pip install -e .` | `make install` or `uv sync` | ✅ Use Makefile/uv |
| Build package | `python setup.py build` | `uv build` | ✅ Use uv |
| Docker login | `./docker-login.sh` | `docker login` | ⚠️ Manual |
| Docker setup | `./setup_phentrieve.sh` | (no equivalent) | ✅ Keep script |

---

## 💡 Recommendations

### Priority 1: Remove Obsolete Files (NOW)
```bash
rm setup.py lint.sh
git add -u
git commit -m "chore: remove obsolete setup.py and lint.sh

- setup.py: Obsolete, all config in pyproject.toml (PEP 621)
- lint.sh: Black replaced by Ruff, use 'make format' instead
"
```

**Impact:** Zero risk, cleaner codebase

---

### Priority 2: Update CLAUDE.md (THIS WEEK)

Add Docker deployment section:
```markdown
### Docker Development
```bash
# Using Makefile (recommended)
make docker-build                                    # Build Docker images
make docker-up                                       # Start containers
make docker-down                                     # Stop containers

# First-time production setup
./setup_phentrieve.sh                                # Comprehensive setup script
# Handles: env config, directories, NPM integration, data prep
```
```

---

### Priority 3: Consider Modernization (FUTURE)

**Option A:** Enhance Makefile
```makefile
setup-production: ## Full production setup with NPM integration
	@echo "Running production setup..."
	./setup_phentrieve.sh
```

**Option B:** Migrate to Makefile
- Move setup_phentrieve.sh logic into Makefile targets
- Keep script as backup

**Recommendation:** Keep as separate script (complexity justifies it)

---

## 📚 Best Practices Applied

✅ **DRY (Don't Repeat Yourself)**
- Remove lint.sh (duplicates Makefile)
- setup_phentrieve.sh could improve here

✅ **KISS (Keep It Simple, Stupid)**
- Delete setup.py (unnecessary abstraction)
- Makefile is simpler than scattered scripts

✅ **Single Source of Truth**
- pyproject.toml for all package metadata
- Makefile for all dev commands
- CI/CD for all automation

✅ **Least Surprise Principle**
- Developers expect `make <target>`, not `./script.sh`
- Modern Python uses pyproject.toml, not setup.py

---

## 🎯 Final Decision Matrix

| File | Keep? | Reason | Action |
|------|-------|--------|--------|
| setup.py | ❌ NO | Obsolete (PEP 621) | **DELETE** |
| lint.sh | ❌ NO | Black removed (Phase 1) | **DELETE** |
| docker-login.sh | ⚠️ OPTIONAL | Low value, harmless | Keep or delete |
| setup_phentrieve.sh | ✅ YES | Valuable deployment tool | **Document** |

---

## 📝 Commit Plan

```bash
# 1. Remove obsolete files
rm setup.py lint.sh

# 2. Stage and commit
git add -u
git commit -m "chore: remove obsolete setup.py and lint.sh

setup.py:
- Obsolete legacy setuptools entry point
- All package metadata now in pyproject.toml (PEP 621)
- setuptools 61.0+ supports pyproject.toml natively
- No functional impact

lint.sh:
- Used Black for formatting (removed in Phase 1)
- Replaced by Ruff: 'make format' and 'make lint'
- Black fully removed from project
- No functional impact

Both files provide no value after tooling modernization.
Improves project cleanliness and reduces confusion.
"

# 3. Update IMPLEMENTATION-STATUS.md to note script cleanup
# 4. Optionally remove docker-login.sh if desired
# 5. Update CLAUDE.md with docker deployment section
```

---

## 🏁 Conclusion

**Obsolete Files Identified:** 2 (setup.py, lint.sh)
**Safe to Remove:** YES (100% confidence)
**Risk Level:** ZERO
**Impact:** Cleaner codebase, no functional changes

**Deployment Scripts:** 1 valuable (setup_phentrieve.sh)
**Action Required:** Document in CLAUDE.md

**Optional File:** 1 (docker-login.sh)
**Decision:** Keep for convenience or document alternative

---

**Status:** AUDIT COMPLETE ✅
**Ready for Action:** YES
**Confidence:** HIGH (thorough analysis, zero risk)

---

*Audit completed: 2025-11-14*
*Next: Execute removal of obsolete files*
