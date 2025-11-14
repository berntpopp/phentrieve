# Rollback Scripts

This directory contains rollback scripts for reverting tooling modernization changes if issues are discovered.

---

## Available Scripts

### 1. `rollback-phase-1.sh` - Revert Ruff Migration

Reverts Phase 1 changes (Black → Ruff migration).

**When to use:**
- Ruff formatting causes unacceptable code changes
- Ruff linting rules are too strict or cause issues
- CI/CD pipelines fail due to Ruff
- Team prefers to continue using Black

**What it does:**
- ✅ Reformats all code with Black
- ✅ Ensures Black is installed
- ⚠️  Warns about manual steps (pyproject.toml, Makefile)

**Usage:**
```bash
cd /path/to/phentrieve
./scripts/rollback-phase-1.sh
```

**Manual steps required after rollback:**
1. Restore Black configuration in `pyproject.toml`
2. Revert Phase 1 commits if code quality issues exist
3. Update `Makefile` to use Black instead of Ruff
4. Run tests: `pytest tests/`
5. Verify formatting: `black phentrieve/ api/ tests/ --check`

---

### 2. `rollback-phase-2.sh` - Revert uv Migration

Reverts Phase 2 changes (pip → uv migration).

**When to use:**
- uv resolves incompatible package versions
- Package version conflicts cause runtime errors
- Application functionality breaks after migration
- Team prefers to continue using pip

**What it does:**
- ✅ Removes uv virtual environment (`.venv`)
- ✅ Removes uv lockfile (`uv.lock`)
- ✅ Creates new pip-based virtual environment
- ✅ Installs packages from baseline requirements
- ✅ Verifies package imports
- ⚠️  Warns about manual steps (pyproject.toml, Makefile, CI/CD)

**Usage:**
```bash
cd /path/to/phentrieve
./scripts/rollback-phase-2.sh
```

**Prerequisites:**
- `plan/requirements-baseline.txt` must exist (created in Phase 0, Step 2)

**Manual steps required after rollback:**
1. Update `Makefile` to use pip instead of uv
2. Update CI/CD workflows (`.github/workflows/`) to use pip
3. Remove `[tool.uv]` section from `pyproject.toml`
4. Restore original dependencies format in `pyproject.toml`
5. Run tests: `pytest tests/`
6. Verify CLI: `phentrieve --help`

---

## Rollback Decision Matrix

### Phase 1 (Ruff)

| Issue | Severity | Rollback? | Alternative |
|-------|----------|-----------|-------------|
| Formatting differences | Low | ❌ No | Accept differences (both PEP 8 compliant) |
| Linting too strict | Low | ❌ No | Disable specific rules in `pyproject.toml` |
| CI/CD failures | Medium | ⚠️ Maybe | Fix CI/CD config first, rollback if persistent |
| Code breaks | High | ✅ Yes | Immediate rollback |
| Team strongly objects | Medium | ⚠️ Maybe | Discuss and vote, rollback if unanimous |

### Phase 2 (uv)

| Issue | Severity | Rollback? | Alternative |
|-------|----------|-----------|-------------|
| Slower than pip | Low | ❌ No | uv is faster in 99% of cases |
| Version differences | Medium | ⚠️ Maybe | Pin versions in `pyproject.toml` first |
| Package conflicts | High | ✅ Yes | Rollback if cannot resolve |
| Runtime errors | High | ✅ Yes | Immediate rollback |
| Missing packages | High | ✅ Yes | Immediate rollback |
| Tests fail | High | ✅ Yes | Rollback if cannot fix quickly |

---

## Rollback Procedure

### Pre-Rollback Checklist

Before running rollback scripts:

1. **Document the issue**
   - What went wrong?
   - Error messages and logs
   - Steps to reproduce
   - Impact assessment

2. **Create issue backup**
   ```bash
   git branch backup-before-rollback-$(date +%Y%m%d)
   ```

3. **Notify team**
   - Inform team of rollback decision
   - Share issue documentation
   - Coordinate timing

4. **Verify baseline exists**
   - For Phase 2: Ensure `plan/requirements-baseline.txt` exists
   - For Phase 1: Ensure Black config is available

### Post-Rollback Checklist

After running rollback scripts:

1. **Complete manual steps** (see script output)
2. **Run full test suite**
   ```bash
   pytest tests/ -v
   ```
3. **Verify functionality**
   - Test CLI commands
   - Test API endpoints (if applicable)
   - Test frontend (if applicable)
4. **Update documentation**
   - Document rollback in project history
   - Update CHANGELOG if exists
5. **Create retrospective**
   - What went wrong?
   - How to prevent in future?
   - Lessons learned

---

## Testing Rollback Scripts

To test rollback scripts without actually rolling back:

```bash
# Dry-run test (will prompt for confirmation - answer 'N')
./scripts/rollback-phase-1.sh
# Answer 'N' when prompted

# Or use a test branch
git checkout -b test-rollback
./scripts/rollback-phase-1.sh
# Answer 'Y' to test full rollback
git checkout main
git branch -D test-rollback
```

---

## Rollback vs. Fix Forward

### When to Rollback

- ✅ Critical production issues
- ✅ Cannot identify root cause quickly
- ✅ Affects multiple team members
- ✅ No clear fix within 1-2 hours
- ✅ Risk to project timeline

### When to Fix Forward

- ✅ Minor issues (linting, formatting)
- ✅ Clear root cause identified
- ✅ Fix can be implemented quickly
- ✅ Issue affects only development (not production)
- ✅ Team has bandwidth to fix

---

## Emergency Rollback

For critical production issues:

```bash
# Phase 1 (Ruff) - Quick rollback
./scripts/rollback-phase-1.sh  # Answer 'Y'
black phentrieve/ api/ tests/
pytest tests/
git commit -am "Emergency rollback: Phase 1"
git push

# Phase 2 (uv) - Quick rollback
./scripts/rollback-phase-2.sh  # Answer 'Y'
pytest tests/
phentrieve --help  # Verify CLI
git commit -am "Emergency rollback: Phase 2"
git push
```

---

## Maintenance

### Updating Rollback Scripts

If project structure changes:

1. Update script paths in `rollback-phase-*.sh`
2. Test scripts in test branch
3. Document changes in this README
4. Update Phase plans in `plan/` directory

### Deprecating Rollback Scripts

Once a phase is considered stable (e.g., 3+ months without issues):

1. Move scripts to `scripts/archive/`
2. Update this README to reflect deprecation
3. Keep for historical reference

---

## Support

If rollback scripts fail or need assistance:

1. Check script output for error messages
2. Review `plan/MASTER-PLAN.md` for context
3. Consult phase-specific plans (`plan/phase-*.md`)
4. Create GitHub issue with:
   - Script output
   - Error messages
   - System information (`python --version`, `uv --version`, etc.)

---

## Script Maintenance Status

| Script | Status | Last Updated | Tested |
|--------|--------|--------------|--------|
| `rollback-phase-1.sh` | ✅ Active | 2025-11-14 | ✅ Yes |
| `rollback-phase-2.sh` | ✅ Active | 2025-11-14 | ✅ Yes |

---

**Note:** These scripts are safety nets. The goal is to never need them, but they provide confidence during migrations.
