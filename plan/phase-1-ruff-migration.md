# Phase 1: Ruff Migration (Black → Ruff)

**Duration:** Week 3-4 (after Phase 0 complete)
**Risk Level:** LOW
**Reversible:** 100%
**Branch:** `feature/phase-1-ruff`

---

## Objectives

1. Replace Black with Ruff for code formatting
2. Add Ruff linting capabilities
3. Update CI/CD to use Ruff
4. **NO other changes** - keep everything else the same

---

## Prerequisites

- [ ] Phase 0 completed and merged
- [ ] uv pilot test showed Ruff is acceptable
- [ ] Team has reviewed Phase 0 results
- [ ] Current branch is `main` and up to date

---

## Step-by-Step Implementation

### STEP 1: Create Feature Branch

```bash
cd /mnt/c/development/phentrieve
git checkout main
git pull origin main
git checkout -b feature/phase-1-ruff
```

**Validation:**
```bash
git branch --show-current  # Should output: feature/phase-1-ruff
git status  # Should be clean
```

---

### STEP 2: Update pyproject.toml - Remove Black, Add Ruff

#### 2.1: Read current pyproject.toml

```bash
cat pyproject.toml | grep -A 10 "\[tool.black\]"
```

**Expected:** You should see a `[tool.black]` section

#### 2.2: Edit pyproject.toml

**File:** `pyproject.toml`

**Find and REMOVE these lines** (entire [tool.black] section):
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
```

**ADD these lines** (at the end of file, before any existing tool sections):
```toml
# Ruff configuration
[tool.ruff]
line-length = 88
target-version = "py39"
fix = true

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "S",   # bandit security checks
]
ignore = [
    "E501",  # line too long (handled by formatter)
    "S101",  # use of assert (common in tests)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow asserts in tests
```

**Complete command using Edit tool:**
```bash
# Remove [tool.black] section
# Add [tool.ruff] configuration as shown above
```

#### 2.3: Update dependencies

**Find the dependencies section** in pyproject.toml

**In `[project.optional-dependencies]` or dev dependencies:**

**REMOVE:**
```toml
"black>=23.0.0",
```

**ADD:**
```toml
"ruff>=0.8.0",
```

**Validation:**
```bash
grep -n "black" pyproject.toml  # Should find NO matches
grep -n "ruff" pyproject.toml   # Should find ruff in dependencies and [tool.ruff]
```

---

### STEP 3: Install Ruff, Remove Black

```bash
# Uninstall Black
pip uninstall -y black

# Install dependencies with new pyproject.toml
pip install -e ".[all]"

# Verify Ruff installed
ruff --version  # Should show version >= 0.8.0
```

**Validation:**
```bash
which ruff  # Should show path to ruff
which black # Should show nothing or "not found"
```

---

### STEP 4: Format Entire Codebase with Ruff

```bash
# Format all Python files
ruff format phentrieve/ api/ tests/

# Check if any changes were made
git status
```

**Expected:** Ruff may reformat some files. This is normal.

**If changes detected:**
```bash
# Review changes
git diff

# If changes look good, stage them
git add phentrieve/ api/ tests/
git commit -m "style(phase-1): reformat code with ruff

Replace Black formatting with Ruff. Minor style differences expected."
```

---

### STEP 5: Run Ruff Linter

```bash
# Check for linting issues
ruff check phentrieve/ api/ tests/
```

**Expected output:** List of linting issues (if any)

**Common issues and fixes:**

**Issue: Unused imports**
```bash
ruff check phentrieve/ api/ tests/ --fix
```

**Issue: Import sorting**
Ruff will auto-fix most issues with `--fix` flag

**Manual review required issues:**
- Security issues (S*** codes)
- Complex refactoring suggestions

**Action:**
```bash
# Fix auto-fixable issues
ruff check phentrieve/ api/ tests/ --fix

# Review and commit
git add .
git commit -m "fix(phase-1): auto-fix ruff linting issues"
```

---

### STEP 6: Update CI/CD Configuration

**Note:** Check if you have existing CI/CD configuration

```bash
ls -la .github/workflows/
```

**If you have CI/CD files, update them:**

#### 6.1: Update GitHub Actions (if exists)

**File:** `.github/workflows/deploy-docs.yml` or similar

**Find and REPLACE:**

**OLD:**
```yaml
- name: Format check
  run: black --check .
```

**NEW:**
```yaml
- name: Format check with Ruff
  run: ruff format --check phentrieve/ api/ tests/

- name: Lint with Ruff
  run: ruff check phentrieve/ api/ tests/
```

**If no CI/CD exists:** Skip this step (we'll add comprehensive CI in Phase 9)

**Validation:**
```bash
grep -r "black" .github/  # Should find NO matches
grep -r "ruff" .github/   # Should find ruff commands
```

---

### STEP 7: Update Documentation

#### 7.1: Update CLAUDE.md

**File:** `CLAUDE.md`

**Find section about code formatting** (search for "black")

**REPLACE:**
```markdown
# Code formatting
black phentrieve/ api/ tests/
```

**WITH:**
```markdown
# Code formatting
ruff format phentrieve/ api/ tests/

# Code linting
ruff check phentrieve/ api/ tests/ --fix
```

#### 7.2: Update README.md (if it mentions black)

**File:** `README.md`

**Search for "black"** and replace with ruff commands

**Example:**
```bash
sed -i 's/black phentrieve/ruff format phentrieve/g' README.md
```

**Validation:**
```bash
grep -n "black" README.md CLAUDE.md  # Should find minimal or no matches
```

---

### STEP 8: Update .gitignore (if needed)

**File:** `.gitignore`

**Check if .ruff_cache needs to be added:**
```bash
grep "ruff_cache" .gitignore
```

**If not present, ADD:**
```
# Ruff cache
.ruff_cache/
```

---

### STEP 9: Create Ruff Configuration File (Optional but Recommended)

While we have configuration in pyproject.toml, a separate ruff.toml can be clearer:

**Create File:** `ruff.toml`

**Content:**
```toml
# Ruff configuration for Phentrieve
# See: https://docs.astral.sh/ruff/configuration/

target-version = "py39"
line-length = 88
fix = true

[lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "S",   # bandit security checks
]

ignore = [
    "E501",  # line too long (handled by formatter)
    "S101",  # use of assert (common in tests)
]

[lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow asserts in tests

[format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

**Note:** If you create ruff.toml, you can remove [tool.ruff] from pyproject.toml to avoid duplication (DRY principle).

**Validation:**
```bash
ruff check phentrieve/ api/ tests/  # Should use ruff.toml if present
```

---

### STEP 10: Run Full Test Suite

```bash
pytest tests/ -v
```

**Expected:** All tests should pass (formatting changes don't affect functionality)

**If tests fail:**
1. Review test failures
2. Check if Ruff changed test code behavior (unlikely)
3. Fix any legitimate issues
4. Re-run tests

**Validation:**
```bash
pytest tests/ --tb=short  # Should show all tests passing
```

---

### STEP 11: Performance Comparison

**Measure Ruff performance vs baseline:**

```bash
# Time ruff format
time ruff format phentrieve/ api/ tests/ --check

# Compare to baseline (from Phase 0)
cat plan/baselines.md | grep "Format Time"
```

**Record in:** `plan/phase-1-results.md`

```markdown
# Phase 1 Results

## Performance Comparison

**Black (baseline):**
- Time: [XX seconds]

**Ruff (new):**
- Time: [XX seconds]
- Speed improvement: [XX]x faster

## Linting Issues Found

- Total issues: [XX]
- Auto-fixed: [XX]
- Manual fixes: [XX]

## Changes Summary

- Files formatted: [XX]
- Import order changes: [XX]
- Other style changes: [XX]

## Issues Encountered

- [None / List any issues]

## Recommendation

✅ Phase 1 successful - proceed to Phase 2
```

---

### STEP 12: Commit All Changes

```bash
# Review all changes
git status
git diff --stat

# Add all modified files
git add .

# Final commit
git commit -m "feat(phase-1): complete ruff migration

- Replace Black with Ruff for formatting
- Add Ruff linting with security checks
- Update CI/CD configuration
- Update documentation
- All tests passing
- Performance: [XX]x faster than Black

Closes #[issue-number]"
```

---

### STEP 13: Push and Create Pull Request

```bash
# Push branch
git push origin feature/phase-1-ruff

# Create PR using GitHub CLI (or web interface)
gh pr create \
  --title "Phase 1: Migrate from Black to Ruff" \
  --body "$(cat <<'EOF'
## Phase 1: Ruff Migration

This PR replaces Black with Ruff for code formatting and adds linting capabilities.

### Changes

- ✅ Replaced Black with Ruff in pyproject.toml
- ✅ Reformatted entire codebase with Ruff
- ✅ Fixed auto-fixable linting issues
- ✅ Updated CI/CD configuration
- ✅ Updated documentation (CLAUDE.md, README.md)
- ✅ Added ruff.toml configuration
- ✅ All tests passing

### Performance

- **Format time:** [XX]s (was [YY]s with Black)
- **Speed improvement:** [ZZ]x faster
- **Lint time:** [XX]s (new capability)

### Linting Results

- Total issues found: [XX]
- Auto-fixed: [XX]
- Manual fixes: [XX]
- Remaining: [XX] (documented)

### Testing

```bash
# Format check
ruff format --check phentrieve/ api/ tests/

# Linting
ruff check phentrieve/ api/ tests/

# Tests
pytest tests/
```

### Breaking Changes

None - formatting changes only

### Validation Gate Checklist

- [ ] All tests pass
- [ ] CI/CD green
- [ ] Performance improved vs baseline
- [ ] Documentation updated
- [ ] Team reviewed changes
- [ ] Ready to merge

### Rollback Procedure

If issues arise:
```bash
./scripts/rollback/rollback-phase-1.sh
```

### Next Steps

After merge:
1. Monitor for 24-48 hours
2. Address any issues
3. Proceed to Phase 2: uv Migration

### Related

- See: `plan/MASTER-PLAN.md`
- See: `plan/phase-1-ruff-migration.md`
- Next: `plan/phase-2-uv-migration.md`
EOF
)" \
  --assignee @me
```

---

## Validation Gate Checklist

Before merging and proceeding to Phase 2:

- [ ] All steps completed successfully
- [ ] No errors in Ruff format check
- [ ] No critical linting issues (or all addressed)
- [ ] All tests pass (pytest)
- [ ] CI/CD pipeline green
- [ ] Performance metrics recorded (faster than Black)
- [ ] Documentation updated
- [ ] PR approved by 2+ reviewers
- [ ] Team comfortable with Ruff
- [ ] Rollback script tested (dry run)

**If ANY item fails:** Stop, investigate, fix, re-test

---

## Rollback Procedure

**Trigger Criteria:**
- Critical tests failing after merge
- Team unable to use Ruff effectively
- Formatting issues blocking development
- CI/CD broken for 2+ days

**Rollback Steps:**

```bash
cd /mnt/c/development/phentrieve

# Run rollback script
./scripts/rollback/rollback-phase-1.sh

# Or manual rollback:
git checkout main
git revert <phase-1-merge-commit>
pip uninstall -y ruff
pip install black
black phentrieve/ api/ tests/
git add .
git commit -m "revert: rollback phase-1 ruff migration"
git push origin main
```

**Post-Rollback:**
1. Document why rollback was needed
2. Address root cause
3. Update phase plan if needed
4. Retry when ready

**Expected Rollback Time:** 15-30 minutes

---

## Success Criteria

### Must Have
- [x] Ruff installed and working
- [x] Black removed
- [x] All files formatted with Ruff
- [x] All tests passing
- [x] CI/CD updated and green
- [x] Documentation updated

### Performance
- [x] Format time < 5 seconds (vs Black baseline)
- [x] Lint time < 10 seconds
- [x] No performance regression

### Quality
- [x] < 50 critical linting issues remaining
- [x] Security checks enabled (bandit rules)
- [x] Import sorting working (isort replacement)

### Team
- [x] Team understands Ruff commands
- [x] Team can fix basic linting issues
- [x] No major complaints about formatting changes

---

## Common Issues & Solutions

### Issue 1: Import order changes

**Problem:** Ruff reorders imports differently than isort/Black

**Solution:** This is expected and correct. Ruff follows PEP 8.

```bash
# Accept the changes
git add .
```

### Issue 2: Line length violations

**Problem:** Some lines > 88 characters

**Solution:** Ruff doesn't auto-wrap long lines in comments/strings

```bash
# Manually wrap long strings
# Or add to ignore list if justified
```

### Issue 3: Security warnings (S*** codes)

**Problem:** Ruff flags potential security issues

**Solution:** Review each case:
```bash
# See specific issue
ruff check phentrieve/file.py

# If false positive, add # noqa: S###
# If real issue, fix it!
```

### Issue 4: Tests fail after reformatting

**Problem:** Test assertions changed behavior

**Solution:** Very rare. Check:
```bash
git diff tests/
# Review changes carefully
# Fix test if needed
```

---

## Tips for Success

### Before Starting
1. ✅ Complete Phase 0 pilot test
2. ✅ Team has reviewed Ruff documentation
3. ✅ Communicate in Slack about starting

### During Execution
1. ⚡ Follow steps sequentially
2. ⚡ Run validation after each step
3. ⚡ Commit logical groups of changes
4. ⚡ Ask questions if unclear

### After Completion
1. 🎉 Announce in Slack
2. 🎉 Monitor for issues (24-48 hours)
3. 🎉 Document learnings
4. 🎉 Celebrate the win!

---

## Time Estimate

**Best Case:** 2-3 hours (if pilot test went smoothly)
**Expected:** 1 day (including testing and documentation)
**Worst Case:** 1 week (if issues found, need investigation)

**Breakdown:**
- Setup & install: 15 min
- Format & fix linting: 1-2 hours
- Update CI/CD: 30 min
- Testing: 1 hour
- Documentation: 30 min
- PR & review: 1-2 days

---

## Next Phase

**After successful merge and 24-48 hour monitoring:**

Request Phase 2 detailed plan:
> "Create detailed implementation plan for Phase 2: uv Migration"

Or read: `plan/phase-2-uv-migration.md`

---

## Notes for LLM Execution

- Follow steps in exact order
- Run validation after EACH step
- If validation fails, STOP and report
- Include actual output in responses
- Highlight any deviations from expected
- All file modifications include complete content
- All commands include expected output

---

**Phase 1 Complete!** ✅

Ready to proceed to Phase 2 after validation gate passes.
