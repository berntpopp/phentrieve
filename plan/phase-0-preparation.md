# Phase 0: Preparation & Validation

**Duration:** Week 1-2
**Risk Level:** NONE
**Reversible:** 100%
**Branch:** `feature/phase-0-preparation`

---

## Objectives

1. Establish performance baselines
2. Pilot test uv with subset of dependencies
3. Prepare team and infrastructure
4. Create rollback scripts
5. Document current state

---

## Prerequisites

- [ ] Team has reviewed MASTER-PLAN.md
- [ ] Stakeholders approved 20-25 week timeline
- [ ] Development environment available
- [ ] Git repository clean (`git status`)

---

## Step-by-Step Implementation

### STEP 1: Create Feature Branch

```bash
cd /mnt/c/development/phentrieve
git checkout main
git pull origin main
git checkout -b feature/phase-0-preparation
```

**Validation:**
```bash
git branch --show-current  # Should output: feature/phase-0-preparation
```

---

###STEP 2: Document Current Performance Baselines

#### 2.1: Measure Current Install Time

```bash
# Clear any existing env
rm -rf .venv __pycache__

# Time pip install
time pip install -e .
```

**Record output in:** `plan/baselines.md`

#### 2.2: Measure Current Format Time

```bash
# Time Black (if installed)
time black phentrieve/ api/ tests/ --check
```

**Record output in:** `plan/baselines.md`

#### 2.3: Measure Current CI Time

```bash
# Check last successful CI run
gh run list --limit 1 --json durationMs,conclusion
```

**Record output in:** `plan/baselines.md`

#### 2.4: Create Baselines Document

**Create File:** `plan/baselines.md`

**Content:**
```markdown
# Performance Baselines

**Date:** 2025-11-14
**Branch:** main
**Commit:** [insert commit SHA]

## Metrics

### Install Time
- **Command:** `pip install -e .`
- **Time:** [XXX seconds]
- **Python Version:** [X.X.X]

### Format Time (Black)
- **Command:** `black phentrieve/ api/ tests/ --check`
- **Time:** [XXX seconds]
- **Files:** [XXX files]

### CI/CD Time
- **Last Run:** [link to GitHub Actions]
- **Duration:** [XXX minutes]
- **Outcome:** [success/failure]

### Test Suite
- **Command:** `pytest tests/`
- **Duration:** [XXX seconds]
- **Tests:** [XXX passed]

## Current Dependency Versions

```
pip freeze > plan/requirements-baseline.txt
```

## Notes
- [Any observations]
```

**Execute:**
```bash
# Create baselines file
touch plan/baselines.md

# Save current dependencies
pip freeze > plan/requirements-baseline.txt

# Commit baseline
git add plan/baselines.md plan/requirements-baseline.txt
git commit -m "chore(phase-0): add performance baselines"
```

---

### STEP 3: Install and Test uv (Pilot)

#### 3.1: Install uv Globally

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version  # Should show version >= 0.5.0
```

**Validation:**
```bash
which uv  # Should show path to uv binary
uv --version  # Should output version number
```

#### 3.2: Create Test Environment with uv

**DO NOT modify existing pyproject.toml yet**

```bash
# Create test directory
mkdir -p /tmp/phentrieve-uv-test
cd /tmp/phentrieve-uv-test

# Copy current pyproject.toml
cp /mnt/c/development/phentrieve/pyproject.toml ./

# Initialize uv (this creates uv.lock)
uv init --no-readme --no-workspace

# Try to install dependencies
time uv sync
```

**Record:**
- Installation time
- Any errors or warnings
- Resolved versions

#### 3.3: Compare Dependency Versions

```bash
# Generate uv resolved versions
uv pip freeze > /tmp/uv-resolved-versions.txt

# Compare with baseline
diff /mnt/c/development/phentrieve/plan/requirements-baseline.txt /tmp/uv-resolved-versions.txt > /tmp/version-diff.txt
```

**Create Report:** `plan/uv-pilot-report.md`

```markdown
# uv Pilot Test Report

**Date:** 2025-11-14

## Installation Performance

- **uv sync time:** [XXX seconds]
- **pip install time (baseline):** [XXX seconds]
- **Speed improvement:** [XX]x faster

## Version Resolution Differences

\`\`\`diff
[paste contents of /tmp/version-diff.txt]
\`\`\`

## Issues Encountered

- [ ] Issue 1: [description]
- [ ] Issue 2: [description]

## Recommendation

- [ ] ✅ Proceed with uv migration
- [ ] ⚠️ Proceed with caution (document issues)
- [ ] ❌ Do not proceed (critical issues found)

## Notes

[Any observations, concerns, or recommendations]
```

---

### STEP 4: Test Ruff (Pilot)

#### 4.1: Install Ruff in Test Environment

```bash
cd /tmp/phentrieve-uv-test
uv add --dev ruff
```

#### 4.2: Test Ruff Format

```bash
# Copy source files to test
cp -r /mnt/c/development/phentrieve/phentrieve ./
cp -r /mnt/c/development/phentrieve/api ./
cp -r /mnt/c/development/phentrieve/tests ./

# Test Ruff format
time ruff format . --check

# Test Ruff lint
time ruff check .
```

**Record:**
- Format time
- Lint time
- Any formatting differences vs Black
- Number of lint issues found

**Create Report:** `plan/ruff-pilot-report.md`

```markdown
# Ruff Pilot Test Report

**Date:** 2025-11-14

## Performance

- **ruff format time:** [XXX seconds]
- **black time (baseline):** [XXX seconds]
- **Speed improvement:** [XX]x faster

## Linting Results

- **Files checked:** [XXX]
- **Issues found:** [XXX]
- **Categories:** [list issue types]

## Formatting Differences

- [ ] No differences (perfect drop-in replacement)
- [ ] Minor differences (acceptable)
- [ ] Major differences (requires review)

## Top Issues to Address

1. [Issue type]: [count] instances
2. [Issue type]: [count] instances
3. [Issue type]: [count] instances

## Recommendation

- [ ] ✅ Proceed with Ruff migration
- [ ] ⚠️ Proceed with caution (review differences)
- [ ] ❌ Do not proceed (too many issues)
```

---

### STEP 5: Create Rollback Scripts

#### 5.1: Create Rollback Script Directory

```bash
mkdir -p /mnt/c/development/phentrieve/scripts/rollback
```

#### 5.2: Create Phase 1 Rollback Script

**Create File:** `scripts/rollback/rollback-phase-1.sh`

**Content:**
```bash
#!/bin/bash
# Rollback Script for Phase 1 (Ruff Migration)
set -e

echo "🔄 Rolling back Phase 1: Ruff Migration"

# 1. Checkout main branch files
echo "📦 Restoring original files..."
git checkout main -- pyproject.toml

# 2. Uninstall Ruff, reinstall Black
echo "🔧 Restoring Black..."
pip uninstall -y ruff
pip install black

# 3. Reformat with Black
echo "✨ Reformatting with Black..."
black phentrieve/ api/ tests/

# 4. Verify
echo "✅ Verifying rollback..."
black phentrieve/ api/ tests/ --check

if [ $? -eq 0 ]; then
    echo "✅ Rollback successful!"
    echo "   You can now delete the feature branch:"
    echo "   git branch -D feature/phase-1-ruff"
else
    echo "❌ Rollback verification failed"
    exit 1
fi
```

**Make executable:**
```bash
chmod +x scripts/rollback/rollback-phase-1.sh
```

#### 5.3: Create Phase 2 Rollback Script

**Create File:** `scripts/rollback/rollback-phase-2.sh`

**Content:**
```bash
#!/bin/bash
# Rollback Script for Phase 2 (uv Migration)
set -e

echo "🔄 Rolling back Phase 2: uv Migration"

# 1. Remove uv files
echo "🗑️  Removing uv files..."
rm -f uv.lock .python-version

# 2. Checkout original files
echo "📦 Restoring original files..."
git checkout main -- pyproject.toml

# 3. Reinstall with pip
echo "📦 Reinstalling with pip..."
pip install -e .

# 4. Verify
echo "✅ Verifying rollback..."
python -c "import phentrieve; print('✅ Package imports successfully')"

if [ $? -eq 0 ]; then
    echo "✅ Rollback successful!"
    echo "   You can now delete the feature branch:"
    echo "   git branch -D feature/phase-2-uv"
else
    echo "❌ Rollback verification failed"
    exit 1
fi
```

**Make executable:**
```bash
chmod +x scripts/rollback/rollback-phase-2.sh
```

---

### STEP 6: Document Current System

#### 6.1: Document Current File Structure

**Create File:** `plan/current-structure.md`

**Content:**
```markdown
# Current Phentrieve Structure

**Date:** 2025-11-14

## Repository Structure

\`\`\`
phentrieve/
├── api/
│   ├── Dockerfile
│   ├── main.py
│   ├── dependencies.py
│   ├── routers/
│   └── schemas/
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── .eslintrc.js
│   ├── vite.config.js
│   └── src/
├── phentrieve/
│   ├── cli/
│   ├── data_processing/
│   ├── evaluation/
│   ├── indexing/
│   ├── retrieval/
│   └── text_processing/
├── tests/
├── pyproject.toml
├── docker-compose.yml
└── README.md
\`\`\`

## Build Tools

- **Python:** setuptools (in pyproject.toml)
- **Frontend:** Vite 4.4.11
- **Docker:** docker-compose

## Current Commands

### Python
\`\`\`bash
pip install -e .
black phentrieve/ api/ tests/
pytest tests/
\`\`\`

### Frontend
\`\`\`bash
cd frontend
npm install
npm run dev
npm run build
npm run lint
\`\`\`

### Docker
\`\`\`bash
docker-compose up
docker-compose -f docker-compose.dev.yml up
\`\`\`

## Dependencies

See `plan/requirements-baseline.txt` for full list.

## Known Issues

- [List any known issues]
```

**Execute:**
```bash
cd /mnt/c/development/phentrieve
touch plan/current-structure.md
# Fill in content manually or use tree command
tree -L 2 -I '__pycache__|node_modules|.venv|dist' > plan/tree-output.txt
```

---

### STEP 7: Team Preparation

#### 7.1: Create Training Schedule

**Create File:** `plan/training-schedule.md`

**Content:**
```markdown
# Team Training Schedule

## Overview

Weekly 1-hour training sessions throughout migration (Weeks 1-23)

## Schedule

### Week 1-2: Introduction & uv
**Date:** TBD
**Topics:**
- Migration overview
- Why we're modernizing
- uv basics: install, sync, add, remove
- Hands-on: Install uv locally

**Materials:**
- MASTER-PLAN.md
- uv documentation: https://docs.astral.sh/uv/

---

### Week 3-4: Ruff (Linting & Formatting)
**Date:** TBD
**Topics:**
- What is Ruff?
- How it replaces Black + Flake8 + isort
- VS Code integration
- Hands-on: Run ruff format and ruff check

**Materials:**
- Ruff docs: https://docs.astral.sh/ruff/

---

### Week 5-7: uv Workflows
**Date:** TBD
**Topics:**
- Adding dependencies with uv
- Lockfile benefits
- Troubleshooting
- Hands-on: Add a dependency

**Materials:**
- uv guides: https://docs.astral.sh/uv/guides/

---

[Continue for all phases...]

## Office Hours

**Time:** TBD
**Where:** TBD
**Purpose:** Q&A, troubleshooting, pair programming

## Resources

- Slack: #tooling-modernization
- Wiki: [link to documentation]
- This repo: `/plan` directory
```

#### 7.2: Create Communication Plan

**Create File:** `plan/communication-plan.md`

**Content:**
```markdown
# Communication Plan

## Stakeholders

- **Development Team:** Daily users of tooling
- **DevOps Team:** CI/CD and infrastructure
- **Product Management:** Timeline and priorities
- **Leadership:** Budget and resources

## Communication Channels

### Slack: #tooling-modernization
- **Purpose:** Daily updates, Q&A, troubleshooting
- **Audience:** All team members
- **Frequency:** As needed

### Email: Weekly Summary
- **Purpose:** Week-in-review, next week preview
- **Audience:** Stakeholders + team
- **Frequency:** Every Friday
- **Template:**
  ```
  Subject: Tooling Modernization - Week X Update

  ## This Week
  - Completed: [list]
  - In progress: [list]
  - Blocked: [list if any]

  ## Next Week
  - Goals: [list]
  - Training: [topic]

  ## Metrics
  - [relevant metrics]

  ## Questions/Concerns
  - [if any]
  ```

### GitHub: Project Board
- **Purpose:** Task tracking, progress visualization
- **Audience:** Team
- **Frequency:** Real-time updates

### Meetings: Bi-weekly Demo
- **Purpose:** Show progress, get feedback
- **Audience:** Stakeholders
- **Duration:** 30 minutes
- **Agenda:**
  1. Recap (5 min)
  2. Demo (15 min)
  3. Q&A (10 min)

## Escalation Path

1. **Minor issues:** Ask in Slack
2. **Blocking issues:** Direct message tech lead
3. **Critical issues:** Emergency meeting + email

## Success Metrics Communication

Monthly report showing:
- Performance improvements
- Time savings
- Issues resolved
- Team satisfaction
```

---

### STEP 8: Create Phase 0 Deliverables Summary

**Create File:** `plan/phase-0-deliverables.md`

**Content:**
```markdown
# Phase 0 Deliverables

**Status:** [In Progress / Complete]
**Date Completed:** [YYYY-MM-DD]

## Documents Created

- [x] `plan/baselines.md` - Performance baselines
- [x] `plan/requirements-baseline.txt` - Current dependencies
- [x] `plan/uv-pilot-report.md` - uv pilot test results
- [x] `plan/ruff-pilot-report.md` - Ruff pilot test results
- [x] `plan/current-structure.md` - Current system documentation
- [x] `plan/training-schedule.md` - Team training plan
- [x] `plan/communication-plan.md` - Communication strategy
- [x] `plan/phase-0-deliverables.md` - This file

## Scripts Created

- [x] `scripts/rollback/rollback-phase-1.sh` - Phase 1 rollback
- [x] `scripts/rollback/rollback-phase-2.sh` - Phase 2 rollback

## Pilot Tests Completed

- [x] uv installation and dependency resolution
- [x] Ruff formatting and linting

## Decisions Made

### Proceed with uv?
- [ ] ✅ Yes - proceed to Phase 2
- [ ] ⚠️ Yes with caution - [document concerns]
- [ ] ❌ No - [document reasons]

### Proceed with Ruff?
- [ ] ✅ Yes - proceed to Phase 1
- [ ] ⚠️ Yes with caution - [document concerns]
- [ ] ❌ No - [document reasons]

## Team Readiness

- [x] Training schedule created
- [x] Communication channels established
- [x] Team has reviewed MASTER-PLAN.md
- [ ] Team trained on Git workflow
- [ ] Team understands rollback procedures

## Next Steps

1. Team review of Phase 0 deliverables
2. Address any concerns from pilot tests
3. Get approval to proceed to Phase 1
4. Schedule Phase 1 kickoff meeting
```

---

### STEP 9: Commit and Push Phase 0 Work

```bash
cd /mnt/c/development/phentrieve

# Add all Phase 0 files
git add plan/ scripts/rollback/

# Commit
git commit -m "chore(phase-0): complete preparation and validation

- Add performance baselines
- Complete uv pilot test
- Complete Ruff pilot test
- Create rollback scripts
- Document current structure
- Create training and communication plans

All deliverables complete. Ready for team review."

# Push
git push origin feature/phase-0-preparation
```

---

### STEP 10: Create Pull Request

**Title:** `Phase 0: Preparation & Validation`

**Description:**
```markdown
## Phase 0: Preparation & Validation

This PR contains all preparation work for the tooling modernization project.

### Deliverables

- ✅ Performance baselines established
- ✅ uv pilot test completed
- ✅ Ruff pilot test completed
- ✅ Rollback scripts created
- ✅ Current system documented
- ✅ Training schedule created
- ✅ Communication plan created

### Pilot Test Results

**uv Migration:**
- Install time: [X seconds] (vs [Y seconds] with pip)
- Speed improvement: [Z]x faster
- Version differences: [See report]
- **Recommendation:** ✅ Proceed

**Ruff Migration:**
- Format time: [X seconds] (vs [Y seconds] with Black)
- Speed improvement: [Z]x faster
- Lint issues found: [N issues]
- **Recommendation:** ✅ Proceed

### Files Changed

- `plan/baselines.md` - Performance baselines
- `plan/*-pilot-report.md` - Pilot test reports
- `scripts/rollback/*.sh` - Rollback scripts
- [list other files]

### Next Steps

1. Team review this PR
2. Approve if pilot tests look good
3. Merge to main (no code changes, safe)
4. Proceed to Phase 1: Ruff Migration

### Validation Gate

- [ ] All tests pass (should be no changes)
- [ ] CI/CD green (should be no changes)
- [ ] Team reviewed pilot reports
- [ ] Training schedule approved
- [ ] Ready to proceed to Phase 1

### Reviewers

@team-leads @devops-team

### Related

- See: `MASTER-PLAN.md`
- See: `plan/phase-0-preparation.md` (this file)
- Next: `plan/phase-1-ruff-migration.md`
```

---

## Validation Gate Checklist

Before proceeding to Phase 1:

- [ ] All Phase 0 deliverables created
- [ ] uv pilot test shows acceptable results
- [ ] Ruff pilot test shows acceptable results
- [ ] No critical issues found in pilot tests
- [ ] Team has reviewed baselines and reports
- [ ] Training schedule approved
- [ ] Communication plan approved
- [ ] Rollback scripts tested (dry run)
- [ ] PR approved and merged
- [ ] Team ready to proceed

---

## Rollback Procedure

**Phase 0 has NO rollback** - no code changes made

To reset:
```bash
git checkout main
git branch -D feature/phase-0-preparation
```

---

## Success Criteria

- [ ] Baselines documented
- [ ] uv pilot successful (install time < 1 min)
- [ ] Ruff pilot successful (format time < 5 sec)
- [ ] Team understands next steps
- [ ] Approval to proceed to Phase 1

---

## Time Estimate

- **Minimum:** 3-4 hours (if pilot tests go smoothly)
- **Expected:** 1 week (including team review)
- **Maximum:** 2 weeks (if issues found, need investigation)

---

## Notes for LLM Execution

This plan is designed to be executed step-by-step. Each step includes:
- **Exact commands** to run
- **Expected outputs** to verify
- **Files to create** with full content
- **Validation checks** after each step

Follow steps sequentially. Do not skip validation checks.

---

**Phase 0 Complete!** → Proceed to `phase-1-ruff-migration.md`
