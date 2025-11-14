# Team Training Schedule

**Project:** Phentrieve Tooling Modernization
**Duration:** 20-25 weeks (Phases 1-9)
**Last Updated:** 2025-11-14

---

## Training Philosophy

**Approach:** Just-in-time training before each phase
**Format:** Mix of self-study, hands-on workshops, and reference documentation
**Time Commitment:** 2-4 hours per phase (spread across phase duration)

---

## Phase-by-Phase Training Plan

### Phase 1: Ruff Migration (Weeks 3-4)

**Training Date:** Week 3, Day 1
**Duration:** 2 hours
**Format:** Workshop + Documentation

#### Objectives
- Understand Ruff capabilities (linter + formatter)
- Learn Ruff configuration in `pyproject.toml`
- Understand auto-fix workflow
- Practice using Ruff in development

#### Materials
1. **Pre-Reading** (30 minutes)
   - Ruff official docs: https://docs.astral.sh/ruff/
   - `plan/ruff-pilot-report.md` - Our findings
   - Ruff vs Black comparison

2. **Workshop** (90 minutes)
   - Introduction to Ruff (15 min)
   - Configuration walkthrough (15 min)
   - Hands-on: Format and lint code (30 min)
   - Auto-fix demo and practice (20 min)
   - Q&A and troubleshooting (10 min)

3. **Reference Materials**
   - Ruff rule reference: https://docs.astral.sh/ruff/rules/
   - `pyproject.toml` Ruff configuration
   - `plan/phase-1-ruff-migration.md`

#### Commands to Learn
```bash
# Format code
ruff format phentrieve/ api/ tests/

# Check linting
ruff check phentrieve/ api/ tests/

# Auto-fix issues
ruff check phentrieve/ api/ tests/ --fix

# Check specific file
ruff check phentrieve/cli/query_commands.py
```

#### Success Criteria
- ✅ Can format code with Ruff
- ✅ Can run linting checks
- ✅ Can interpret Ruff error messages
- ✅ Can use auto-fix feature
- ✅ Comfortable with Ruff configuration

---

### Phase 2: uv Migration (Weeks 5-7)

**Training Date:** Week 5, Day 1
**Duration:** 3 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand uv package manager
- Learn dependency management with `uv.lock`
- Practice virtual environment creation
- Understand version resolution differences

#### Materials
1. **Pre-Reading** (45 minutes)
   - uv official docs: https://docs.astral.sh/uv/
   - `plan/uv-pilot-report.md` - Our findings
   - uv vs pip comparison
   - Version pinning strategies

2. **Workshop** (135 minutes)
   - Introduction to uv (20 min)
   - Installation and setup (15 min)
   - Virtual environment management (20 min)
   - Dependency management with uv.lock (30 min)
   - Version resolution walkthrough (20 min)
   - Troubleshooting common issues (20 min)
   - Q&A (10 min)

3. **Reference Materials**
   - uv command reference: https://docs.astral.sh/uv/reference/
   - `plan/requirements-baseline.txt` - Version baseline
   - `scripts/rollback-phase-2.sh` - Rollback procedure

#### Commands to Learn
```bash
# Create virtual environment
uv venv

# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Remove dependency
uv remove package-name

# Update dependencies
uv lock --upgrade

# Install in editable mode
uv pip install -e .
```

#### Success Criteria
- ✅ Can create virtual environments with uv
- ✅ Can install dependencies with `uv sync`
- ✅ Can add/remove dependencies
- ✅ Understands uv.lock file
- ✅ Can troubleshoot version conflicts
- ✅ Knows when to use `--upgrade`

---

### Phase 3: mypy Integration (Weeks 8-9)

**Training Date:** Week 8, Day 1
**Duration:** 2.5 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand static type checking benefits
- Learn Python type hints syntax
- Practice adding type annotations
- Interpret mypy error messages

#### Materials
1. **Pre-Reading** (45 minutes)
   - mypy official docs: https://mypy.readthedocs.io/
   - Python typing module docs
   - Type hints best practices
   - Gradual typing approach

2. **Workshop** (105 minutes)
   - Introduction to static typing (15 min)
   - Type hints syntax overview (20 min)
   - mypy configuration (15 min)
   - Hands-on: Add type hints (30 min)
   - Interpreting mypy errors (15 min)
   - Q&A (10 min)

3. **Reference Materials**
   - mypy cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
   - typing module reference
   - `pyproject.toml` mypy configuration

#### Commands to Learn
```bash
# Check types
mypy phentrieve/ api/

# Check specific file
mypy phentrieve/cli/query_commands.py

# Show error context
mypy --show-error-context phentrieve/

# Generate type stubs
stubgen -p phentrieve
```

#### Success Criteria
- ✅ Can add basic type hints (str, int, List, Dict)
- ✅ Can add function annotations (parameters and return types)
- ✅ Can interpret mypy error messages
- ✅ Understands Optional, Union, Any
- ✅ Comfortable with gradual typing approach

---

### Phase 4: Frontend Modernization (Weeks 10-12)

**Training Date:** Week 10, Day 1
**Duration:** 3 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand ESLint 9 flat config
- Learn Prettier configuration
- Practice frontend linting and formatting
- Understand migration from old ESLint

#### Materials
1. **Pre-Reading** (60 minutes)
   - ESLint 9 flat config guide: https://eslint.org/docs/latest/use/configure/configuration-files
   - Prettier docs: https://prettier.io/docs/
   - Vue.js ESLint plugin docs
   - ESLint flat config migration guide

2. **Workshop** (120 minutes)
   - ESLint 9 flat config overview (20 min)
   - Migration from .eslintrc (15 min)
   - Prettier setup and integration (15 min)
   - Hands-on: Lint and format Vue files (30 min)
   - ESLint + Prettier integration (20 min)
   - Q&A (20 min)

3. **Reference Materials**
   - ESLint rules: https://eslint.org/docs/latest/rules/
   - Prettier options: https://prettier.io/docs/en/options.html
   - `frontend/eslint.config.js` - Our config
   - `frontend/.prettierrc` - Our config

#### Commands to Learn
```bash
cd frontend/

# Lint code
npm run lint

# Format code
npm run format

# Lint and fix
npm run lint:fix

# Check Prettier formatting
npm run format:check
```

#### Success Criteria
- ✅ Understands ESLint 9 flat config structure
- ✅ Can configure ESLint rules
- ✅ Can use Prettier for formatting
- ✅ Knows how to disable rules when needed
- ✅ Comfortable with Vue-specific linting

---

### Phase 5: Vitest Setup (Weeks 13-14)

**Training Date:** Week 13, Day 1
**Duration:** 2.5 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand Vitest testing framework
- Learn Vue component testing
- Practice writing unit tests
- Understand Vite integration

#### Materials
1. **Pre-Reading** (45 minutes)
   - Vitest docs: https://vitest.dev/
   - Vue Test Utils: https://test-utils.vuejs.org/
   - Vitest vs Jest comparison
   - Testing best practices

2. **Workshop** (105 minutes)
   - Introduction to Vitest (15 min)
   - Configuration in vite.config.js (15 min)
   - Writing unit tests (25 min)
   - Component testing with Vue Test Utils (25 min)
   - Coverage and reporting (15 min)
   - Q&A (10 min)

3. **Reference Materials**
   - Vitest API: https://vitest.dev/api/
   - Vue Test Utils API: https://test-utils.vuejs.org/api/
   - `frontend/vite.config.js` - Our config

#### Commands to Learn
```bash
cd frontend/

# Run tests
npm run test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run specific test file
npm run test -- src/components/MyComponent.spec.js
```

#### Success Criteria
- ✅ Can write basic unit tests
- ✅ Can test Vue components
- ✅ Can run tests and interpret results
- ✅ Understands test coverage reports
- ✅ Comfortable with Vitest configuration

---

### Phase 6: Modular Makefile (Weeks 15-16)

**Training Date:** Week 15, Day 1
**Duration:** 1.5 hours
**Format:** Workshop + Documentation

#### Objectives
- Understand Makefile structure
- Learn modular Makefile organization
- Practice running make targets
- Understand common targets

#### Materials
1. **Pre-Reading** (30 minutes)
   - GNU Make documentation (basics)
   - Makefile best practices
   - `plan/phase-6-*.md` (when created)

2. **Workshop** (60 minutes)
   - Makefile basics (15 min)
   - Our modular structure (mk/ directory) (15 min)
   - Common targets walkthrough (20 min)
   - Q&A (10 min)

3. **Reference Materials**
   - `Makefile` - Main file
   - `mk/python.mk` - Python targets
   - `mk/frontend.mk` - Frontend targets
   - `mk/docker.mk` - Docker targets

#### Commands to Learn
```bash
# Show available targets
make help

# Python targets
make format        # Format Python code
make lint          # Lint Python code
make test          # Run Python tests
make typecheck     # Run mypy

# Frontend targets
make frontend-lint     # Lint frontend
make frontend-format   # Format frontend
make frontend-test     # Run frontend tests

# Docker targets
make docker-build  # Build Docker images
make docker-up     # Start containers
make docker-down   # Stop containers

# Unified targets
make all           # Format + lint + test everything
make clean         # Clean build artifacts
```

#### Success Criteria
- ✅ Can run common make targets
- ✅ Understands Makefile structure
- ✅ Knows where to find target definitions
- ✅ Can add simple new targets if needed
- ✅ Comfortable with unified workflow

---

### Phase 7: GHCR Migration (Weeks 17-19)

**Training Date:** Week 17, Day 1
**Duration:** 2 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand GitHub Container Registry (GHCR)
- Learn authentication and permissions
- Practice pushing/pulling images
- Understand CI/CD integration

#### Materials
1. **Pre-Reading** (40 minutes)
   - GHCR docs: https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry
   - GHCR vs DockerHub comparison
   - GitHub Personal Access Tokens

2. **Workshop** (80 minutes)
   - Introduction to GHCR (15 min)
   - Authentication setup (15 min)
   - Pushing/pulling images (20 min)
   - CI/CD integration (20 min)
   - Q&A (10 min)

3. **Reference Materials**
   - GHCR authentication guide
   - `docker-compose.yml` - Updated config
   - `.github/workflows/` - CI/CD workflows

#### Commands to Learn
```bash
# Login to GHCR
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin

# Tag image for GHCR
docker tag phentrieve ghcr.io/USERNAME/phentrieve:latest

# Push to GHCR
docker push ghcr.io/USERNAME/phentrieve:latest

# Pull from GHCR
docker pull ghcr.io/USERNAME/phentrieve:latest
```

#### Success Criteria
- ✅ Can authenticate with GHCR
- ✅ Can push images to GHCR
- ✅ Can pull images from GHCR
- ✅ Understands GHCR permissions
- ✅ Comfortable with GHCR in CI/CD

---

### Phase 8: Dependabot (Weeks 20-21)

**Training Date:** Week 20, Day 1
**Duration:** 1 hour
**Format:** Documentation + Demo

#### Objectives
- Understand Dependabot automation
- Learn to review Dependabot PRs
- Practice merging dependency updates
- Understand security alerts

#### Materials
1. **Pre-Reading** (20 minutes)
   - Dependabot docs: https://docs.github.com/en/code-security/dependabot
   - Dependabot configuration reference
   - Security best practices

2. **Workshop** (40 minutes)
   - Dependabot overview (10 min)
   - Configuration walkthrough (10 min)
   - PR review process (10 min)
   - Security alerts demo (5 min)
   - Q&A (5 min)

3. **Reference Materials**
   - `.github/dependabot.yml` - Our config
   - Dependabot PR review checklist
   - Security vulnerability handling

#### Success Criteria
- ✅ Can review Dependabot PRs
- ✅ Knows when to merge vs investigate
- ✅ Understands security alerts
- ✅ Can configure Dependabot settings
- ✅ Comfortable with automated updates

**IMPORTANT:** NO auto-merge per security review!

---

### Phase 9: CI/CD Optimization (Weeks 22-23)

**Training Date:** Week 22, Day 1
**Duration:** 2 hours
**Format:** Workshop + Hands-on Practice

#### Objectives
- Understand GitHub Actions optimization
- Learn change detection strategies
- Practice debugging workflows
- Understand caching strategies

#### Materials
1. **Pre-Reading** (40 minutes)
   - GitHub Actions docs: https://docs.github.com/en/actions
   - GitHub Actions best practices
   - Workflow optimization guide

2. **Workshop** (80 minutes)
   - GitHub Actions review (10 min)
   - Change detection setup (20 min)
   - Caching strategies (20 min)
   - Debugging workflows (20 min)
   - Q&A (10 min)

3. **Reference Materials**
   - `.github/workflows/` - All workflows
   - GitHub Actions syntax reference
   - Workflow optimization checklist

#### Commands to Learn
```bash
# Test workflow locally (with act)
act -l  # List workflows
act push  # Run push workflow

# View workflow logs
gh run list
gh run view RUN_ID
gh run watch RUN_ID
```

#### Success Criteria
- ✅ Can understand workflow files
- ✅ Can debug failing workflows
- ✅ Understands change detection
- ✅ Knows caching strategies
- ✅ Comfortable with GitHub Actions

---

## Training Resources

### Online Documentation

**Official Docs:**
- Ruff: https://docs.astral.sh/ruff/
- uv: https://docs.astral.sh/uv/
- mypy: https://mypy.readthedocs.io/
- ESLint: https://eslint.org/docs/
- Prettier: https://prettier.io/docs/
- Vitest: https://vitest.dev/
- GHCR: https://docs.github.com/en/packages
- Dependabot: https://docs.github.com/en/code-security/dependabot
- GitHub Actions: https://docs.github.com/en/actions

**Project Docs:**
- `plan/MASTER-PLAN.md` - Overall strategy
- `plan/README.md` - Navigation guide
- `plan/phase-*.md` - Phase-specific plans
- `scripts/README.md` - Rollback procedures

### Internal Resources

**Knowledge Base:**
- `CLAUDE.md` - Project overview and commands
- `plan/current-structure.md` - Current system structure
- `plan/baselines.md` - Performance baselines
- `plan/*-pilot-report.md` - Pilot test findings

**Slack/Communication:**
- Create #tooling-modernization channel
- Daily standups during migration phases
- Weekly retrospectives

---

## Training Checklist

### Before Each Phase

- [ ] Review phase-specific plan (`plan/phase-*.md`)
- [ ] Complete pre-reading materials (30-60 minutes)
- [ ] Attend workshop session (1-3 hours)
- [ ] Practice hands-on exercises
- [ ] Ask questions in team channel

### During Each Phase

- [ ] Reference documentation as needed
- [ ] Share learnings with team
- [ ] Report issues/blockers immediately
- [ ] Participate in code reviews
- [ ] Help team members if they struggle

### After Each Phase

- [ ] Retrospective: What worked? What didn't?
- [ ] Update documentation if needed
- [ ] Share tips and tricks with team
- [ ] Prepare for next phase

---

## Individual Learning Paths

### Backend Developers (Python Focus)

**Priority Phases:**
1. ⭐ Phase 1: Ruff (critical)
2. ⭐ Phase 2: uv (critical)
3. ⭐ Phase 3: mypy (critical)
4. ✅ Phase 6: Makefile (important)
5. ✅ Phase 8: Dependabot (good to know)
6. ✅ Phase 9: CI/CD (good to know)

**Optional:**
- Phase 4: Frontend (helpful)
- Phase 5: Vitest (helpful)
- Phase 7: GHCR (helpful)

### Frontend Developers (Vue/JS Focus)

**Priority Phases:**
1. ⭐ Phase 4: Frontend Modernization (critical)
2. ⭐ Phase 5: Vitest (critical)
3. ✅ Phase 6: Makefile (important)
4. ✅ Phase 8: Dependabot (good to know)
5. ✅ Phase 9: CI/CD (good to know)

**Optional:**
- Phase 1: Ruff (helpful)
- Phase 2: uv (helpful)
- Phase 3: mypy (helpful)
- Phase 7: GHCR (helpful)

### Full-Stack Developers

**All Phases:**
1. ⭐ Phases 1-6: Critical (all tools)
2. ✅ Phases 7-9: Important (deployment/automation)

### DevOps/Infrastructure

**Priority Phases:**
1. ⭐ Phase 2: uv (critical for deployment)
2. ⭐ Phase 6: Makefile (critical for automation)
3. ⭐ Phase 7: GHCR (critical)
4. ⭐ Phase 8: Dependabot (critical)
5. ⭐ Phase 9: CI/CD (critical)

**Optional:**
- Phase 1: Ruff (good to know)
- Phase 3: mypy (good to know)
- Phase 4: Frontend (good to know)
- Phase 5: Vitest (good to know)

---

## Support and Questions

### During Training

**Questions?**
1. Ask in workshop sessions (preferred)
2. Post in #tooling-modernization Slack channel
3. Create GitHub discussion
4. Reach out to phase lead

### After Training

**Ongoing Support:**
- Documentation in `plan/` directory
- Rollback scripts in `scripts/` directory
- Team knowledge sharing
- Pair programming sessions

**Common Issues:**
- Check `plan/phase-*.md` troubleshooting sections
- Review pilot reports (`*-pilot-report.md`)
- Consult official documentation
- Ask team for help

---

## Training Feedback

After each phase, please provide feedback on:
1. Was pre-reading sufficient?
2. Was workshop duration appropriate?
3. Were hands-on exercises helpful?
4. What could be improved?
5. Additional resources needed?

**Feedback Channels:**
- Phase retrospectives
- Anonymous survey (if preferred)
- Direct feedback to phase lead

---

## Timeline Summary

| Phase | Week | Training Date | Duration | Focus |
|-------|------|---------------|----------|-------|
| 1: Ruff | 3-4 | Week 3, Day 1 | 2h | Linter/Formatter |
| 2: uv | 5-7 | Week 5, Day 1 | 3h | Package Manager |
| 3: mypy | 8-9 | Week 8, Day 1 | 2.5h | Type Checking |
| 4: Frontend | 10-12 | Week 10, Day 1 | 3h | ESLint 9 + Prettier |
| 5: Vitest | 13-14 | Week 13, Day 1 | 2.5h | Frontend Testing |
| 6: Makefile | 15-16 | Week 15, Day 1 | 1.5h | Build System |
| 7: GHCR | 17-19 | Week 17, Day 1 | 2h | Container Registry |
| 8: Dependabot | 20-21 | Week 20, Day 1 | 1h | Dependency Updates |
| 9: CI/CD | 22-23 | Week 22, Day 1 | 2h | Workflow Optimization |

**Total Training Time:** ~19.5 hours (spread across 20 weeks)
**Average:** ~1 hour per week

---

**Status:** Training schedule complete ✅
**Next:** Communication plan (Step 8)
**Purpose:** Ensure team is prepared for each phase of modernization
