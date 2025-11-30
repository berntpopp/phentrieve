# Security Scanning & CI/CD Hardening Plan

**Status:** In Progress (Implementation Complete, Awaiting PR Merge)
**Created:** 2025-11-30
**Owner:** Development Team
**Priority:** High
**Branch:** `feature/security-scanning-infrastructure`

---

## Executive Summary

This plan establishes a comprehensive security scanning infrastructure for Phentrieve, adding multiple layers of security checks to complement the existing setup. Based on best practices research from GitHub, OWASP, and security tool documentation.

### Current State (Already Implemented)

| Tool | Purpose | Status |
|------|---------|--------|
| **Dependabot** | Dependency updates | Configured for pip, npm, Docker, GitHub Actions |
| **Ruff + Bandit rules** | Python linting + security | `"S"` (bandit) rules enabled in pyproject.toml |
| **Hadolint** | Dockerfile linting | In docker-publish.yml |
| **Trivy** | Container vulnerability scanning | In docker-publish.yml |
| **Docker hardening** | Non-root users, read-only FS, capability dropping | In docker-compose.yml |

### Gaps Identified

| Gap | Risk Level | Proposed Solution |
|-----|------------|-------------------|
| No Python dependency vulnerability scanning | High | pip-audit |
| No JavaScript dependency vulnerability scanning | High | npm audit |
| No JavaScript SAST | Medium | eslint-plugin-security |
| No comprehensive SAST (both languages) | Medium | CodeQL |
| No CIS benchmark compliance for containers | Low | Dockle |
| No PR dependency review | Medium | Dependency Review Action |
| GitHub secret scanning not verified | High | Enable in repository settings |

---

## Success Criteria

- [ ] pip-audit scanning in CI (Python dependencies)
- [ ] npm audit scanning in CI (JavaScript dependencies)
- [ ] eslint-plugin-security integrated in frontend
- [ ] CodeQL workflow for Python and JavaScript
- [ ] Dockle CIS benchmark scanning for Docker images
- [ ] Dependency Review Action blocking vulnerable PRs
- [ ] GitHub Secret Scanning enabled
- [ ] All security findings uploaded to GitHub Security tab (SARIF)
- [ ] Zero HIGH/CRITICAL unaddressed vulnerabilities

---

## Implementation Phases

### Phase 1: Python Dependency Scanning (pip-audit) - Day 1

**Why:** Dependabot finds known CVEs, but pip-audit uses the PyPI Advisory Database with more comprehensive coverage and can scan the actual resolved environment.

**Implementation:**

1. Add pip-audit to dev dependencies in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    # ... existing deps ...
    "pip-audit>=2.9.0",  # Python dependency vulnerability scanner
]
```

2. Add to CI workflow (`.github/workflows/ci.yml`):

```yaml
      - name: Run pip-audit
        run: |
          uv run pip-audit --strict --progress-spinner off
        continue-on-error: false  # Fail on vulnerabilities
```

3. Add GitHub Action for SARIF upload (`.github/workflows/security.yml` - new file):

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM UTC

jobs:
  pip-audit:
    name: Python Dependency Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Run pip-audit
        uses: pypa/gh-action-pip-audit@v1.1.0
        with:
          vulnerability-service: pypi
          # Ignore specific vulns if needed:
          # ignore-vulns: GHSA-xxxx-xxxx-xxxx
```

**Validation:**
```bash
uv add --group dev pip-audit
uv run pip-audit
```

**References:**
- [pip-audit GitHub](https://github.com/pypa/pip-audit)
- [pip-audit GitHub Action](https://github.com/pypa/gh-action-pip-audit)

---

### Phase 2: JavaScript Dependency & Security Scanning - Day 1-2

**Why:** npm packages are frequently targeted (see Shai-Hulud 2025 supply chain attack). The existing ESLint setup lacks security-focused rules.

#### 2a. Add npm audit to CI

Add to `.github/workflows/ci.yml` in frontend-ci job:

```yaml
      - name: Run npm audit
        working-directory: frontend
        run: |
          npm audit --audit-level=high
        continue-on-error: true  # Report but don't fail (many false positives)

      - name: Run npm audit (production only)
        working-directory: frontend
        run: |
          npm audit --omit=dev --audit-level=critical
        continue-on-error: false  # Fail on critical production deps
```

#### 2b. Add eslint-plugin-security

1. Install the plugin:

```bash
cd frontend
npm install --save-dev eslint-plugin-security
```

2. Update `frontend/eslint.config.js`:

```javascript
import pluginSecurity from 'eslint-plugin-security';

export default [
  // ... existing config ...
  pluginSecurity.configs.recommended,
  {
    rules: {
      // Customize security rules as needed
      'security/detect-object-injection': 'warn',  // Common false positive
      'security/detect-non-literal-regexp': 'warn',
    }
  }
];
```

3. Add npm scripts to `frontend/package.json`:

```json
{
  "scripts": {
    "audit": "npm audit --audit-level=high",
    "audit:fix": "npm audit fix",
    "lint:security": "eslint --plugin security ."
  }
}
```

**Validation:**
```bash
cd frontend
npm install eslint-plugin-security --save-dev
npm run lint
```

**References:**
- [eslint-plugin-security npm](https://www.npmjs.com/package/eslint-plugin-security)
- [npm audit docs](https://www.nodejs-security.com/blog/how-to-use-npm-audit)
- [OWASP NPM Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/NPM_Security_Cheat_Sheet.html)

---

### Phase 3: CodeQL SAST Analysis - Day 2

**Why:** CodeQL provides deep semantic analysis, finding complex vulnerabilities that pattern-based tools miss. Free for open source, integrates with GitHub Security tab.

Create `.github/workflows/codeql.yml`:

```yaml
name: CodeQL Analysis

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 8 * * 1'  # Weekly Monday 8 AM UTC

jobs:
  analyze:
    name: Analyze (${{ matrix.language }})
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      packages: read
      actions: read
      contents: read

    strategy:
      fail-fast: false
      matrix:
        language: [python, javascript]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          # Use extended queries for more thorough analysis
          queries: +security-extended,security-and-quality

      # Python needs no build step (interpreted)
      # JavaScript needs no build step (we scan source, not bundles)

      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

**Configuration** (optional `.github/codeql/codeql-config.yml`):

```yaml
name: "CodeQL Config"

queries:
  - uses: security-extended
  - uses: security-and-quality

paths:
  - phentrieve
  - api
  - frontend/src

paths-ignore:
  - '**/node_modules'
  - '**/test/**'
  - '**/tests/**'
  - '**/*.test.js'
  - '**/*.spec.js'
```

**Validation:**
- Push workflow, check Actions tab
- Review Security tab for findings

**References:**
- [CodeQL Action GitHub](https://github.com/github/codeql-action)
- [CodeQL for Python](https://codeql.github.com/docs/codeql-language-support/python/)
- [GitHub Blog on CodeQL](https://github.blog/security/application-security/how-to-secure-your-github-actions-workflows-with-codeql/)

---

### Phase 4: Dockle CIS Benchmark Compliance - Day 2-3

**Why:** Hadolint checks Dockerfile syntax; Dockle checks the built image against CIS Docker Benchmark best practices.

Add to `.github/workflows/docker-publish.yml` after each build step:

```yaml
      - name: Run Dockle on API image
        uses: goodwithtech/dockle-action@v0.1.2
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME_API }}:${{ steps.meta.outputs.version }}
          format: sarif
          output: dockle-api-results.sarif
          exit-code: 1  # Fail on WARN or higher
          exit-level: warn

      - name: Upload Dockle API results
        uses: github/codeql-action/upload-sarif@v4
        if: always()
        with:
          sarif_file: dockle-api-results.sarif
          category: dockle-api
```

**Dockle checks include:**
- CIS-DI-0001: Create a user for the container
- CIS-DI-0005: Enable Content Trust
- CIS-DI-0006: Add HEALTHCHECK instruction
- CIS-DI-0008: Remove setuid/setgid from images
- DKL-DI-0001: Avoid :latest tag
- DKL-DI-0005: Clear apt-get caches

**Note:** Many checks already pass due to existing hardening. This validates compliance.

**References:**
- [Dockle GitHub](https://github.com/goodwithtech/dockle)
- [Dockle Action](https://github.com/goodwithtech/dockle-action)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)

---

### Phase 5: Dependency Review Action - Day 3

**Why:** Blocks PRs that introduce dependencies with known vulnerabilities before they merge.

Create or update `.github/workflows/ci.yml`:

```yaml
  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          # Allow specific licenses
          allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD
          # Block copyleft licenses that might conflict
          deny-licenses: GPL-3.0, AGPL-3.0
```

**Benefits:**
- Prevents introducing vulnerable dependencies
- License compliance checking
- Integrates with GitHub's dependency graph

**References:**
- [Dependency Review Action](https://github.com/actions/dependency-review-action)
- [GitHub Docs on Dependency Review](https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/about-dependency-review)

---

### Phase 6: GitHub Repository Security Settings - Day 3

**Why:** GitHub provides native security features that should be enabled.

**Manual Steps (Repository Settings > Security):**

1. **Enable Dependabot Security Updates**
   - Settings > Code security and analysis
   - Enable "Dependabot security updates"
   - This auto-creates PRs for vulnerable dependencies

2. **Enable Secret Scanning**
   - Settings > Code security and analysis
   - Enable "Secret scanning"
   - Enable "Push protection" (blocks commits with secrets)

3. **Enable Private Vulnerability Reporting**
   - Settings > Code security and analysis
   - Enable "Private vulnerability reporting"
   - Allows security researchers to report issues privately

4. **Branch Protection Rules**
   - Settings > Branches > main
   - Require status checks: `CodeQL`, `pip-audit`, `frontend-ci`
   - Require pull request reviews
   - Require signed commits (optional but recommended)

5. **Security Policy**
   - Create `SECURITY.md` in repository root

**SECURITY.md Template:**

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a vulnerability:

1. **Do NOT** open a public issue
2. Use GitHub's private vulnerability reporting (preferred)
3. Or email: security@[project-domain].com

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- Initial response: 48 hours
- Status update: 7 days
- Fix timeline: depends on severity

## Security Measures

This project uses:
- CodeQL SAST analysis
- Dependabot dependency updates
- pip-audit and npm audit
- Trivy container scanning
- Non-root Docker containers
- Read-only filesystems

## Known Security Considerations

- ML models are loaded from Hugging Face Hub
- Verify model checksums when possible
- API accepts clinical text input (sanitization applied)
```

**References:**
- [GitHub Security Features Docs](https://docs.github.com/en/code-security/getting-started/github-security-features)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)

---

## Complete Security Workflow

After implementation, create unified `.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM UTC

concurrency:
  group: security-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # Python dependency scanning
  pip-audit:
    name: Python Dependency Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Run pip-audit
        uses: pypa/gh-action-pip-audit@v1.1.0
        with:
          vulnerability-service: pypi

  # Python SAST (Bandit via Ruff)
  bandit:
    name: Python SAST (Bandit)
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Run Bandit via Ruff
        run: |
          uv run ruff check phentrieve/ api/ --select S --output-format github

      # Optional: Run standalone Bandit for SARIF output
      - name: Run Bandit (SARIF)
        uses: PyCQA/bandit-action@v1
        with:
          targets: phentrieve api
          configfile: pyproject.toml
        continue-on-error: true

  # JavaScript security
  npm-audit:
    name: JavaScript Dependency Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6

      - name: Setup Node.js
        uses: actions/setup-node@v6
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json

      - name: Install dependencies
        working-directory: frontend
        run: npm ci

      - name: Run npm audit (production)
        working-directory: frontend
        run: npm audit --omit=dev --audit-level=critical

      - name: Run npm audit (full report)
        working-directory: frontend
        run: npm audit --audit-level=high || true

  # Dependency review (PR only)
  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v6
      - uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 0BSD

  # Security summary
  security-summary:
    name: Security Summary
    needs: [pip-audit, bandit, npm-audit]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Check results
        run: |
          echo "pip-audit: ${{ needs.pip-audit.result }}"
          echo "bandit: ${{ needs.bandit.result }}"
          echo "npm-audit: ${{ needs.npm-audit.result }}"

          if [[ "${{ needs.pip-audit.result }}" == "failure" ]] || \
             [[ "${{ needs.bandit.result }}" == "failure" ]]; then
            echo "::error::Security scan failed"
            exit 1
          fi

          echo "::notice::All security scans passed"
```

---

## Makefile Updates

Add convenience targets to `Makefile`:

```makefile
# Security scanning
.PHONY: security security-python security-frontend

security: security-python security-frontend  ## Run all security scans

security-python:  ## Run Python security scans
	@echo "Running pip-audit..."
	uv run pip-audit --strict
	@echo "Running Bandit via Ruff..."
	uv run ruff check phentrieve/ api/ --select S

security-frontend:  ## Run frontend security scans
	@echo "Running npm audit..."
	cd frontend && npm audit --audit-level=high || true
	@echo "Running ESLint security rules..."
	cd frontend && npm run lint
```

---

## Pre-commit Integration (Optional)

Add security checks to `.pre-commit-config.yaml`:

```yaml
repos:
  # Python security
  - repo: https://github.com/PyCQA/bandit
    rev: 1.9.2
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml", "-r", "phentrieve", "api"]
        additional_dependencies: ["bandit[toml]"]

  # pip-audit (slower, run manually or in CI)
  # - repo: https://github.com/pypa/pip-audit
  #   rev: v2.9.0
  #   hooks:
  #     - id: pip-audit
```

---

## Timeline

| Phase | Task | Duration | Dependency |
|-------|------|----------|------------|
| 1 | pip-audit integration | 2 hours | None |
| 2 | npm audit + eslint-plugin-security | 3 hours | None |
| 3 | CodeQL workflow | 2 hours | None |
| 4 | Dockle integration | 2 hours | Phase 3 |
| 5 | Dependency Review Action | 1 hour | None |
| 6 | GitHub settings + SECURITY.md | 1 hour | None |
| - | **Total** | **~11 hours** | - |

**Recommended Order:** Phases 1-3 can run in parallel, then 4-6.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| False positives block development | High | Use `continue-on-error` initially, tune later |
| Security scan timeouts | Medium | Set appropriate timeouts, cache dependencies |
| Alert fatigue | Medium | Focus on HIGH/CRITICAL first, use suppressions |
| License violations discovered | Low | Review deny list, adjust if needed |

---

## Rollback Plan

If security scanning causes issues:

```bash
# Revert security workflow
git revert [security-workflow-commit]

# Disable specific scan in workflow
# Edit .github/workflows/security.yml
# Add: if: false
```

---

## Success Metrics

After implementation:

1. **Security tab populated** with CodeQL, Trivy, Dockle findings
2. **All HIGH/CRITICAL vulnerabilities** triaged within 7 days
3. **PRs blocked** if introducing known vulnerabilities
4. **Secret scanning active** with push protection enabled
5. **Weekly security reports** generated automatically

---

## References

### Official Documentation
- [GitHub Security Features](https://docs.github.com/en/code-security/getting-started/github-security-features)
- [GitHub CodeQL](https://github.com/github/codeql-action)
- [Dependabot Docs](https://docs.github.com/en/code-security/dependabot)

### Security Tools
- [pip-audit](https://github.com/pypa/pip-audit) - Python dependency scanning
- [Bandit](https://bandit.readthedocs.io/) - Python SAST
- [eslint-plugin-security](https://www.npmjs.com/package/eslint-plugin-security) - JavaScript security linting
- [Dockle](https://github.com/goodwithtech/dockle) - Docker CIS benchmark
- [Trivy](https://github.com/aquasecurity/trivy) - Container scanning

### Best Practices
- [OWASP NPM Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/NPM_Security_Cheat_Sheet.html)
- [GitHub Supply Chain Security](https://docs.github.com/en/code-security/supply-chain-security)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)

### 2025 Security Incidents (Reference)
- [Shai-Hulud npm Attack (CISA)](https://www.cisa.gov/news-events/alerts/2025/09/23/widespread-supply-chain-compromise-impacting-npm-ecosystem)
- [eslint-config-prettier CVE-2025-54313](https://www.endorlabs.com/learn/cve-2025-54313-eslint-config-prettier-compromise----high-severity-but-windows-only)

---

**Generated:** 2025-11-30
**Author:** Claude Code (AI-assisted research)
**Review Status:** Pending human review
