# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability:

### Preferred Method: GitHub Private Vulnerability Reporting

1. Go to the [Security tab](../../security) of this repository
2. Click "Report a vulnerability"
3. Fill in the details and submit

This method is preferred because it:
- Keeps the report private until a fix is ready
- Allows direct communication with maintainers
- Integrates with our security workflow

### Alternative: Email

If GitHub's private reporting is not available:
- Email the maintainers directly (see CODEOWNERS or repository settings)
- Include "SECURITY" in the subject line
- Do **NOT** open a public issue

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Affected versions
- Suggested fix (if any)

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Initial acknowledgment | 48 hours |
| Status update | 7 days |
| Fix timeline | Depends on severity |

**Severity-based fix targets:**
- **Critical**: 7 days
- **High**: 30 days
- **Medium**: 60 days
- **Low**: 90 days

## Security Measures

This project implements multiple layers of security:

### Code Security

| Tool | Purpose | Scope |
|------|---------|-------|
| **CodeQL** | SAST analysis | Python, JavaScript |
| **Bandit** | Python security linting | Python code |
| **eslint-plugin-security** | JavaScript security rules | Frontend code |
| **Ruff (S rules)** | Security checks via Ruff | Python code |

### Dependency Security

| Tool | Purpose | Scope |
|------|---------|-------|
| **Dependabot** | Automated dependency updates | All ecosystems |
| **pip-audit** | Python vulnerability scanning | Python packages |
| **npm audit** | JavaScript vulnerability scanning | npm packages |
| **Dependency Review** | PR blocking for vulnerable deps | Pull requests |

### Container Security

| Tool | Purpose | Scope |
|------|---------|-------|
| **Trivy** | Container vulnerability scanning | Docker images |
| **Dockle** | CIS benchmark compliance | Docker images |
| **Hadolint** | Dockerfile linting | Dockerfiles |

### Runtime Security (Docker)

- **Non-root users**: API runs as UID 10001, Frontend as UID 101
- **Read-only filesystems**: Containers use read-only root FS
- **Capability dropping**: All capabilities dropped (CAP_DROP: ALL)
- **Resource limits**: CPU and memory limits enforced
- **No privilege escalation**: `no-new-privileges` security option
- **Network isolation**: Internal bridge network with explicit subnets

## Security Scanning Schedule

| Scan Type | Trigger | Frequency |
|-----------|---------|-----------|
| CodeQL | Push/PR to main/develop | On every change |
| Security workflow | Push/PR to main/develop | On every change |
| Scheduled security scan | Cron | Weekly (Monday 6 AM UTC) |
| Dependabot updates | Cron | Weekly (Monday 9 AM CET) |
| Container scans | Docker build | On every build |

## Known Security Considerations

### ML Model Loading

- Models are loaded from Hugging Face Hub
- Default models are from reputable sources (BAAI, sentence-transformers)
- Users can specify custom models via configuration
- **Recommendation**: Verify model checksums when possible

### API Input Handling

- API accepts clinical text input for HPO term extraction
- Input sanitization is applied
- Rate limiting should be configured in production (via reverse proxy)

### Data Handling

- HPO data is stored in SQLite database (read-only in container)
- ChromaDB indexes store vector embeddings
- No PHI/PII is stored by default
- **Recommendation**: Configure appropriate access controls in production

## Security Updates

Security updates are prioritized and released as:

1. **Patch releases**: For critical/high severity fixes
2. **Minor releases**: For medium severity fixes
3. **Dependabot PRs**: For dependency updates

Subscribe to releases to be notified of security updates.

## Compliance

This project follows security best practices including:

- OWASP Top 10 awareness
- CIS Docker Benchmark compliance (validated by Dockle)
- Principle of least privilege
- Defense in depth

## Contact

For security-related questions that don't involve vulnerabilities:
- Open a [Discussion](../../discussions) with the "security" label
- Review existing security-related issues and discussions

---

**Last Updated**: 2025-11-30
**Policy Version**: 1.0
