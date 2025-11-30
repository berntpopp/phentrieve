# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes |
| < 0.4   | No |

## Reporting a Vulnerability

**Do NOT open a public issue.**

1. Use [GitHub's private vulnerability reporting](../../security/advisories/new)
2. Or email maintainers directly (see repository settings)

Include: description, steps to reproduce, impact, affected versions.

**Response times:** Initial acknowledgment within 48 hours. Fix timeline depends on severity (Critical: 7 days, High: 30 days).

## Security Measures

This project uses automated security scanning:

- **SAST**: CodeQL, Bandit, eslint-plugin-security
- **Dependencies**: Dependabot, pip-audit, npm audit
- **Containers**: Trivy, Dockle, Hadolint
- **Runtime**: Non-root users, read-only FS, dropped capabilities

Scans run on every push/PR and weekly.

## Considerations

- ML models loaded from Hugging Face Hub (verify checksums when possible)
- Configure rate limiting in production
- No PHI/PII stored by default
