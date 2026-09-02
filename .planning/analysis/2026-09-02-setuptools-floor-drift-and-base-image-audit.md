# setuptools Floor Drift and Container Base Image Audit

Date: 2026-09-02
Scope: the single open Trivy code-scanning alert on the published API image, plus a
confirm-and-document audit of the API and frontend base images.

## Summary

The alert was not caused by the mismatched `setuptools` floors themselves. It was
caused by the floors being *non-binding*: `api/Dockerfile` upgraded setuptools with
`--upgrade` against a floor five releases below the patched one, so the resolved
version was decided by "whatever was newest when BuildKit materialised the layer",
and the registry build cache then replayed that layer unchanged for months.

All three floors are now pinned to `83.0.0`, the fixed release, and
`tests/unit/test_dependency_security_policy.py` fails if they drift apart again.

The base images are stale but the published images scan clean. No base image was
bumped in this change; the recommendations below are follow-up work.

## Task 1 - Root cause of CVE-2026-59890

### The advisory

`CVE-2026-59890`, `PYSEC-2026-3447` and `GHSA-h35f-9h28-mq5c` are aliases of one
advisory, not three. OSV gives a single fixed version:

| Field | Value |
| --- | --- |
| Summary | MANIFEST.in exclusion bypass in sdist via Unicode normalization collision (NFC/NFD) on macOS APFS/HFS+ |
| Affected | setuptools `< 83.0.0` |
| Fixed | `83.0.0` |
| Trivy severity | MEDIUM |

The dev-group floor `setuptools>=83.0.0` was therefore already correct and already
covered this CVE. Only the build-system floor and the Dockerfile floor were short.

### What was disproved

The prior hypothesis was that `pyproject.toml`'s
`[build-system] requires = ["setuptools>=82.0.1"]` leaked into `/opt/venv`. A probe
build reproducing the exact Dockerfile venv sequence shows it does not:

| Probe point | setuptools in `/opt/venv` |
| --- | --- |
| after `python -m venv` | 65.5.0 (bundled by ensurepip) |
| after `pip install --upgrade "setuptools>=78.1.1"` | 84.0.0 |
| after `pip install .` | 84.0.0 (unchanged) |

`pip install .` resolves the build requirement inside pip's isolated build
environment. Verbose output shows two separate installs: `setuptools-84.0.0` into the
throwaway build env, then `phentrieve-0.28.0` into the venv. The build-system floor
never touches the runtime venv.

Note also that the stale `>=78.1.1` floor resolves to 84.0.0 today. On its own it
cannot explain a shipped 82.0.1 either.

### What was proved

The layer never re-ran. Evidence:

| Artifact | Image built | setuptools shipped | Was that version newest at build time? |
| --- | --- | --- | --- |
| local API image | 2026-06-15 | 81.0.0 | no, 81.0.0 was newest only 2026-02-06 to 2026-02-08 |
| published API image | 2026-09-02 | 82.0.1 | no, superseded 2026-07-04 by 83.0.0 |
| fresh probe build | 2026-09-02 | 84.0.0 | yes |

An image rebuilt on 2026-09-02 shipping a March release, while an uncached build of
the same instruction resolves 84.0.0, means the `pip install --upgrade` layer was
served from cache rather than executed. `docker-publish.yml` supplies that cache with
`cache-from: type=registry,ref=...:buildcache`. Because the `RUN` command string never
changed, the layer stayed valid indefinitely and froze its original resolution.

This is the same failure mode already documented in `api/Dockerfile` for the OpenSSL
packages, where a warm registry cache pinned a stale `apt-get upgrade` layer. The fix
follows the same pattern: name the patched version so the constraint is binding, and
so that editing the line busts the cached layer.

### Scanner state

The published API image has exactly one Trivy finding at any severity:

```
ghcr.io/berntpopp/phentrieve/api:latest   TOTAL=1 CRITICAL=0 HIGH=0
    MEDIUM   setuptools@82.0.1 -> fixed 83.0.0  CVE-2026-59890  [Python]
```

The published frontend image has zero findings at any severity. Trivy's current
database rates this MEDIUM, below the `severity: CRITICAL,HIGH` filter in
`docker-publish.yml`, so a future scan will no longer emit it regardless. Pinning the
floor removes the finding at its source rather than relying on that filter.

## Task 2 - Base image audit

### Current pins and available patch tags

| Dockerfile | Pinned | Pushed | Newest equivalent | Pushed |
| --- | --- | --- | --- | --- |
| `api/Dockerfile` | `python:3.11.13-slim-trixie` | 2025-10-02 | `python:3.11.16-slim-trixie` | 2026-09-01 |
| `frontend/Dockerfile` | `node:20.19-alpine3.20` | 2025-05-16 | `node:20.20.2-alpine3.23` | 2026-04-17 |
| `frontend/Dockerfile` | `nginx-unprivileged:1.27-alpine3.20-slim` | 2025-02-01 | `nginx-unprivileged:1.29-alpine3.23-slim` | current |

### Trivy on the base images

Same filters CI uses (`--severity CRITICAL,HIGH --ignore-unfixed`):

| Base image | Total | Critical | High |
| --- | --- | --- | --- |
| `python:3.11.13-slim-trixie` | 58 | 3 | 55 |
| `python:3.11.16-slim-trixie` | 2 | 0 | 2 |
| `node:20.19-alpine3.20` | 37 | 3 | 34 |
| `node:20.20.2-alpine3.23` | 24 | 1 | 23 |
| `nginx-unprivileged:1.27-alpine3.20-slim` | 21 | 2 | 19 |
| `nginx-unprivileged:1.29-alpine3.23-slim` | 4 | 0 | 4 |

None of this reaches the published images, because both Dockerfiles already patch at
build time. The API runtime stage runs `apt-get upgrade`, force-upgrades the OpenSSL
family, and deletes the base image's `setuptools`, `pkg_resources` and `wheel`, which
is what clears the `setuptools@65.5.1` and `wheel@0.45.1` rows. The frontend runtime
stage runs `apk update && apk upgrade`.

That mitigation is verifiably working. The published frontend image, on an
alpine 3.20.5 base, carries the patched packages:

| Package | Installed | Trivy's fixed version |
| --- | --- | --- |
| libssl3 / libcrypto3 | 3.3.7-r0 | 3.3.7-r0 |
| musl | 1.2.5-r3 | 1.2.5-r3 |
| zlib | 1.3.2-r0 | 1.3.2-r0 |

### End-of-life exposure

This is the finding that matters more than the patch-tag lag.

| Component | Status | Source |
| --- | --- | --- |
| Node 20 | end-of-life 2026-04-30 | nodejs/Release schedule.json |
| Alpine 3.20 | end-of-life 2026-04-01 | alpinelinux.org/releases.json |
| Debian 13 trixie | current stable | - |

Both frontend base images sit on branches that are past end-of-life. The clean scan
depends on `apk upgrade` continuing to serve fixes from the alpine 3.20 branch, which
it still does today, but that is no longer a supported guarantee.

### Bumping does not uniformly help

Node is the clear case. `node:20.19-alpine3.20` and `node:20.20.2-alpine3.23` report
an identical set of npm-bundled findings, because both bundle the same
`tar@6.2.1`, `minimatch@9.0.5`, `brace-expansion@2.0.1`, `cross-spawn@7.0.3`,
`glob@10.4.2`, `ip-address@9.0.5`, `pacote@18.0.6` and `sigstore@2.3.1`. The bump
fixes only the Alpine OS layer. Since node appears in builder stages only and never
in the runtime image, none of it reaches the published frontend image either way.

### Recommendations, not applied here

1. Bump `api/Dockerfile` to `python:3.11.16-slim-trixie`. Pure patch bump inside
   3.11, drops the base from 58 findings to 2, and reduces reliance on
   `apt-get upgrade`. Verify with a Trivy scan of the rebuilt runtime image.
2. Move the frontend off end-of-life bases: Node 22 or 24 LTS on alpine 3.22+, and
   `nginx-unprivileged` on alpine 3.23. This is a Node major upgrade and warrants its
   own change with a frontend build and test run, not a rider on a CVE floor fix.
3. Consider a scheduled cache-busting rebuild, or `--no-cache` on a periodic job, so
   that pinned-floor drift cannot silently re-accumulate in cached layers.

## Changes made

| File | Change |
| --- | --- |
| `pyproject.toml` | `[build-system]` floor `82.0.1` to `83.0.0`, with the rationale |
| `api/Dockerfile` | pip upgrade floor `78.1.1` to `83.0.0`, documenting the cache-freeze mechanism |
| `tests/unit/test_dependency_security_policy.py` | new test tying the build-system and Dockerfile floors to the dev-group floor |

The Dockerfile edit changes the `RUN` command string, so the next CI build cannot
reuse the cached layer. Every later layer in the `python-builder` stage is invalidated
with it, so expect one full rebuild of the API image.
