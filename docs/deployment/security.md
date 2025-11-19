# Security Configuration

Phentrieve's Docker deployment is hardened for production use to minimize attack surfaces and follow security best practices. This document details the comprehensive security measures implemented in the Docker containers.

## Security Philosophy

The deployment follows a **defense-in-depth** approach with multiple layers of security controls:

1. **Principle of Least Privilege**: Containers run with minimal permissions
2. **Immutable Infrastructure**: Read-only filesystems prevent runtime tampering
3. **Resource Isolation**: CPU and memory limits prevent DoS attacks
4. **Network Segmentation**: Internal networks isolate services
5. **Minimal Attack Surface**: Only essential capabilities granted

## Container Security Hardening

### 1. Non-Root Execution

**Configuration:**
```yaml
# API Container
user: "10001:10001"  # phentrieve:phentrieve

# Frontend Container (Nginx)
user: "101:101"      # nginx:nginx
```

**Why This Matters:**
- Prevents container breakout from escalating to root on host
- Limits damage if application is compromised
- Enforces principle of least privilege

**Verification:**
```bash
# Check user inside API container
docker exec phentrieve-api-1 id
# Output: uid=10001(phentrieve) gid=10001(phentrieve)

# Check user inside frontend container
docker exec phentrieve-frontend-1 id
# Output: uid=101(nginx) gid=101(nginx)
```

### 2. Read-Only Root Filesystem

**Configuration:**
```yaml
read_only: true
```

**Why This Matters:**
- Prevents malicious code from modifying the container image
- Makes containers immutable after deployment
- Detects and blocks runtime tampering attempts

**Writable Areas (Explicit tmpfs Mounts):**

API Container:
```yaml
tmpfs:
  - /tmp:uid=10001,gid=10001,mode=1777,size=1G         # Temporary files
  - /app/.cache:uid=10001,gid=10001,mode=0755,size=2G  # Model cache
```

Frontend Container:
```yaml
tmpfs:
  - /tmp:uid=101,gid=101,mode=1777,size=100M           # Nginx temp
  - /var/cache/nginx:uid=101,gid=101,mode=0755,size=50M
  - /var/run:uid=101,gid=101,mode=0755,size=10M
```

**Verification:**
```bash
# Try to create file in read-only area (should fail)
docker exec phentrieve-api-1 touch /app/test.txt
# Error: Read-only file system

# Create file in allowed tmpfs area (should succeed)
docker exec phentrieve-api-1 touch /tmp/test.txt
# Success
```

### 3. Capability Dropping

**Configuration:**
```yaml
cap_drop:
  - ALL  # Drop all Linux capabilities
```

**What Gets Dropped:**

All 38+ Linux capabilities including:
- `CAP_NET_ADMIN` - Network configuration
- `CAP_SYS_ADMIN` - System administration operations
- `CAP_CHOWN` - Ownership changes
- `CAP_DAC_OVERRIDE` - File permission bypassing
- `CAP_SETUID/SETGID` - UID/GID changes
- `CAP_KILL` - Signal sending to arbitrary processes

**Why This Matters:**
- Prevents privilege escalation attacks
- Limits system calls available to compromised containers
- Follows principle of least privilege

**When to Add Capabilities Back:**

Only if absolutely necessary (extremely rare):
```yaml
cap_add:
  - NET_BIND_SERVICE  # Only if binding to ports < 1024
```

**Verification:**
```bash
# Check capabilities inside container
docker exec phentrieve-api-1 capsh --print
# Should show: Current: (empty set)
```

### 4. Security Options

**Configuration:**
```yaml
security_opt:
  - no-new-privileges:true   # Prevent privilege escalation
  - seccomp:unconfined       # May need tuning for ChromaDB
```

**no-new-privileges:**
- Prevents processes from gaining new privileges via `setuid` binaries
- Blocks `sudo`, `su`, and privilege escalation exploits
- Essential for defense-in-depth

**seccomp (Secure Computing Mode):**
- Currently `unconfined` for ChromaDB compatibility
- Can be hardened with custom seccomp profile if needed
- Filters system calls available to container processes

**Future Hardening:**

Create custom seccomp profile:
```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {"names": ["read", "write", "open", "close", ...], "action": "SCMP_ACT_ALLOW"}
  ]
}
```

## Resource Limits

### CPU and Memory Constraints

**API Container:**
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum 4 CPU cores
      memory: 8G       # Maximum 8GB RAM
    reservations:
      cpus: '1.0'      # Minimum 1 CPU core
      memory: 4G       # Minimum 4GB RAM
```

**Frontend Container:**
```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'      # Maximum 1 CPU core
      memory: 512M     # Maximum 512MB RAM
    reservations:
      cpus: '0.25'     # Minimum 0.25 CPU cores
      memory: 128M     # Minimum 128MB RAM
```

**Why This Matters:**
- **DoS Prevention**: Prevents resource exhaustion attacks
- **Multi-Tenancy**: Ensures fair resource sharing on shared infrastructure
- **Predictability**: Guarantees minimum resources for operation
- **Cost Control**: Prevents runaway processes from consuming excessive resources

**Monitoring Resource Usage:**
```bash
# Real-time stats
docker stats phentrieve-api-1 phentrieve-frontend-1

# Check for OOM kills
docker inspect phentrieve-api-1 | grep -A 5 "OOMKilled"
```

## Log Management

### Structured Logging with Rotation

**Configuration:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"    # Maximum 10MB per log file
    max-file: "3"      # Keep 3 rotated files
    labels: "service,env"
```

**Why This Matters:**
- **Disk Exhaustion Prevention**: Logs don't fill up disk
- **Log Retention**: Keeps recent logs while managing space
- **Structured Format**: JSON format for log aggregation systems

**Log Storage:**

Total log storage per container:
- API: 30MB maximum (10MB × 3 files)
- Frontend: 30MB maximum (10MB × 3 files)

**Accessing Logs:**
```bash
# View recent logs
docker logs phentrieve-api-1 --tail 100

# Follow logs in real-time
docker logs -f phentrieve-api-1

# Export logs with timestamp
docker logs phentrieve-api-1 --since 1h > api-logs.txt
```

## Network Isolation

### Network Architecture

**Internal Network:**
```yaml
phentrieve_internal_net:
  driver: bridge
  internal: false  # Allows internet access for model downloads
```

**External Proxy Network (Optional):**
```yaml
networks:
  npm_proxy_network:
    external: true
```

**Why This Matters:**
- **Service Isolation**: Backend not directly exposed to internet
- **Controlled Access**: Only frontend accessible via proxy
- **Defense in Depth**: Network segmentation limits lateral movement

**Network Topology:**
```
Internet
   │
   ├─> npm_proxy_network (external)
   │        │
   │        └─> phentrieve_frontend (nginx)
   │                 │
   └─> phentrieve_internal_net
            │
            └─> phentrieve_api (FastAPI)
```

**Verification:**
```bash
# List networks
docker network ls

# Inspect network
docker network inspect phentrieve_phentrieve_internal_net
```

## Data Volume Security

### Mount Permissions

**Read-Only Data Mount:**
```yaml
volumes:
  - ${PHENTRIEVE_HOST_DATA_DIR}:/phentrieve_data_mount:ro
```

**Selective Read-Write Mounts:**
```yaml
volumes:
  # Only indexes need write access
  - ${PHENTRIEVE_HOST_DATA_DIR}/indexes:/phentrieve_data_mount/indexes:rw
  # Model cache needs write access
  - ${PHENTRIEVE_HOST_HF_CACHE_DIR}:/app/.cache/huggingface:rw
```

**Why This Matters:**
- **Data Integrity**: Core data cannot be corrupted by container processes
- **Principle of Least Privilege**: Write access only where absolutely necessary
- **Audit Trail**: Changes limited to specific directories

**Security Best Practices:**

1. **Separate Data Directory**: Don't mount entire host filesystem
2. **Explicit Permissions**: Always specify `:ro` or `:rw` explicitly
3. **No Sensitive Data**: Don't mount host `/etc`, `/var`, or `/home`
4. **UID/GID Alignment**: Ensure host permissions match container UID 10001

**Verification:**
```bash
# Check mount permissions
docker inspect phentrieve-api-1 | grep -A 10 Mounts

# Verify data directory ownership on host
ls -la ${PHENTRIEVE_HOST_DATA_DIR}
# Should show: drwxr-xr-x ... 10001 10001 ...
```

## Health Checks

### Liveness and Readiness Probes

**API Health Check:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"]
  interval: 30s       # Check every 30 seconds
  timeout: 10s        # 10 second timeout
  retries: 5          # 5 retries before unhealthy
  start_period: 180s  # 3 minute grace period
```

**Frontend Health Check:**
```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:80/health || exit 1"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 30s
```

**Why This Matters:**
- **Automatic Recovery**: Unhealthy containers restarted automatically
- **Load Balancer Integration**: Health checks guide traffic routing
- **Early Detection**: Identifies issues before users notice

**Monitoring Health:**
```bash
# Check health status
docker ps --filter name=phentrieve

# View health logs
docker inspect --format='{{json .State.Health}}' phentrieve-api-1 | jq
```

## Restart Policies

**Configuration:**
```yaml
restart: unless-stopped
```

**Options:**
- `no`: Never restart (not recommended for production)
- `always`: Always restart (can cause boot loops)
- `on-failure`: Restart only on crashes (good for debugging)
- `unless-stopped`: Restart except when explicitly stopped (recommended)

**Why `unless-stopped`:**
- Survives server reboots
- Doesn't restart when manually stopped
- Prevents boot loops during maintenance

## Security Checklist for Production

Before deploying to production, verify:

- [ ] Containers run as non-root users (UID 10001 for API, 101 for frontend)
- [ ] Read-only filesystem enabled (`read_only: true`)
- [ ] All capabilities dropped (`cap_drop: [ALL]`)
- [ ] Security options set (`no-new-privileges:true`)
- [ ] Resource limits configured (CPU, memory)
- [ ] Log rotation enabled (max-size, max-file)
- [ ] Health checks configured and passing
- [ ] Volumes mounted with explicit permissions (`:ro` or `:rw`)
- [ ] No secrets in environment variables (use Docker secrets instead)
- [ ] Latest security patches applied (rebuild images regularly)

## Vulnerability Management

### Regular Security Updates

**Update Schedule:**
1. **Base Images**: Update monthly or when CVEs announced
2. **Dependencies**: `uv lock --upgrade` weekly, test, deploy
3. **Security Patches**: Apply critical patches immediately

**Scanning for Vulnerabilities:**
```bash
# Scan local images with Docker Scout
docker scout cves ghcr.io/berntpopp/phentrieve/api:latest

# Scan with Trivy (more detailed)
trivy image ghcr.io/berntpopp/phentrieve/api:latest

# Only show HIGH and CRITICAL
trivy image --severity HIGH,CRITICAL ghcr.io/berntpopp/phentrieve/api:latest
```

**Automated Scanning:**
- GitHub Dependabot enabled (weekly dependency checks)
- GitHub Actions CI scans on every PR
- GHCR vulnerability scanning on push

### Secrets Management

**NEVER do this:**
```yaml
environment:
  - DATABASE_PASSWORD=secretpassword  # ❌ INSECURE
  - API_KEY=abc123xyz                 # ❌ INSECURE
```

**DO this instead:**
```yaml
secrets:
  - database_password
  - api_key

environment:
  - DATABASE_PASSWORD_FILE=/run/secrets/database_password
```

**Creating Docker Secrets:**
```bash
# Create secret from file
echo "my_secret_value" | docker secret create db_password -

# Use in docker-compose.yml
secrets:
  db_password:
    external: true
```

## Incident Response

### If Container is Compromised

1. **Isolate Immediately:**
   ```bash
   docker network disconnect phentrieve_internal_net phentrieve-api-1
   docker pause phentrieve-api-1
   ```

2. **Capture Forensics:**
   ```bash
   # Export logs
   docker logs phentrieve-api-1 > incident-logs.txt

   # Export filesystem
   docker export phentrieve-api-1 > compromised-container.tar

   # Check processes
   docker top phentrieve-api-1
   ```

3. **Investigate:**
   ```bash
   # Shell into paused container (for analysis only!)
   docker exec -it phentrieve-api-1 /bin/sh

   # Check recent file modifications
   find /app -type f -mtime -1 -ls
   ```

4. **Rebuild and Redeploy:**
   ```bash
   docker-compose down
   docker pull ghcr.io/berntpopp/phentrieve/api:latest
   docker-compose up -d
   ```

5. **Post-Incident:**
   - Review and update security controls
   - Patch vulnerabilities
   - Update incident response plan

## Security Monitoring

### Recommended Tools

**Host-Level:**
- **Falco**: Runtime security monitoring
- **Auditd**: Linux kernel audit logs
- **OSSEC/Wazuh**: Host-based intrusion detection

**Container-Level:**
- **Docker Bench Security**: Automated security audit
- **Clair/Trivy**: Vulnerability scanning
- **Sysdig**: Container forensics and monitoring

**Network-Level:**
- **Zeek**: Network traffic analysis
- **Suricata**: Intrusion detection/prevention

### Audit Logging

Enable Docker daemon audit logging:
```json
{
  "log-level": "info",
  "log-opts": {
    "max-size": "10m",
    "max-file": "5"
  },
  "audit": true
}
```

Monitor events:
```bash
# Docker events
docker events --filter type=container --filter event=start

# Audit logs (if enabled)
journalctl -u docker.service -f
```

## Compliance Considerations

These security measures help satisfy requirements for:

- **ISO 27001**: Information security management
- **SOC 2 Type II**: Security, availability, confidentiality
- **HIPAA**: Healthcare data protection (with additional controls)
- **GDPR**: Data protection and privacy
- **PCI DSS**: Payment card data security (if applicable)

**Note**: Full compliance requires additional organizational and process controls beyond container security.

## Further Reading

- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [NIST Application Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)

!!! warning "Security is a Process, Not a Product"
    These measures provide strong baseline security, but must be combined with:
    - Regular security audits and penetration testing
    - Incident response planning and drills
    - Security awareness training for developers
    - Continuous monitoring and threat intelligence
    - Patch management and vulnerability remediation

!!! tip "Security Updates"
    Subscribe to security advisories for all dependencies:
    - Docker Security: https://docs.docker.com/engine/security/
    - Python Security: https://www.python.org/news/security/
    - FastAPI Security: https://fastapi.tiangolo.com/deployment/security/
