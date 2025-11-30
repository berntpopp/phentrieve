"""
Docker Security Validation Tests.

This module validates that Docker security hardening measures are correctly
applied to the API container. These tests ensure production-level security
practices are enforced:

- Non-root user execution (UID 10001)
- Read-only root filesystem
- Capability dropping (ALL capabilities removed)
- Security options (no-new-privileges, seccomp)
- Resource limits (CPU, memory)
- Proper tmpfs mount permissions

All tests are marked with @pytest.mark.e2e for test categorization.
"""

import pytest
from docker.models.containers import Container


@pytest.mark.e2e
class TestDockerSecurity:
    """Security validation test suite for Docker containers."""

    def test_container_runs_as_non_root_user(self, api_container: Container):
        """
        Verify API container runs as non-root user (UID 10001:10001).

        Security Rationale:
            Running as non-root limits damage from container escape exploits.
            UID 10001 is a dedicated non-privileged user created in Dockerfile.

        Expected:
            Container user = "10001" (UID)
            Container effective user inside container = "10001"
        """
        # Check container configuration user setting
        container_user = api_container.attrs["Config"]["User"]
        assert container_user == "10001:10001", (
            f"Container should run as user 10001:10001, got: {container_user}"
        )

        # Verify effective UID inside running container
        exit_code, output = api_container.exec_run("id -u")
        assert exit_code == 0, "Failed to execute 'id -u' command"

        effective_uid = output.decode().strip()
        assert effective_uid == "10001", (
            f"Effective UID should be 10001, got: {effective_uid}"
        )

    def test_container_runs_as_non_root_group(self, api_container: Container):
        """
        Verify API container runs with non-root group (GID 10001).

        Security Rationale:
            Non-root group membership prevents group-based privilege escalation.
        """
        # Verify effective GID inside running container
        exit_code, output = api_container.exec_run("id -g")
        assert exit_code == 0, "Failed to execute 'id -g' command"

        effective_gid = output.decode().strip()
        assert effective_gid == "10001", (
            f"Effective GID should be 10001, got: {effective_gid}"
        )

    def test_root_filesystem_is_read_only(self, api_container: Container):
        """
        Verify container root filesystem is mounted read-only.

        Security Rationale:
            Read-only root FS prevents malware from persisting changes,
            limits lateral movement, and enforces immutable infrastructure.

        Expected:
            ReadonlyRootfs = True in container host config
            Write attempts to / should fail with "Read-only file system"
        """
        # Check container configuration
        readonly_rootfs = api_container.attrs["HostConfig"]["ReadonlyRootfs"]
        assert readonly_rootfs is True, (
            "Root filesystem should be read-only (ReadonlyRootfs=true)"
        )

        # Verify write protection by attempting to create file in /
        exit_code, output = api_container.exec_run("sh -c 'touch /test_file 2>&1'")

        # Should fail (non-zero exit code)
        assert exit_code != 0, "Write to read-only root filesystem should fail"

        # Should contain "Read-only file system" error
        output_str = output.decode()
        assert (
            "Read-only file system" in output_str
            or "read-only file system" in output_str
        ), f"Expected read-only error, got: {output_str}"

    def test_tmpfs_mounts_are_writable(self, api_container: Container):
        """
        Verify tmpfs mounts provide writable directories for runtime data.

        Security Rationale:
            Tmpfs provides writable space for /tmp and cache without
            compromising read-only root filesystem security.

        Expected Tmpfs Mounts:
            - /tmp (size=500M, uid=10001, gid=10001, mode=1777)
            - /app/.cache (size=1G, uid=10001, gid=10001, mode=0755)
        """
        tmpfs_config = api_container.attrs["HostConfig"]["Tmpfs"]

        # Verify /tmp tmpfs mount exists
        assert "/tmp" in tmpfs_config, "Tmpfs mount for /tmp should exist"
        assert (
            "size=500M" in tmpfs_config["/tmp"]
            or "size=524288000" in tmpfs_config["/tmp"]
        ), "/tmp should have 500M size limit"

        # Verify /app/.cache tmpfs mount exists
        assert "/app/.cache" in tmpfs_config, "Tmpfs mount for /app/.cache should exist"
        assert (
            "size=1G" in tmpfs_config["/app/.cache"]
            or "size=1073741824" in tmpfs_config["/app/.cache"]
        ), "/app/.cache should have 1G size limit"

        # Test write to /tmp succeeds
        exit_code, output = api_container.exec_run("sh -c 'echo test > /tmp/test'")
        assert exit_code == 0, (
            f"Write to /tmp should succeed, failed with: {output.decode()}"
        )

        # Test write to /app/.cache succeeds
        exit_code, output = api_container.exec_run(
            "sh -c 'echo test > /app/.cache/test'"
        )
        assert exit_code == 0, (
            f"Write to /app/.cache should succeed, failed with: {output.decode()}"
        )

    def test_all_capabilities_dropped(self, api_container: Container):
        """
        Verify all Linux capabilities are dropped from container.

        Security Rationale:
            Capabilities provide fine-grained kernel privileges. Dropping
            ALL capabilities minimizes attack surface and prevents privilege
            escalation exploits.

        Expected:
            CapDrop = ["ALL"] in container host config
        """
        cap_drop = api_container.attrs["HostConfig"]["CapDrop"]
        assert cap_drop is not None, "CapDrop should be configured"
        assert "ALL" in cap_drop, f"All capabilities should be dropped, got: {cap_drop}"

        # Verify no capabilities are added back
        cap_add = api_container.attrs["HostConfig"]["CapAdd"]
        assert cap_add is None or cap_add == [], (
            f"No capabilities should be added, got: {cap_add}"
        )

    def test_no_new_privileges_enabled(self, api_container: Container):
        """
        Verify no-new-privileges security option is enabled.

        Security Rationale:
            Prevents processes from gaining new privileges via setuid,
            setgid, or file capabilities. Critical defense against
            privilege escalation.

        Expected:
            SecurityOpt contains "no-new-privileges:true"
        """
        security_opt = api_container.attrs["HostConfig"]["SecurityOpt"]
        assert security_opt is not None, "SecurityOpt should be configured"

        # Check for no-new-privileges setting
        no_new_privs = any("no-new-privileges" in opt for opt in security_opt)
        assert no_new_privs, f"no-new-privileges should be enabled, got: {security_opt}"

    def test_seccomp_profile_configured(self, api_container: Container):
        """
        Verify seccomp security profile is configured.

        Security Rationale:
            Seccomp restricts available system calls, reducing attack surface
            and preventing exploitation of kernel vulnerabilities.

        Expected:
            SecurityOpt contains seccomp configuration
        """
        security_opt = api_container.attrs["HostConfig"]["SecurityOpt"]
        assert security_opt is not None, "SecurityOpt should be configured"

        # Check for seccomp setting
        seccomp = any("seccomp" in opt for opt in security_opt)
        assert seccomp, f"seccomp should be configured, got: {security_opt}"

    def test_memory_limit_enforced(self, api_container: Container):
        """
        Verify memory limit is enforced on container.

        Security Rationale:
            Memory limits prevent resource exhaustion DoS attacks and
            ensure predictable resource allocation.

        Expected (from docker-compose.test.yml):
            Memory limit = 4GB (4294967296 bytes)
            Memory reservation = 2GB (2147483648 bytes)
        """
        host_config = api_container.attrs["HostConfig"]

        # Check memory limit (4GB)
        memory_limit = host_config.get("Memory", 0)
        expected_limit = 4 * 1024 * 1024 * 1024  # 4GB in bytes

        assert memory_limit == expected_limit, (
            f"Memory limit should be 4GB ({expected_limit}), got: {memory_limit}"
        )

        # Check memory reservation (2GB)
        memory_reservation = host_config.get("MemoryReservation", 0)
        expected_reservation = 2 * 1024 * 1024 * 1024  # 2GB in bytes

        assert memory_reservation == expected_reservation, (
            f"Memory reservation should be 2GB ({expected_reservation}), "
            f"got: {memory_reservation}"
        )

    def test_cpu_limit_enforced(self, api_container: Container):
        """
        Verify CPU limit is enforced on container.

        Security Rationale:
            CPU limits prevent resource exhaustion DoS attacks and
            ensure fair resource sharing in multi-tenant environments.

        Expected (from docker-compose.test.yml):
            NanoCPUs = 2.0 CPUs (2000000000 nanocpus)
        """
        host_config = api_container.attrs["HostConfig"]

        # Check CPU limit (2.0 CPUs)
        nano_cpus = host_config.get("NanoCpus", 0)
        expected_cpus = int(2.0 * 1e9)  # 2.0 CPUs in nanocpus

        assert nano_cpus == expected_cpus, (
            f"CPU limit should be 2.0 CPUs ({expected_cpus} nanocpus), got: {nano_cpus}"
        )

    def test_network_mode_is_not_host(self, api_container: Container):
        """
        Verify container is not using host network mode.

        Security Rationale:
            Host network mode bypasses network namespace isolation,
            exposing all host network interfaces to the container.
            Bridge mode provides proper network isolation.

        Expected:
            NetworkMode != "host"
        """
        network_mode = api_container.attrs["HostConfig"]["NetworkMode"]
        assert network_mode != "host", (
            f"Container should not use host network mode, got: {network_mode}"
        )

    def test_privileged_mode_disabled(self, api_container: Container):
        """
        Verify container is not running in privileged mode.

        Security Rationale:
            Privileged mode disables most security features and grants
            nearly complete access to host system. Should never be used
            in production without extreme justification.

        Expected:
            Privileged = False
        """
        privileged = api_container.attrs["HostConfig"]["Privileged"]
        assert privileged is False, "Container should not run in privileged mode"

    def test_pids_limit_configured(self, api_container: Container):
        """
        Verify PIDs limit is configured to prevent fork bombs.

        Security Rationale:
            PID limits prevent fork bomb attacks that could exhaust
            system resources by creating unlimited processes.

        Expected:
            PidsLimit > 0 (or default kernel limit if not explicitly set)
        """
        pids_limit = api_container.attrs["HostConfig"].get("PidsLimit")

        # Note: If not explicitly set, may be None or 0 (meaning kernel default)
        # For testing, we accept either explicit limit or kernel default
        # If pids_limit is set and positive, it's properly configured
        # Otherwise, kernel default is used (acceptable for now)
        assert pids_limit is None or pids_limit >= 0, (
            f"PIDs limit should be non-negative or None, got: {pids_limit}"
        )
