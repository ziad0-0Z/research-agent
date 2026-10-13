"""URL safety for anything the agent fetches (code-orchestrated, never the LLM).

Blocks the classic SSRF targets: local files, localhost, private/internal
networks, cloud metadata endpoints, odd ports and credentials in URLs.
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home", ".corp")
MAX_URL_LENGTH = 2048


@dataclass(frozen=True)
class UrlCheck:
    allowed: bool
    reason: str
    host: str = ""


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local  # includes 169.254.169.254 cloud metadata
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url(url: str, *, resolve_dns: bool = False) -> UrlCheck:
    """Decide whether a URL may be fetched.

    resolve_dns=True also resolves the hostname and blocks names that point at
    private addresses (used in live mode; tests keep it off to stay offline).
    """
    if not isinstance(url, str) or not url.strip():
        return UrlCheck(False, "empty URL")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        return UrlCheck(False, "URL too long")

    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return UrlCheck(False, "malformed URL")

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return UrlCheck(False, f"scheme '{parts.scheme or 'none'}' not allowed (only http/https)")
    if parts.username or parts.password:
        return UrlCheck(False, "credentials in URL not allowed")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return UrlCheck(False, "missing host")
    if port not in ALLOWED_PORTS:
        return UrlCheck(False, f"port {port} not allowed", host)
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        return UrlCheck(False, f"internal host '{host}' not allowed", host)

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if _ip_is_blocked(ip):
            return UrlCheck(False, f"private or reserved address {ip} not allowed", host)
        return UrlCheck(True, "ok", host)

    if "." not in host:
        return UrlCheck(False, f"single-label host '{host}' not allowed", host)

    if resolve_dns:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return UrlCheck(False, f"host '{host}' does not resolve", host)
        for info in infos:
            resolved = ipaddress.ip_address(info[4][0])
            if _ip_is_blocked(resolved):
                return UrlCheck(False, f"host '{host}' resolves to private address {resolved}", host)

    return UrlCheck(True, "ok", host)
