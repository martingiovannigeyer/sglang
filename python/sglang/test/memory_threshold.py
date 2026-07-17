"""E2E memory-capacity thresholds for CI unittests.

Runtime path (preferred): after a server is healthy, ``GET /server_info`` and
compare capacity fields against floors in ``memory_thresholds.json``.

Offline path: ``scripts/ci/utils/update_memory_thresholds.py`` mines scheduled
PR-test / nightly logs (KV / SWA / Mamba / DSV4 allocation lines), averages
recent values, and writes floors at ``mean * 0.99``.

Threshold key: ``{suite}::{test_file}`` where ``suite`` comes from
``SGLANG_TEST_SUITE`` (set by ``run_unittest_files``) and ``test_file`` is the
repo-relative path of the running test module.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

THRESHOLDS_FILENAME = "memory_thresholds.json"
DEFAULT_FACTOR = 0.99

# Capacity fields: higher is better (more tokens / larger usable pools).
# Observed value must be >= floor.
CAPACITY_FIELDS = (
    "token_capacity",  # max_total_num_tokens / #tokens
    "kv_cache_gb",  # allocated KV pool GB
    "swa_size",
    "full_size",
    "swa_mem_gb",
    "mamba_cache_size",
    "mamba_conv_gb",
    "mamba_ssm_gb",
    "dsv4_full",
    "dsv4_swa",
    "dsv4_c4",
    "dsv4_c128",
    "dsv4_c4_state",
    "dsv4_c128_state",
)

# ---- log line parsers (shared with the update script) ----

KV_RE = re.compile(
    r"KV Cache is allocated\.\s*dtype:\s*(?P<dtype>\S+),\s*#tokens:\s*(?P<tokens>\d+),\s*"
    r"(?:KV size:\s*(?P<kv_size>[\d.]+)\s*GB|"
    r"K size:\s*(?P<k_size>[\d.]+)\s*GB,\s*V size:\s*(?P<v_size>[\d.]+)\s*GB)"
)
SWA_RE = re.compile(
    r"SWAKVPool mem usage:\s*(?P<mem>[\d.]+)\s*GB,\s*"
    r"swa size:\s*(?P<swa>\d+),\s*full size:\s*(?P<full>\d+)"
)
MAMBA_RE = re.compile(
    r"Mamba Cache is allocated\.\s*max_mamba_cache_size:\s*(?P<mamba>\d+),\s*"
    r"conv_state size:\s*(?P<conv>[\d.]+)\s*GB,?\s*"
    r"ssm_state size:\s*(?P<ssm>[\d.]+)\s*GB"
)
DSV4_RE = re.compile(
    r"DSV4 pool sizes:\s*full=(?P<full>\d+),\s*swa=(?P<swa>\d+),\s*"
    r"c4=(?P<c4>\d+),\s*c128=(?P<c128>\d+),\s*"
    r"c4_state=(?P<c4_state>\d+),\s*c128_state=(?P<c128_state>\d+)"
)

_lock = threading.Lock()
_launch_counters: Dict[str, int] = {}
_thresholds_cache: Optional[Dict[str, Any]] = None


def thresholds_path() -> Path:
    return Path(__file__).resolve().parent / THRESHOLDS_FILENAME


def load_thresholds() -> Dict[str, Any]:
    global _thresholds_cache
    if _thresholds_cache is not None:
        return _thresholds_cache
    path = thresholds_path()
    if not path.is_file():
        _thresholds_cache = {}
        return _thresholds_cache
    with path.open() as f:
        _thresholds_cache = json.load(f)
    return _thresholds_cache


def reload_thresholds() -> Dict[str, Any]:
    """Force-reload thresholds (used by the update script / tests)."""
    global _thresholds_cache
    _thresholds_cache = None
    return load_thresholds()


def parse_memory_log_line(line: str) -> Optional[Dict[str, float]]:
    """Parse one engine log line into a partial capacity snapshot."""
    m = KV_RE.search(line)
    if m:
        tokens = int(m.group("tokens"))
        if m.group("kv_size") is not None:
            kv_gb = float(m.group("kv_size"))
        else:
            kv_gb = float(m.group("k_size")) + float(m.group("v_size"))
        return {"token_capacity": tokens, "kv_cache_gb": kv_gb}

    m = SWA_RE.search(line)
    if m:
        return {
            "swa_mem_gb": float(m.group("mem")),
            "swa_size": int(m.group("swa")),
            "full_size": int(m.group("full")),
            # Prefer full pool as the primary capacity for SWA hybrids.
            "token_capacity": int(m.group("full")),
        }

    m = MAMBA_RE.search(line)
    if m:
        return {
            "mamba_cache_size": int(m.group("mamba")),
            "mamba_conv_gb": float(m.group("conv")),
            "mamba_ssm_gb": float(m.group("ssm")),
        }

    m = DSV4_RE.search(line)
    if m:
        return {
            "dsv4_full": int(m.group("full")),
            "dsv4_swa": int(m.group("swa")),
            "dsv4_c4": int(m.group("c4")),
            "dsv4_c128": int(m.group("c128")),
            "dsv4_c4_state": int(m.group("c4_state")),
            "dsv4_c128_state": int(m.group("c128_state")),
            "token_capacity": int(m.group("full")),
        }

    return None


def _fingerprint(snap: Dict[str, float]) -> tuple:
    """Stable fingerprint for deduping multi-TP identical log lines."""
    return tuple(sorted((k, snap[k]) for k in CAPACITY_FIELDS if k in snap))


def extract_snapshots_from_log(text: str) -> List[Dict[str, float]]:
    """Extract ordered capacity snapshots from engine log text.

    Multi-TP ranks emit identical allocation lines; consecutive identical
    fingerprints are collapsed. Related lines from a single server start
    (Mamba + KV, or SWA full/swa sub-pool KV lines + SWAKVPool summary) are
    merged so each snapshot approximates one ``GET /server_info`` sample.
    """
    raw: List[Dict[str, float]] = []
    for line in text.splitlines():
        snap = parse_memory_log_line(line)
        if snap is None:
            continue
        if raw and _fingerprint(snap) == _fingerprint(raw[-1]):
            continue  # TP duplicate
        # Merge into previous if non-conflicting (same server start)
        if raw and _can_merge(raw[-1], snap):
            raw[-1] = {**raw[-1], **snap}
        else:
            raw.append(dict(snap))
    return _collapse_hybrid_subpools(raw)


def _can_merge(a: Dict[str, float], b: Dict[str, float]) -> bool:
    # Two pure-KV snaps with different sizes are distinct server launches
    # (e.g. EAGLE draft vs target). Only collapse exact TP duplicates.
    if _is_kv_only(a) and _is_kv_only(b):
        return _fingerprint(a) == _fingerprint(b)
    for k in b:
        if k in a and a[k] != b[k]:
            # token_capacity / kv_cache_gb differ across SWA full vs swa
            # sub-pools and across Mamba+KV partials; allow merge.
            if k in ("token_capacity", "kv_cache_gb"):
                continue
            return False
    return True


def _is_kv_only(snap: Dict[str, float]) -> bool:
    return set(snap.keys()).issubset({"token_capacity", "kv_cache_gb"})


def _collapse_hybrid_subpools(
    snaps: List[Dict[str, float]],
) -> List[Dict[str, float]]:
    """Drop pure-KV snaps that are SWA sub-pools of a following SWAKVPool line.

    Hybrid SWA logs emit two ``KV Cache is allocated`` lines (swa + full) plus
    one ``SWAKVPool mem usage`` summary for a *single* server process. Runtime
    checks only see one /server_info snapshot, so floors must match that.
    """
    if not snaps:
        return snaps
    out: List[Dict[str, float]] = []
    for snap in snaps:
        if "swa_size" in snap or "full_size" in snap:
            swa = snap.get("swa_size")
            full = snap.get("full_size") or snap.get("token_capacity")
            kept: List[Dict[str, float]] = []
            for prev in out:
                if not _is_kv_only(prev):
                    kept.append(prev)
                    continue
                tc = prev.get("token_capacity")
                if tc is not None and tc in (swa, full):
                    # Fold kv_cache_gb into the hybrid snap (prefer larger).
                    if "kv_cache_gb" in prev:
                        snap["kv_cache_gb"] = max(
                            float(snap.get("kv_cache_gb", 0.0)),
                            float(prev["kv_cache_gb"]),
                        )
                    continue
                kept.append(prev)
            out = kept
        out.append(snap)
    return out


def snapshot_from_server_info(info: Dict[str, Any]) -> Dict[str, float]:
    """Build a capacity snapshot from a ``/server_info`` JSON response."""
    snap: Dict[str, float] = {}

    if "max_total_num_tokens" in info and info["max_total_num_tokens"] is not None:
        snap["token_capacity"] = int(info["max_total_num_tokens"])

    mem = None
    internal = info.get("internal_states")
    if isinstance(internal, list) and internal:
        mem = internal[0].get("memory_usage")
    if not isinstance(mem, dict):
        mem = info.get("memory_usage")
    if not isinstance(mem, dict):
        mem = {}

    if "token_capacity" in mem and mem["token_capacity"] is not None:
        snap["token_capacity"] = int(mem["token_capacity"])
    if "kvcache" in mem and mem["kvcache"] is not None:
        snap["kv_cache_gb"] = float(mem["kvcache"])

    # Optional richer fields (populated when the server exposes them).
    int_fields = (
        "swa_size",
        "full_size",
        "mamba_cache_size",
        "dsv4_full",
        "dsv4_swa",
        "dsv4_c4",
        "dsv4_c128",
        "dsv4_c4_state",
        "dsv4_c128_state",
    )
    float_fields = (
        "swa_mem_gb",
        "mamba_conv_gb",
        "mamba_ssm_gb",
    )
    for field in int_fields:
        if field in mem and mem[field] is not None:
            snap[field] = int(mem[field])
    for field in float_fields:
        if field in mem and mem[field] is not None:
            snap[field] = float(mem[field])

    return snap


def fetch_server_memory_snapshot(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, float]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.get(
        f"{base_url.rstrip('/')}/server_info",
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()
    return snapshot_from_server_info(resp.json())


def normalize_test_file(path: str) -> str:
    """Strip absolute CI checkout prefixes down to ``test/...`` or similar."""
    path = path.replace("\\", "/")
    markers = (
        "/sglang/test/",
        "/test/registered/",
        "/test/manual/",
        "test/registered/",
        "test/manual/",
        "test/",
    )
    for marker in markers:
        idx = path.find(marker)
        if idx >= 0:
            # Keep from "test/"
            if marker.startswith("/"):
                return (
                    path[idx + 1 :]
                    if path[idx + 1 :].startswith("test/")
                    else path[idx + len("/sglang/") :]
                )
            return path[idx:]
    # Fallback: basename under a best-effort relative path
    if path.startswith("test/"):
        return path
    return path.lstrip("./")


def current_test_file() -> Optional[str]:
    env = os.environ.get("SGLANG_TEST_FILE")
    if env:
        return normalize_test_file(env)
    if sys.argv and sys.argv[0]:
        return normalize_test_file(os.path.abspath(sys.argv[0]))
    return None


def current_suite() -> Optional[str]:
    return os.environ.get("SGLANG_TEST_SUITE") or None


def threshold_key(suite: Optional[str], test_file: str) -> str:
    suite = suite or "_unknown_suite"
    return f"{suite}::{test_file}"


def mean_floor(values: Sequence[float], factor: float = DEFAULT_FACTOR) -> float:
    if not values:
        raise ValueError("mean_floor requires non-empty values")
    return sum(values) / len(values) * factor


def check_snapshot_against_floor(
    observed: Dict[str, float],
    floor: Dict[str, float],
    *,
    key: str,
    launch_idx: int,
) -> List[str]:
    """Return a list of failure messages (empty if OK)."""
    failures: List[str] = []
    for field in CAPACITY_FIELDS:
        if field not in floor:
            continue
        if field not in observed:
            # Floor has a field the live server did not report — skip rather
            # than fail (e.g. SWA fields before server_info enrichment lands).
            logger.warning(
                "Memory threshold %s launch[%d]: floor has %s=%.4g but server "
                "did not report it; skipping field",
                key,
                launch_idx,
                field,
                floor[field],
            )
            continue
        obs = float(observed[field])
        thr = float(floor[field])
        if obs < thr:
            failures.append(
                f"{field}: observed={obs:g} < floor={thr:g} "
                f"(key={key}, launch={launch_idx})"
            )
    return failures


def _next_launch_index(key: str) -> int:
    with _lock:
        idx = _launch_counters.get(key, 0)
        _launch_counters[key] = idx + 1
        return idx


def reset_launch_counters() -> None:
    """Test helper: clear per-key launch counters."""
    with _lock:
        _launch_counters.clear()


def memory_threshold_check_enabled() -> bool:
    """Enabled in CI by default; opt-in locally via SGLANG_CHECK_MEMORY_THRESHOLDS=1."""
    if os.environ.get("SGLANG_CHECK_MEMORY_THRESHOLDS", "").lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False
    if os.environ.get("SGLANG_CHECK_MEMORY_THRESHOLDS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() in ("1", "true", "yes")


def maybe_check_server_memory(
    base_url: str,
    *,
    api_key: Optional[str] = None,
    test_file: Optional[str] = None,
    suite: Optional[str] = None,
) -> None:
    """Fetch /server_info and assert capacity >= stored floors.

    No-op when disabled, when the threshold file has no entry for this test,
    or when the server cannot be queried. Raises ``AssertionError`` on
    regression.
    """
    if not memory_threshold_check_enabled():
        return

    test_file = test_file or current_test_file()
    suite = suite if suite is not None else current_suite()
    if not test_file:
        return

    thresholds = load_thresholds()
    key = threshold_key(suite, test_file)
    entry = thresholds.get(key)
    if entry is None:
        # Also try without suite (legacy / single-key entries).
        entry = thresholds.get(test_file)
        if entry is None:
            return
        key = test_file

    launches: List[Dict[str, float]] = entry.get("launches") or []
    if not launches:
        return

    launch_idx = _next_launch_index(key)
    if launch_idx >= len(launches):
        # Extra launches beyond recorded sequence — ignore.
        logger.info(
            "Memory threshold %s: launch[%d] beyond recorded %d; skipping",
            key,
            launch_idx,
            len(launches),
        )
        return

    floor = launches[launch_idx]
    try:
        observed = fetch_server_memory_snapshot(base_url, api_key=api_key)
    except Exception as e:
        logger.warning(
            "Memory threshold check skipped for %s launch[%d]: failed to query "
            "/server_info (%s)",
            key,
            launch_idx,
            e,
        )
        return

    if not observed:
        logger.warning(
            "Memory threshold check skipped for %s launch[%d]: empty snapshot",
            key,
            launch_idx,
        )
        return

    logger.info(
        "Memory threshold check %s launch[%d]: observed=%s floor=%s",
        key,
        launch_idx,
        observed,
        floor,
    )
    failures = check_snapshot_against_floor(
        observed, floor, key=key, launch_idx=launch_idx
    )
    if failures:
        raise AssertionError(
            "Memory capacity regression detected:\n  "
            + "\n  ".join(failures)
            + "\nRe-run scripts/ci/utils/update_memory_thresholds.py after an "
            "intentional memory optimization."
        )
