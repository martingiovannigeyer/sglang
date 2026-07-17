#!/usr/bin/env python3
"""Mine recent scheduled PR-test / nightly CI logs and update e2e memory floors.

Parses engine allocation lines from GitHub Actions job logs:

  * ``KV Cache is allocated. ... #tokens: N, KV size: X GB``
    (or ``K size: / V size:``)
  * ``SWAKVPool mem usage: X GB, swa size: N, full size: M``
  * ``Mamba Cache is allocated. max_mamba_cache_size: N, ...``
  * ``DSV4 pool sizes: full=..., swa=..., c4=..., c128=..., ...``

For each ``(suite, test_file)`` pair, averages capacity metrics across recent
runs and writes floors at ``mean * 0.99`` into
``python/sglang/test/memory_thresholds.json``.

Runtime checks use ``GET /server_info`` (see ``sglang.test.memory_threshold``).

Usage:
    # Default: last few successful scheduled pr-test + nightly-nvidia runs
    python3 scripts/ci/utils/update_memory_thresholds.py

    # Dry-run (print summary only)
    python3 scripts/ci/utils/update_memory_thresholds.py --dry-run

    # From pre-downloaded log directories
    python3 scripts/ci/utils/update_memory_thresholds.py --log-dir /tmp/ci_logs

    # Custom run ids
    python3 scripts/ci/utils/update_memory_thresholds.py \\
        --run-id 29458283004 --run-id 29462125107

Requires ``gh`` authenticated against sgl-project/sglang for remote fetch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Allow running without installing the package: import sibling helper.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.test.memory_threshold import (  # noqa: E402
    CAPACITY_FIELDS,
    DEFAULT_FACTOR,
    extract_snapshots_from_log,
    mean_floor,
    normalize_test_file,
    threshold_key,
    thresholds_path,
)

REPO = "sgl-project/sglang"
PR_TEST_WORKFLOW = "pr-test.yml"
NIGHTLY_WORKFLOW = "nightly-test-nvidia.yml"

# GHA log prefix: "job-name\tSTEP\t2026-...\tactual line"
# Match both relative and absolute checkout paths for the test entrypoint.
TEST_START_RE = re.compile(
    r"python3\s+(?:\S*?/)?(?P<path>test/(?:registered|manual)/\S+\.py)"
)
FILENAME_END_RE = re.compile(
    r"filename=['\"]?(?:\S+/)?(?P<path>test/(?:registered|manual)/\S+\.py)"
)
# Prefer the real suite from `run_suite.py --suite <name>` — job display names
# often differ (e.g. job nightly-test-general-1-gpu-h100 runs --suite nightly-1-gpu;
# call-pr-test-extra / extra-a-test-... wraps the suite in the workflow job name).
SUITE_FROM_RUN_SUITE_RE = re.compile(
    r"run_suite\.py\b[^\n]*?--suite\s+(?P<suite>[^\s\\]+)"
)
SUITE_FROM_JOB_RE = re.compile(
    r"(?P<suite>"
    r"(?:base|extra|stage)-[a-z]-[a-z0-9-]+"
    r"|nightly-[a-z0-9-]+"
    r"|per-commit-[a-z0-9-]+"
    r")"
)


@dataclass
class LaunchObservation:
    """One server-launch capacity snapshot from a single CI job run."""

    suite: str
    test_file: str
    run_id: str
    job_id: str
    launch_idx: int
    metrics: Dict[str, float]


@dataclass
class Aggregate:
    samples: List[Dict[str, float]] = field(default_factory=list)

    def add(self, metrics: Dict[str, float]) -> None:
        self.samples.append(dict(metrics))

    def floor(self, factor: float) -> Dict[str, float]:
        by_field: Dict[str, List[float]] = defaultdict(list)
        for s in self.samples:
            for k, v in s.items():
                if k in CAPACITY_FIELDS:
                    by_field[k].append(float(v))
        out: Dict[str, float] = {}
        for k, vals in by_field.items():
            floor = mean_floor(vals, factor=factor)
            # Integer pool sizes / token counts: floor then int()
            if k.endswith("_gb"):
                out[k] = round(floor, 4)
            else:
                out[k] = int(floor)
        return out


def _run(cmd: List[str], *, check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"Command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr}"
        )
    return r.stdout


def list_recent_runs(
    workflow: str,
    *,
    event: Optional[str] = None,
    limit: int = 5,
    branch: str = "main",
) -> List[dict]:
    q = f"repos/{REPO}/actions/workflows/{workflow}/runs?per_page={limit}&branch={branch}"
    if event:
        q += f"&event={event}"
    raw = _run(["gh", "api", q])
    data = json.loads(raw)
    runs = []
    for r in data.get("workflow_runs", []):
        if r.get("status") != "completed":
            continue
        # Keep success and failure — failed runs still have useful memory lines
        # for tests that passed before the failure.
        runs.append(r)
    return runs


def list_jobs(run_id: int | str) -> List[dict]:
    jobs: List[dict] = []
    page = 1
    while True:
        raw = _run(
            [
                "gh",
                "api",
                f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100&page={page}",
            ]
        )
        data = json.loads(raw)
        batch = data.get("jobs", [])
        if not batch:
            break
        jobs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return jobs


def download_job_log(job_id: int | str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    # gh run view --log writes the full annotated log
    r = subprocess.run(
        ["gh", "run", "view", f"--job={job_id}", "--log"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not r.stdout:
        # Fallback: REST API (binary zip-ish text)
        r2 = subprocess.run(
            ["gh", "api", f"repos/{REPO}/actions/jobs/{job_id}/logs"],
            capture_output=True,
            text=True,
        )
        if r2.returncode != 0 or not r2.stdout:
            return False
        dest.write_text(r2.stdout, errors="replace")
        return True
    dest.write_text(r.stdout, errors="replace")
    return True


def suite_from_run_suite_log(text: str) -> Optional[str]:
    """Parse ``run_suite.py --suite <name>`` from a job log (preferred key)."""
    m = SUITE_FROM_RUN_SUITE_RE.search(text)
    if m:
        return m.group("suite").strip()
    return None


def suite_from_job_name(job_name: str) -> str:
    """Fallback suite token from a GHA job display name.

    Prefer :func:`suite_from_run_suite_log` — job names often disagree with
    ``--suite`` (nightly job names, ``call-pr-test-extra / ...`` wrappers).

    Examples when the job *is* the suite:
      'base-b-test-1-gpu-small / base-b-test-1-gpu-small (1)' -> base-b-test-1-gpu-small
    """
    # Prefer the rightmost segment that looks like a suite (handles
    # 'call-pr-test-extra / extra-a-test-1-gpu-small / extra-a-test-1-gpu-small (0)').
    parts = [p.strip() for p in job_name.split(" / ")]
    for part in reversed(parts):
        head = re.sub(r"\s*\(\d+\)\s*$", "", part).strip()
        m = SUITE_FROM_JOB_RE.search(head)
        if m:
            return m.group("suite")
    head = re.sub(r"\s*\(\d+\)\s*$", "", parts[0] if parts else job_name).strip()
    return head or "_unknown_suite"


def parse_job_log(
    text: str,
    *,
    suite: str,
    run_id: str,
    job_id: str,
) -> List[LaunchObservation]:
    """Associate memory snapshots with the currently running test file."""
    # Override the caller-provided suite if the log records run_suite --suite.
    suite = suite_from_run_suite_log(text) or suite

    current_test: Optional[str] = None
    # Buffer raw log text per test file, then extract snapshots once the next
    # test starts (or at EOF). This keeps multi-line / multi-TP launches ordered.
    buffers: Dict[str, List[str]] = defaultdict(list)
    test_order: List[str] = []

    for line in text.splitlines():
        m = TEST_START_RE.search(line)
        if m:
            current_test = normalize_test_file(m.group("path"))
            if current_test not in buffers:
                test_order.append(current_test)
            continue
        # Some runners only print filename=... at the end; still useful as a
        # backstop if we missed the python3 line (shouldn't happen often).
        if current_test is None:
            m2 = FILENAME_END_RE.search(line)
            if m2:
                current_test = normalize_test_file(m2.group("path"))
                if current_test not in buffers:
                    test_order.append(current_test)
        if current_test is not None:
            buffers[current_test].append(line)

    observations: List[LaunchObservation] = []
    for test_file in test_order:
        body = "\n".join(buffers[test_file])
        snaps = extract_snapshots_from_log(body)
        for idx, snap in enumerate(snaps):
            if not snap:
                continue
            observations.append(
                LaunchObservation(
                    suite=suite,
                    test_file=test_file,
                    run_id=str(run_id),
                    job_id=str(job_id),
                    launch_idx=idx,
                    metrics=snap,
                )
            )
    return observations


def group_observations(
    obs: Sequence[LaunchObservation],
) -> Dict[Tuple[str, str], Dict[int, Aggregate]]:
    """(suite, test_file) -> launch_idx -> Aggregate."""
    grouped: Dict[Tuple[str, str], Dict[int, Aggregate]] = defaultdict(
        lambda: defaultdict(Aggregate)
    )
    for o in obs:
        grouped[(o.suite, o.test_file)][o.launch_idx].add(o.metrics)
    return grouped


def build_thresholds(
    grouped: Dict[Tuple[str, str], Dict[int, Aggregate]],
    *,
    factor: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "_meta": {
            "factor": factor,
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fields": list(CAPACITY_FIELDS),
            "note": (
                "Floors are mean(samples)*factor. Runtime check uses GET /server_info "
                "and requires observed >= floor for each present field."
            ),
        }
    }
    for (suite, test_file), by_idx in sorted(grouped.items()):
        launches: List[Dict[str, float]] = []
        sample_counts: List[int] = []
        for idx in sorted(by_idx.keys()):
            agg = by_idx[idx]
            floor = agg.floor(factor=factor)
            if not floor:
                continue
            launches.append(floor)
            sample_counts.append(len(agg.samples))
        if not launches:
            continue
        key = threshold_key(suite, test_file)
        out[key] = {
            "suite": suite,
            "test_file": test_file,
            "launches": launches,
            "sample_counts": sample_counts,
        }
    return out


def collect_from_log_dir(log_dir: Path) -> List[LaunchObservation]:
    obs: List[LaunchObservation] = []
    for path in sorted(log_dir.rglob("*.txt")):
        # Expect filename like job_<id>.txt or suite__job_<id>.txt
        text = path.read_text(errors="replace")
        # Infer suite from first line job name if present
        first = text.splitlines()[0] if text else ""
        # GHA format: "job name\tSTEP\ttimestamp\tline"
        job_name = first.split("\t")[0] if "\t" in first else path.stem
        suite = suite_from_job_name(job_name)
        job_id = path.stem
        obs.extend(parse_job_log(text, suite=suite, run_id="local", job_id=job_id))
    return obs


def collect_from_runs(
    run_ids: Sequence[str],
    *,
    cache_dir: Path,
    max_jobs_per_run: Optional[int] = None,
    job_name_filter: Optional[str] = None,
) -> List[LaunchObservation]:
    obs: List[LaunchObservation] = []
    for run_id in run_ids:
        print(f"Listing jobs for run {run_id}...", flush=True)
        jobs = list_jobs(run_id)
        gpu_jobs = [
            j
            for j in jobs
            if "gpu" in j.get("name", "").lower()
            or "nightly" in j.get("name", "").lower()
        ]
        if job_name_filter:
            gpu_jobs = [j for j in gpu_jobs if job_name_filter in j.get("name", "")]
        if max_jobs_per_run is not None:
            gpu_jobs = gpu_jobs[:max_jobs_per_run]
        print(f"  {len(gpu_jobs)} jobs to download", flush=True)
        for j in gpu_jobs:
            jid = j["id"]
            name = j.get("name", "")
            suite = suite_from_job_name(name)
            dest = cache_dir / f"run_{run_id}" / f"job_{jid}.txt"
            print(f"  downloading job {jid} ({name})...", flush=True)
            if not download_job_log(jid, dest):
                print(f"    FAILED to download job {jid}", flush=True)
                continue
            text = dest.read_text(errors="replace")
            n_before = len(obs)
            obs.extend(
                parse_job_log(text, suite=suite, run_id=str(run_id), job_id=str(jid))
            )
            print(f"    +{len(obs) - n_before} launch snapshots", flush=True)
    return obs


def resolve_default_run_ids(limit: int) -> List[str]:
    run_ids: List[str] = []
    print("Fetching recent scheduled pr-test runs...", flush=True)
    for r in list_recent_runs(PR_TEST_WORKFLOW, event="schedule", limit=limit):
        run_ids.append(str(r["id"]))
        print(f"  pr-test {r['id']} {r.get('created_at')} {r.get('conclusion')}")
    print("Fetching recent nightly-test-nvidia runs...", flush=True)
    for r in list_recent_runs(NIGHTLY_WORKFLOW, limit=limit):
        if r.get("head_branch") and r["head_branch"] != "main":
            continue
        run_ids.append(str(r["id"]))
        print(f"  nightly {r['id']} {r.get('created_at')} {r.get('conclusion')}")
    return run_ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="GitHub Actions run id (repeatable). Default: recent scheduled/nightly.",
    )
    p.add_argument(
        "--limit-runs",
        type=int,
        default=3,
        help="How many recent runs per workflow when --run-id is omitted (default 3).",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Parse pre-downloaded *.txt logs instead of fetching from GitHub.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "sglang_memory_threshold_logs",
        help="Where to cache downloaded job logs.",
    )
    p.add_argument(
        "--factor",
        type=float,
        default=DEFAULT_FACTOR,
        help=f"Floor = mean * factor (default {DEFAULT_FACTOR}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output JSON path (default: {thresholds_path()}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary and do not write the thresholds file.",
    )
    p.add_argument(
        "--max-jobs-per-run",
        type=int,
        default=None,
        help="Debug: cap jobs downloaded per run.",
    )
    p.add_argument(
        "--job-name-filter",
        type=str,
        default=None,
        help="Only download jobs whose name contains this substring.",
    )
    args = p.parse_args(argv)

    if args.log_dir:
        observations = collect_from_log_dir(args.log_dir)
    else:
        import shutil

        if not shutil.which("gh"):
            print(
                "Error: the 'gh' (GitHub) CLI is required but was not found in PATH.\n"
                "Install it and run 'gh auth login' to authenticate.",
                file=sys.stderr,
            )
            return 1
        run_ids = args.run_id or resolve_default_run_ids(args.limit_runs)
        if not run_ids:
            print("No runs found.", file=sys.stderr)
            return 1
        observations = collect_from_runs(
            run_ids,
            cache_dir=args.cache_dir,
            max_jobs_per_run=args.max_jobs_per_run,
            job_name_filter=args.job_name_filter,
        )

    print(f"Collected {len(observations)} launch observations", flush=True)
    if not observations:
        print("Nothing to write.", file=sys.stderr)
        return 1

    grouped = group_observations(observations)
    thresholds = build_thresholds(grouped, factor=args.factor)

    n_keys = sum(1 for k in thresholds if not k.startswith("_"))
    print(f"Built thresholds for {n_keys} suite::test keys", flush=True)

    # Show a short sample
    shown = 0
    for k, v in thresholds.items():
        if k.startswith("_"):
            continue
        print(f"  {k}: {len(v['launches'])} launch(s), samples={v['sample_counts']}")
        for i, launch in enumerate(v["launches"][:3]):
            print(f"    [{i}] {launch}")
        shown += 1
        if shown >= 15:
            print("  ...")
            break

    out_path = args.output or thresholds_path()
    if args.dry_run:
        print(f"[dry-run] would write {out_path}")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(thresholds, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
