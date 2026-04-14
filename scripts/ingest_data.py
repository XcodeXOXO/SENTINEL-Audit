"""
scripts/ingest_data.py
======================
Phase 2: Instruction Data Engineering for Project Sentinel.

ETL pipeline that extracts smart contract vulnerability data from two sources
(in priority order), transforms it into Chain-of-Thought Alpaca records, and
loads it into data/fine_tuning/dataset.jsonl.

Source Priority:
  1. GitHub REST API  — SWC Registry (smart-contract-weakness-classification)
  2. HuggingFace Datasets API — BCCC-VulSCs / SmartBugs labeled pairs

Idempotency: Each record is SHA-256 hashed (on the raw Solidity input).
Duplicate hashes are skipped so re-runs never corrupt the dataset.

Usage:
    python scripts/ingest_data.py [--hf] [--github] [--all]

    --github   Pull SWC Registry entries from GitHub (default if no flag given)
    --hf       Pull HuggingFace labeled datasets
    --all      Run both sources
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Generator, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. LOAD ENVIRONMENT (Must happen before os.getenv)
# ---------------------------------------------------------------------------
load_dotenv(override=True) 

# 2. INITIALIZE TOKENS
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 3. DIAGNOSTIC PRINTS (Helps you verify connectivity)
if GITHUB_TOKEN:
    print(f"GITHUB_TOKEN detected: {GITHUB_TOKEN[:4]}****")
else:
    print("GITHUB_TOKEN not found in environment!")

if HF_TOKEN:
    print(f"HF_TOKEN detected: {HF_TOKEN[:3]}****")
else:
    print("HF_TOKEN not found in environment!")

# ---------------------------------------------------------------------------
# 4. BOOTSTRAP LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sentinel.ingest")

# ---------------------------------------------------------------------------
# 5. PATH CONSTANTS
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent
RAW_DIR         = ROOT_DIR / "data" / "raw"
FINE_TUNE_DIR   = ROOT_DIR / "data" / "fine_tuning"
DATASET_PATH    = FINE_TUNE_DIR / "dataset.jsonl"
HASH_INDEX_PATH = FINE_TUNE_DIR / ".hash_index.json"

# ---------------------------------------------------------------------------
# 6. REGISTRY CONFIG
# ---------------------------------------------------------------------------
GITHUB_API_BASE  = "https://api.github.com"
SWC_REPO_OWNER   = "SmartContractSecurity"
SWC_REPO_NAME    = "SWC-registry"
SWC_ENTRIES_PATH = "entries/docs"

# Updated for Phase 2 scaling with stable mirrors
# Updated for Phase 2 scaling with stable, official mirrors
HF_DATASETS = [
    {
        "repo_id": "smartbugs/smartbugs-curated", # Official SmartBugs Repo
        "split": "train",
        "sol_column": "source_code",
        "label_column": "vulnerability",
        "vuln_label": "1",
    },
    {
        "repo_id": "mwritescode/slither-audited-smart-contracts", # Official Slither Repo
        "split": "train",
        "sol_column": "source_code",
        "label_column": "slither",
        "vuln_label": None,
    },
]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    """Create required directories if they do not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FINE_TUNE_DIR.mkdir(parents=True, exist_ok=True)


def _load_hash_index() -> set[str]:
    """Load the SHA-256 deduplication index from disk."""
    if HASH_INDEX_PATH.exists():
        with HASH_INDEX_PATH.open("r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_hash_index(index: set[str]) -> None:
    """Persist the deduplication index atomically."""
    with HASH_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(sorted(index), f, indent=2)


def _sha256(text: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _github_headers() -> dict[str, str]:
    """
    Build GitHub API request headers.
    Only injects Authorization when GITHUB_TOKEN is a non-empty string;
    sending an empty Bearer value causes GitHub to return 401 on public repos.
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = GITHUB_TOKEN.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(url: str, params: Optional[dict] = None) -> dict | list:
    """
    Thin wrapper around requests.get for the GitHub API.

    - Handles rate-limit (403/429) by sleeping until the reset window.
    - On 401 (expired/revoked token), retries the request without authentication
      so the pipeline continues against public repos at the unauthenticated rate.
    """
    resp = requests.get(url, headers=_github_headers(), params=params, timeout=30)

    # Graceful fallback: token exists but is invalid — retry without auth
    if resp.status_code == 401 and GITHUB_TOKEN.strip():
        log.warning(
            "GitHub returned 401 with stored token (expired/revoked?). "
            "Retrying anonymously — rate limit will be 60 req/hr."
        )
        anon_headers = {
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        resp = requests.get(url, headers=anon_headers, params=params, timeout=30)

    if resp.status_code in (403, 429):
        reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset_ts - int(time.time()), 5)
        log.warning("GitHub rate limit hit. Sleeping %ds …", wait)
        time.sleep(wait)
        resp = requests.get(url, headers=_github_headers(), params=params, timeout=30)

    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Solidity Cleaner
# ---------------------------------------------------------------------------

_MD_FENCE_RE   = re.compile(r"```[a-zA-Z]*\n?", re.MULTILINE)
_PRAGMA_STRIP  = re.compile(r"//\s*SPDX-License-Identifier:[^\n]*\n?")
_BLANK_LINES   = re.compile(r"\n{3,}")


def clean_solidity(raw: str) -> str:
    """
    Strip non-code markdown artifacts from a raw Solidity snippet so the
    result is a clean, AST-parseable source string.

    Steps:
      1. Remove opening/closing markdown code fences.
      2. Strip HTML comment blocks that some registry entries embed.
      3. Collapse excessive blank lines to a maximum of two.
      4. Strip leading/trailing whitespace.
    """
    # Remove ``` fences
    cleaned = _MD_FENCE_RE.sub("", raw)
    # Remove HTML-style comments (<!-- ... -->)
    cleaned = re.sub(r"<!--.*?-->", "", cleaned, flags=re.DOTALL)
    # Collapse 3+ blank lines → 2
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()


def _is_valid_solidity(text: str) -> bool:
    """
    Lightweight heuristic to confirm a string contains Solidity code.
    Avoids adding pure-markdown or empty entries to the dataset.
    """
    return bool(
        text
        and ("contract " in text or "library " in text or "interface " in text)
        and "pragma solidity" in text
    )


# ---------------------------------------------------------------------------
# Chain-of-Thought output builder
# ---------------------------------------------------------------------------

def build_cot_output(
    swc_id: str,
    title: str,
    description: str,
    remediation: str,
    vulnerable_code: str,
    fixed_code: str,
) -> str:
    """
    Build a Chain-of-Thought audit output in the style of a senior auditor.

    Structure mirrors the Checks-Effects-Interactions mental model and follows
    the SWC entry format so the fine-tuned model learns registry-grounded
    reasoning rather than hallucinated patterns.
    """
    cot = f"""## Vulnerability Identified: {title} ({swc_id})

### Step 1 — Identification
{description.strip()}

The vulnerability class maps to **{swc_id}** in the Smart Contract Weakness Classification Registry.

### Step 2 — Invariant Analysis
"""

    # Heuristic: tag common invariant violations based on SWC ID prefix
    if swc_id in ("SWC-107",):
        cot += (
            "The **Checks-Effects-Interactions (CEI)** invariant is violated. "
            "The contract transfers ETH (interaction) before updating its internal "
            "accounting state (effect), allowing a re-entrant call to drain funds."
        )
    elif swc_id in ("SWC-105", "SWC-106"):
        cot += (
            "The **access-control invariant** is violated. Privileged state-changing "
            "functions lack `onlyOwner` or equivalent guards, enabling unauthorized actors "
            "to invoke them."
        )
    elif swc_id in ("SWC-101",):
        cot += (
            "The **arithmetic safety invariant** is violated. Integer overflow/underflow "
            "can silently wrap around without reverting prior to Solidity 0.8.x, corrupting "
            "token balances or counters."
        )
    else:
        cot += (
            f"Refer to the {swc_id} registry entry for the specific EVM invariant violated. "
            "The root cause typically involves unchecked external interactions, missing guards, "
            "or unsafe type assumptions."
        )

    cot += f"""

### Step 3 — Remediation

{remediation.strip()}

**Fixed Code:**
```solidity
{fixed_code.strip()}
```
"""
    return cot


def build_alpaca_record(
    *,
    swc_id: str,
    title: str,
    description: str,
    remediation: str,
    vulnerable_code: str,
    fixed_code: str,
    vuln_class: Optional[str] = None,
) -> dict[str, str]:
    """
    Assemble a single Alpaca-format instruction record.

    Fields:
      instruction — Persona-wrapped audit directive.
      input       — Cleaned Solidity source (vulnerable contract).
      output      — CoT breakdown with identification, invariant analysis, fix.
    """
    vuln_hint = f", specifically checking for {vuln_class}" if vuln_class else ""
    instruction = (
        f"You are an expert Solidity auditor. Analyze the following smart contract "
        f"for security vulnerabilities{vuln_hint}. "
        f"Identify the root cause, explain the EVM invariant violated, and provide "
        f"a corrected version of the vulnerable code."
    )
    return {
        "instruction": instruction,
        "input": clean_solidity(vulnerable_code),
        "output": build_cot_output(
            swc_id=swc_id,
            title=title,
            description=description,
            remediation=remediation,
            vulnerable_code=vulnerable_code,
            fixed_code=fixed_code,
        ),
    }


# ---------------------------------------------------------------------------
# JSONL Writer
# ---------------------------------------------------------------------------

def append_records(records: list[dict], hash_index: set[str]) -> int:
    """
    Append validated, deduplicated Alpaca records to dataset.jsonl.

    Returns the count of newly written records.
    """
    written = 0
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        for rec in records:
            sol_input = rec.get("input", "")
            if not _is_valid_solidity(sol_input):
                log.debug("Skipping record — does not contain valid Solidity.")
                continue
            digest = _sha256(sol_input)
            if digest in hash_index:
                log.debug("Duplicate detected (%s…), skipping.", digest[:12])
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            hash_index.add(digest)
            written += 1
    return written


# ---------------------------------------------------------------------------
# Source 1: GitHub — SWC Registry
# ---------------------------------------------------------------------------

def _fetch_swc_entry_list() -> list[dict]:
    """
    List all entry files in the SWC Registry's /entries/ directory via
    the GitHub contents API. Returns raw API item dicts.
    """
    url = f"{GITHUB_API_BASE}/repos/{SWC_REPO_OWNER}/{SWC_REPO_NAME}/contents/{SWC_ENTRIES_PATH}"
    log.info("Fetching SWC entry list from %s", url)
    items = _github_get(url)
    return [i for i in items if i.get("name", "").endswith(".md")]


def _parse_swc_markdown(content: str, swc_id: str) -> Generator[dict, None, None]:
    """
    Parse a single SWC Registry markdown entry and yield one or more
    Alpaca records (one per code sample found in the entry).

    The SWC markdown format contains:
      ## Description   — vulnerability explanation
      ## Remediation   — fix guidance
      ### ... code samples in ``` solidity fences with Vulnerable / Fixed labels
    """
    # Extract sections using regex on known SWC heading structure
    desc_match = re.search(
        r"##\s+Description\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL | re.IGNORECASE
    )
    rem_match = re.search(
        r"##\s+Remediation\s*\n(.*?)(?=\n##|\Z)", content, re.DOTALL | re.IGNORECASE
    )
    title_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)

    title       = title_match.group(1).strip() if title_match else swc_id
    description = desc_match.group(1).strip()  if desc_match  else ""
    remediation = rem_match.group(1).strip()   if rem_match   else ""

    if not description:
        log.debug("%s: No description found, skipping.", swc_id)
        return

    # Locate paired code samples: look for "Vulnerable" and "Fixed" labeled blocks
    # Pattern: optional heading → ```solidity … ``` blocks
    code_blocks = re.findall(
        r"([\w\s\-]*?Vulnerable[\w\s\-]*?)\n```(?:solidity)?\n(.*?)```"
        r".*?"
        r"([\w\s\-]*?(?:Fixed|Safe)[\w\s\-]*?)\n```(?:solidity)?\n(.*?)```",
        content,
        re.DOTALL | re.IGNORECASE,
    )

    if not code_blocks:
        # Fallback: grab any solidity fences, treat first as vulnerable, rest skip
        all_blocks = re.findall(r"```(?:solidity)?\n(.*?)```", content, re.DOTALL)
        if all_blocks:
            vuln_code = all_blocks[0]
            fixed_code = all_blocks[1] if len(all_blocks) > 1 else "// See remediation above."
            yield build_alpaca_record(
                swc_id=swc_id,
                title=title,
                description=description,
                remediation=remediation,
                vulnerable_code=vuln_code,
                fixed_code=fixed_code,
                vuln_class=title,
            )
        return

    for _, vuln_code, _, fixed_code in code_blocks:
        yield build_alpaca_record(
            swc_id=swc_id,
            title=title,
            description=description,
            remediation=remediation,
            vulnerable_code=vuln_code,
            fixed_code=fixed_code,
            vuln_class=title,
        )


def ingest_swc_registry(hash_index: set[str]) -> int:
    """
    Primary ETL source: Pull every SWC Registry entry via the GitHub REST API,
    parse code samples, and write Alpaca records to dataset.jsonl.

    Idempotency:
      - Raw markdown files are cached in data/raw/swc/ to avoid re-downloading.
      - SHA-256 deduplication prevents duplicate JSONL entries on re-runs.

    Returns the total number of new records written.
    """
    swc_raw_dir = RAW_DIR / "swc"
    swc_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        entry_list = _fetch_swc_entry_list()
    except requests.HTTPError as exc:
        log.error("Failed to fetch SWC entry list: %s", exc)
        return 0

    total_written = 0
    log.info("Found %d SWC entries to process.", len(entry_list))

    for item in entry_list:
        swc_id   = Path(item["name"]).stem.upper()   # e.g. "SWC-107"
        raw_path = swc_raw_dir / item["name"]        # data/raw/swc/SWC-107.md

        # --- Idempotent download ---
        if raw_path.exists():
            log.info("Cache hit: %s — reading from disk.", item["name"])
            content = raw_path.read_text(encoding="utf-8")
        else:
            log.info("Downloading %s …", item["name"])
            try:
                # Use the download_url from the API response (raw.githubusercontent.com).
                # IMPORTANT: do NOT send Authorization to raw.githubusercontent.com — the
                # CDN returns 404 when it receives unexpected auth headers with an invalid token.
                download_url = item.get("download_url")
                if not download_url:
                    log.warning("No download_url for %s, skipping.", item["name"])
                    continue
                raw_resp = requests.get(
                    download_url,
                    headers={"Accept": "text/plain"},
                    timeout=30,
                )
                raw_resp.raise_for_status()
                content = raw_resp.text
                raw_path.write_text(content, encoding="utf-8")
                time.sleep(0.3)   # polite rate-limit buffer
            except requests.HTTPError as exc:
                log.error("Could not download %s: %s", item["name"], exc)
                continue

        records = list(_parse_swc_markdown(content, swc_id))
        if not records:
            log.debug("%s: No parseable code samples.", swc_id)
            continue

        written = append_records(records, hash_index)
        log.info("%s → %d new record(s) written.", swc_id, written)
        total_written += written

    return total_written


# ---------------------------------------------------------------------------
# Source 2: HuggingFace Datasets API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Source 2: HuggingFace Datasets API
# ---------------------------------------------------------------------------

def _get_hf_config(repo_id: str) -> str:
    """Queries HF to find the first valid configuration name."""
    url = f"https://datasets-server.huggingface.co/splits?dataset={repo_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN.strip() else {}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        splits_data = resp.json()
        # Prefer 'default' if it exists, otherwise take the first one available
        configs = [s["config"] for s in splits_data.get("splits", [])]
        if configs:
            return "default" if "default" in configs else configs[0]
        return "default"
    except Exception as e:
        log.error(f"Could not detect config for {repo_id}: {e}")
        return "default"


def _hf_dataset_rows(
    repo_id: str,
    split: str,
    offset: int = 0,
    length: int = 100,
) -> dict:
    """Fetch rows from HuggingFace with automatic configuration detection."""
    # This is the crucial fix: Dynamically get the config!
    config = _get_hf_config(repo_id)
    
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": repo_id,
        "config": config,  # <--- Uses dynamic config instead of hardcoded "default"
        "split": split,
        "offset": offset,
        "length": length,
    }
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN.strip() else {}
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def ingest_huggingface(hash_index: set[str], max_rows_per_dataset: int = 500) -> int:
    """
    Secondary ETL source: Pull labeled vulnerable contract rows from HuggingFace
    datasets via the datasets-server HTTP API.
    """
    total_written = 0

    for ds_cfg in HF_DATASETS:
        repo_id      = ds_cfg["repo_id"]
        split        = ds_cfg["split"]
        sol_col      = ds_cfg["sol_column"]
        label_col    = ds_cfg["label_column"]
        vuln_label   = ds_cfg["vuln_label"]

        hf_raw_dir = RAW_DIR / "huggingface" / repo_id.replace("/", "_")
        hf_raw_dir.mkdir(parents=True, exist_ok=True)

        log.info("Processing HuggingFace dataset: %s (split=%s)", repo_id, split)

        offset   = 0
        page_sz  = 100
        ds_total = 0

        while ds_total < max_rows_per_dataset:
            cache_file = hf_raw_dir / f"{split}_offset{offset}.json"

            if cache_file.exists():
                log.info("Cache hit: %s", cache_file.name)
                with cache_file.open("r", encoding="utf-8") as cf:
                    page_data = json.load(cf)
            else:
                try:
                    page_data = _hf_dataset_rows(repo_id, split, offset=offset, length=page_sz)
                    with cache_file.open("w", encoding="utf-8") as cf:
                        json.dump(page_data, cf)
                    time.sleep(0.5)
                except requests.HTTPError as exc:
                    log.error("HF API error for %s at offset %d: %s", repo_id, offset, exc)
                    break

            rows = page_data.get("rows", [])
            if not rows:
                break

            records: list[dict] = []
            for row_wrapper in rows:
                row = row_wrapper.get("row", {})
                sol_code = row.get(sol_col, "")
                if not sol_code:
                    continue

                # Filter: skip rows labelled as "safe" when a label column exists
                if vuln_label is not None:
                    row_label = str(row.get(label_col, "")).strip()
                    if row_label != str(vuln_label):
                        continue

                vuln_class = str(row.get(label_col, "")) or None

                fixed_placeholder = (
                    "// Apply the remediation strategy described above.\n"
                    "// Consult the SWC Registry for a canonical fixed example."
                )

                rec = build_alpaca_record(
                    swc_id="SWC-UNKNOWN",
                    title=vuln_class or "Smart Contract Vulnerability",
                    description=(
                        f"This contract was flagged as vulnerable by automated analysis "
                        f"(label: {vuln_class})."
                    ),
                    remediation=(
                        "Apply standard defensive patterns: CEI ordering, reentrancy guards, "
                        "SafeMath (or Solidity ≥0.8), and strict access-control modifiers."
                    ),
                    vulnerable_code=sol_code,
                    fixed_code=fixed_placeholder,
                    vuln_class=vuln_class,
                )
                records.append(rec)

            newly_written = append_records(records, hash_index)
            ds_total    += newly_written
            total_written += newly_written
            log.info(
                "%s [offset=%d] → %d new record(s) written (dataset total: %d).",
                repo_id, offset, newly_written, ds_total,
            )

            offset += page_sz
            if len(rows) < page_sz:
                break   # last page

        log.info("Finished %s: %d records ingested.", repo_id, ds_total)

    return total_written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Sentinel — Phase 2 ETL: build instruction fine-tuning dataset."
    )
    parser.add_argument("--github", action="store_true", help="Ingest SWC Registry via GitHub API")
    parser.add_argument("--hf",     action="store_true", help="Ingest HuggingFace labeled datasets")
    parser.add_argument("--all",    action="store_true", help="Run all sources")
    args = parser.parse_args()

    run_github = args.all or args.github or (not args.hf and not args.all)
    run_hf     = args.all or args.hf

    _ensure_dirs()
    hash_index = _load_hash_index()
    log.info("Loaded deduplication index: %d existing hashes.", len(hash_index))

    grand_total = 0

    if run_github:
        log.info("=" * 60)
        log.info("SOURCE 1: SWC Registry (GitHub REST API)")
        log.info("=" * 60)
        n = ingest_swc_registry(hash_index)
        log.info("GitHub source complete: %d new records.", n)
        grand_total += n

    if run_hf:
        log.info("=" * 60)
        log.info("SOURCE 2: HuggingFace Datasets API")
        log.info("=" * 60)
        n = ingest_huggingface(hash_index)
        log.info("HuggingFace source complete: %d new records.", n)
        grand_total += n

    _save_hash_index(hash_index)
    log.info("=" * 60)
    log.info("ETL pipeline complete. Grand total new records: %d", grand_total)
    log.info("Dataset path: %s", DATASET_PATH)
    log.info("Hash index:   %s", HASH_INDEX_PATH)


if __name__ == "__main__":
    main()
