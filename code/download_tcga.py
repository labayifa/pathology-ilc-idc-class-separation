"""Stage B, plan step 2 + sec. 6: download only the selected cohort's DX slides from GDC.

Async producer-consumer queue with a semaphore to limit concurrent downloads,
retries with backoff, per-request timeouts, md5 verification, and a clean shutdown.
Idempotent: files already present with a matching md5 are skipped, so a rerun
after a crash/interrupt just resumes.

Setup:
    pip install aiohttp

Usage:
    python download_tcga.py --cohort ../cohort/cohort.csv --out-dir ../data/tcga_slides
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import logging
import ssl
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import certifi

GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{file_id}"
CHUNK_SIZE = 1 << 20  # 1MB
MAX_RETRIES = 3
# No total cap -- files are large (~GB) and the connection can be slow.
# Instead, time out only on an actual stall (no bytes for STALL_TIMEOUT_SECONDS).
STALL_TIMEOUT_SECONDS = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("download_tcga")


@dataclass(frozen=True)
class SlideRecord:
    file_id: str
    file_name: str
    md5: str
    size: int


def load_cohort(cohort_csv: Path) -> list[SlideRecord]:
    records = []
    with open(cohort_csv) as f:
        for row in csv.DictReader(f):
            records.append(
                SlideRecord(row["file_id"], row["file_name"], row["md5"], int(row["size"]))
            )
    return records


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


async def download_one(
    session: aiohttp.ClientSession, record: SlideRecord, out_dir: Path
) -> tuple[SlideRecord, bool, str]:
    dest = out_dir / record.file_name
    if dest.exists() and md5sum(dest) == record.md5:
        return record, True, "already present, md5 OK"

    url = GDC_DATA_URL.format(file_id=record.file_id)
    tmp = dest.with_suffix(dest.suffix + ".part")

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=STALL_TIMEOUT_SECONDS)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=timeout) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)

            if md5sum(tmp) != record.md5:
                tmp.unlink(missing_ok=True)
                raise ValueError("md5 mismatch after download")

            tmp.rename(dest)
            return record, True, f"downloaded on attempt {attempt}"

        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            tmp.unlink(missing_ok=True)
            reason = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            log.warning(f"{record.file_name}: attempt {attempt} failed ({reason})")
            if attempt == MAX_RETRIES:
                return record, False, reason
            await asyncio.sleep(2**attempt)  # exponential backoff

    return record, False, "unreachable"  # pragma: no cover


async def worker(
    name: int,
    queue: asyncio.Queue[SlideRecord],
    session: aiohttp.ClientSession,
    out_dir: Path,
    results: list[tuple[SlideRecord, bool, str]],
) -> None:
    while True:
        record = await queue.get()
        try:
            result = await download_one(session, record, out_dir)
            results.append(result)
            status = "OK" if result[1] else "FAILED"
            log.info(f"[worker {name}] {status}: {record.file_name} ({result[2]})")
        finally:
            queue.task_done()


async def run(records: list[SlideRecord], out_dir: Path, concurrency: int) -> list[tuple[SlideRecord, bool, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    queue: asyncio.Queue[SlideRecord] = asyncio.Queue()
    for r in records:
        queue.put_nowait(r)

    results: list[tuple[SlideRecord, bool, str]] = []
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        workers = [
            asyncio.create_task(worker(i, queue, session, out_dir, results))
            for i in range(concurrency)
        ]
        try:
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the selected TCGA-BRCA DX slides from GDC")
    parser.add_argument("--cohort", type=Path, default=Path("../cohort/cohort.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("../data/tcga_slides"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    records = load_cohort(args.cohort)
    total_gb = sum(r.size for r in records) / 1e9
    log.info(f"{len(records)} slides to fetch (~{total_gb:.1f}GB), concurrency={args.concurrency}")

    try:
        results = asyncio.run(run(records, args.out_dir, args.concurrency))
    except KeyboardInterrupt:
        log.warning("Interrupted; partial downloads (.part files) left for resume on rerun.")
        return

    ok = sum(1 for _, success, _ in results if success)
    failed = [r for r, success, _ in results if not success]
    log.info(f"Done: {ok}/{len(records)} succeeded.")
    if failed:
        log.warning(f"{len(failed)} failed: {[r.file_name for r in failed]}")


if __name__ == "__main__":
    main()
