"""YouTube Watcher Agent.

Searches YouTube for videos relevant to the Snow-Town ecosystem (AI agents,
MCP, LLM tooling, solo-dev productivity).  Uses yt-dlp for search and writes
results as JSON signal-data files to data/youtube/.

When the Snow-Town ContractStore is available the agent also writes
ResearchSignal records, matching the pattern of the other agents.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import (
    DATA_DIR,
    YOUTUBE_MAX_RESULTS_PER_QUERY,
    YOUTUBE_MIN_RELEVANCE,
    YOUTUBE_SEARCH_QUERIES,
)

logger = logging.getLogger(__name__)

RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}

YOUTUBE_DATA_DIR = DATA_DIR / "youtube"

# Optional: ContractStore integration when Snow-Town is available
_HAS_CONTRACT_STORE = False
try:
    from ..signal_writer import get_store, signal_exists, write_signal
    from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

    _HAS_CONTRACT_STORE = True
except Exception:
    pass


def _make_signal_id(video_id: str) -> str:
    """Generate a deterministic signal ID from a YouTube video ID."""
    return f"yt-{video_id}"


def _search_youtube(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube via yt-dlp (flat playlist, metadata only).

    Returns list of dicts with: video_id, title, description, url,
    channel, channel_url, duration, views, uploaded_date
    """
    search_term = f"ytsearch{max_results}:{query}"
    cmd = [
        "yt-dlp", "--flat-playlist", "--dump-json",
        "--no-warnings", "--quiet",
        search_term,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"yt-dlp failed for query '{query}': {result.stderr[:200]}")
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"yt-dlp error for query '{query}': {e}")
        return []

    videos: list[dict] = []
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = item.get("id", "")
        if not video_id:
            continue

        channel_id = item.get("channel_id") or ""
        videos.append({
            "video_id": video_id,
            "title": (item.get("title") or "").strip(),
            "description": (item.get("description") or "")[:500],
            "url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
            "channel": item.get("uploader") or item.get("channel") or "",
            "channel_url": f"https://www.youtube.com/channel/{channel_id}" if channel_id else "",
            "duration": int(item.get("duration") or 0),
            "views": item.get("view_count") or 0,
            "uploaded_date": item.get("upload_date") or "",
        })

    return videos


def _write_signal_file(video: dict, query: str, assessment: dict | None = None) -> Path:
    """Write a signal data JSON file for a video.

    Returns the path of the written file.
    """
    YOUTUBE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    signal = {
        "signal_id": _make_signal_id(video["video_id"]),
        "source": "youtube",
        "title": video["title"],
        "summary": video["description"],
        "url": video["url"],
        "channel": video["channel"],
        "query": query,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "raw_data": {
            "video_id": video["video_id"],
            "channel": video["channel"],
            "channel_url": video["channel_url"],
            "duration": video["duration"],
            "views": video["views"],
            "uploaded_date": video["uploaded_date"],
        },
    }

    if assessment:
        signal["relevance"] = assessment.get("relevance", "medium")
        signal["relevance_rationale"] = assessment.get("relevance_rationale", "")
        signal["tags"] = assessment.get("tags", [])
        signal["domain"] = assessment.get("domain")
        signal["persona_tags"] = assessment.get("persona_tags", [])

    filename = f"{video['video_id']}.json"
    path = YOUTUBE_DATA_DIR / filename
    path.write_text(json.dumps(signal, indent=2) + "\n")
    return path


def _load_existing_ids() -> set[str]:
    """Load video IDs already captured in data/youtube/."""
    ids: set[str] = set()
    if YOUTUBE_DATA_DIR.exists():
        for f in YOUTUBE_DATA_DIR.glob("*.json"):
            ids.add(f.stem)  # filename minus .json == video_id
    return ids


def run_agent(dry_run: bool = False) -> str:
    """Run the YouTube watcher agent.

    1. Search YouTube for each configured query via yt-dlp
    2. Deduplicate against existing signal files (and ContractStore if available)
    3. Optionally assess relevance via Claude API
    4. Write JSON signal-data files to data/youtube/
    5. Optionally write to ContractStore

    Returns summary string.
    """
    store = None
    client = None

    if _HAS_CONTRACT_STORE and not dry_run:
        try:
            store = get_store()
            from ..claude_client import assess_relevance as _assess, get_client
            client = get_client()
        except Exception as e:
            logger.info(f"ContractStore/Claude not available, writing JSON only: {e}")
            store = None
            client = None

    existing_ids = _load_existing_ids()

    total_found = 0
    total_new = 0
    total_written = 0
    skipped_low = 0
    files_written: list[str] = []

    try:
        for query in YOUTUBE_SEARCH_QUERIES:
            logger.info(f"Searching YouTube: '{query}'")
            videos = _search_youtube(query, max_results=YOUTUBE_MAX_RESULTS_PER_QUERY)
            total_found += len(videos)

            for video in videos:
                vid = video["video_id"]

                # Dedup against local files
                if vid in existing_ids:
                    continue

                # Dedup against ContractStore
                if store and _HAS_CONTRACT_STORE:
                    sig_id = _make_signal_id(vid)
                    if signal_exists(sig_id, store=store):
                        existing_ids.add(vid)
                        continue

                total_new += 1

                if dry_run:
                    logger.info(f"  [DRY RUN] Would capture: {video['title'][:80]}")
                    continue

                assessment = None
                relevance = "medium"  # default when Claude isn't available

                # Claude relevance assessment (if available)
                if client is not None:
                    try:
                        assessment = _assess(
                            title=video["title"],
                            summary=video["description"],
                            source_context=(
                                f"YouTube video by {video['channel']}, "
                                f"{video['views']} views, "
                                f"duration: {video['duration']}s"
                            ),
                            client=client,
                        )
                        relevance = assessment.get("relevance", "low")
                    except Exception as e:
                        logger.warning(f"  Claude assessment failed: {e}")

                # Apply relevance threshold
                min_level = RELEVANCE_ORDER.get(YOUTUBE_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_low += 1
                    logger.debug(f"  Skipped (low relevance): {video['title'][:60]}")
                    continue

                # Write local JSON signal file
                path = _write_signal_file(video, query, assessment)
                files_written.append(str(path))
                existing_ids.add(vid)
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {video['title'][:60]}")

                # Write to ContractStore if available
                if store and _HAS_CONTRACT_STORE and assessment:
                    try:
                        tags = assessment.get("tags", [])
                        for persona in assessment.get("persona_tags", []):
                            tags.append(f"persona:{persona}")

                        write_signal(
                            signal_id=_make_signal_id(vid),
                            source=SignalSource.DOMAIN_WATCH,  # Reuse closest source
                            title=video["title"],
                            summary=video["description"],
                            url=video["url"],
                            relevance=SignalRelevance(relevance),
                            relevance_rationale=assessment.get("relevance_rationale", ""),
                            tags=tags,
                            domain=assessment.get("domain"),
                            raw_data=video,
                            store=store,
                        )
                    except Exception as e:
                        logger.debug(f"  ContractStore write skipped: {e}")

            # Rate limit courtesy
            time.sleep(1.5)

    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    return (
        f"Found {total_found} videos, {total_new} new, "
        f"{total_written} written, {skipped_low} skipped (low relevance). "
        f"Files: {len(files_written)}"
    )
