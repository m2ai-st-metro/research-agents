"""YouTube Trending Scanner Agent.

Scans YouTube for trending/popular videos in AI, tech, and supply chain topics.
Extracts transcripts, summarizes via Gemini Flash, generates Mermaid diagrams
of key concepts, and writes ResearchSignal records to ST Records ContractStore.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import httpx

from ..claude_client import assess_relevance, get_client
from ..config import (
    YOUTUBE_API_KEY_ENV,
    YOUTUBE_CHANNEL_MAX_VIDEOS,
    YOUTUBE_CHANNELS,
    YOUTUBE_MAX_RESULTS_PER_QUERY,
    YOUTUBE_MIN_RELEVANCE,
    YOUTUBE_SEARCH_QUERIES,
    YOUTUBE_SUMMARIZER_MODEL,
    YOUTUBE_TRANSCRIPT_MAX_CHARS,
)
from ..gemini_client import summarize_transcript as gemini_summarize_transcript
from ..signal_writer import get_store, signal_exists, write_signal

# Re-import after sys.path setup in signal_writer
from contracts.research_signal import SignalRelevance, SignalSource  # noqa: E402

logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
RELEVANCE_ORDER = {"high": 3, "medium": 2, "low": 1}


def _get_youtube_api_key() -> str | None:
    """Get YouTube Data API v3 key from environment."""
    return os.environ.get(YOUTUBE_API_KEY_ENV)


def _make_signal_id(video_id: str) -> str:
    """Generate a deterministic signal ID from a YouTube video ID."""
    return f"youtube-{video_id}"


def _search_youtube_api(
    query: str, api_key: str, max_results: int = 5, order: str = "relevance"
) -> list[dict]:
    """Search YouTube Data API v3 for videos matching the query.

    Returns list of dicts with: video_id, title, description, channel_title,
    published_at, thumbnail_url, view_count
    """
    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "order": order,
        "maxResults": min(max_results, 50),
        "relevanceLanguage": "en",
        "videoDuration": "medium",  # 4-20 minutes (good content length)
        "key": api_key,
    }

    try:
        resp = httpx.get(
            f"{YOUTUBE_API_BASE}/search",
            params=search_params,
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"YouTube API error for query '{query}': {e}")
        return []

    data = resp.json()
    videos = []

    video_ids = []
    snippets: dict[str, dict] = {}

    for item in data.get("items", []):
        vid_id = item.get("id", {}).get("videoId")
        if not vid_id:
            continue
        video_ids.append(vid_id)
        snippet = item.get("snippet", {})
        snippets[vid_id] = {
            "title": snippet.get("title", ""),
            "description": (snippet.get("description") or "")[:500],
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "thumbnail_url": (
                snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            ),
        }

    # Fetch view counts via videos endpoint
    if video_ids:
        stats = _get_video_stats(video_ids, api_key)
    else:
        stats = {}

    for vid_id in video_ids:
        info = snippets[vid_id]
        info["video_id"] = vid_id
        info["url"] = f"https://www.youtube.com/watch?v={vid_id}"
        info["view_count"] = stats.get(vid_id, {}).get("view_count", 0)
        info["like_count"] = stats.get(vid_id, {}).get("like_count", 0)
        info["duration"] = stats.get(vid_id, {}).get("duration", "")
        videos.append(info)

    return videos


def _get_video_stats(video_ids: list[str], api_key: str) -> dict[str, dict]:
    """Fetch video statistics (views, likes, duration) for a batch of video IDs."""
    params = {
        "part": "statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": api_key,
    }

    try:
        resp = httpx.get(
            f"{YOUTUBE_API_BASE}/videos",
            params=params,
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"YouTube stats API error: {e}")
        return {}

    data = resp.json()
    stats: dict[str, dict] = {}
    for item in data.get("items", []):
        vid_id = item.get("id", "")
        statistics = item.get("statistics", {})
        content = item.get("contentDetails", {})
        stats[vid_id] = {
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "duration": content.get("duration", ""),
        }

    return stats


def _search_youtube_fallback(query: str, max_results: int = 5) -> list[dict]:
    """Fallback search using yt-dlp when no YouTube API key is available.

    Uses yt-dlp's ytsearch to find videos. Slower but requires no API key.
    """
    search_query = f"ytsearch{max_results}:{query}"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--flat-playlist",
                "--no-warnings",
                search_query,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        logger.warning("yt-dlp not installed. Cannot use fallback search.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"yt-dlp search timed out for query '{query}'")
        return []

    if result.returncode != 0:
        logger.warning(f"yt-dlp search failed for '{query}': {result.stderr[:200]}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        vid_id = entry.get("id", "")
        if not vid_id:
            continue

        videos.append({
            "video_id": vid_id,
            "title": entry.get("title", ""),
            "description": (entry.get("description") or "")[:500],
            "channel_title": entry.get("channel", entry.get("uploader", "")),
            "published_at": entry.get("upload_date", ""),
            "thumbnail_url": entry.get("thumbnail", ""),
            "url": entry.get("url", f"https://www.youtube.com/watch?v={vid_id}"),
            "view_count": entry.get("view_count", 0) or 0,
            "like_count": entry.get("like_count", 0) or 0,
            "duration": entry.get("duration_string", ""),
        })

    return videos


def search_youtube(query: str, max_results: int = 5, order: str = "relevance") -> list[dict]:
    """Search YouTube for videos. Uses API if key available, else yt-dlp fallback."""
    api_key = _get_youtube_api_key()
    if api_key:
        logger.info(f"Using YouTube Data API v3 for search: '{query}' (order={order})")
        return _search_youtube_api(query, api_key, max_results, order=order)
    else:
        logger.info(f"No YouTube API key, using yt-dlp fallback for: '{query}'")
        return _search_youtube_fallback(query, max_results)


def _fetch_channel_videos_api(
    handle: str, api_key: str, max_results: int = 5
) -> list[dict]:
    """Fetch recent uploads from a YouTube channel via the Data API v3.

    Uses the search endpoint filtered by channelId or handle, ordered by date.
    """
    # First resolve handle to channel ID
    try:
        ch_resp = httpx.get(
            f"{YOUTUBE_API_BASE}/channels",
            params={"part": "contentDetails", "forHandle": handle.lstrip("@"), "key": api_key},
            timeout=15.0,
        )
        ch_resp.raise_for_status()
        ch_data = ch_resp.json()
        items = ch_data.get("items", [])
        if not items:
            logger.warning(f"Could not resolve channel for handle '{handle}'")
            return []
        channel_id = items[0]["id"]
    except httpx.HTTPError as e:
        logger.warning(f"YouTube API channel lookup failed for '{handle}': {e}")
        return []

    # Search recent uploads for that channel
    search_params = {
        "part": "snippet",
        "channelId": channel_id,
        "type": "video",
        "order": "date",
        "maxResults": min(max_results, 50),
        "key": api_key,
    }

    try:
        resp = httpx.get(
            f"{YOUTUBE_API_BASE}/search",
            params=search_params,
            timeout=30.0,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"YouTube API channel search failed for '{handle}': {e}")
        return []

    data = resp.json()
    video_ids = []
    snippets: dict[str, dict] = {}

    for item in data.get("items", []):
        vid_id = item.get("id", {}).get("videoId")
        if not vid_id:
            continue
        video_ids.append(vid_id)
        snippet = item.get("snippet", {})
        snippets[vid_id] = {
            "title": snippet.get("title", ""),
            "description": (snippet.get("description") or "")[:500],
            "channel_title": snippet.get("channelTitle", ""),
            "published_at": snippet.get("publishedAt", ""),
            "thumbnail_url": (
                snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            ),
        }

    # Fetch stats
    stats = _get_video_stats(video_ids, api_key) if video_ids else {}

    videos = []
    for vid_id in video_ids:
        info = snippets[vid_id]
        info["video_id"] = vid_id
        info["url"] = f"https://www.youtube.com/watch?v={vid_id}"
        info["view_count"] = stats.get(vid_id, {}).get("view_count", 0)
        info["like_count"] = stats.get(vid_id, {}).get("like_count", 0)
        info["duration"] = stats.get(vid_id, {}).get("duration", "")
        videos.append(info)

    return videos


def _fetch_channel_videos_fallback(
    handle: str, max_results: int = 5
) -> list[dict]:
    """Fetch recent uploads from a YouTube channel via yt-dlp (no API key needed)."""
    channel_url = f"https://www.youtube.com/{handle}/videos"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--flat-playlist",
                "--no-warnings",
                f"--playlist-items=1-{max_results}",
                channel_url,
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except FileNotFoundError:
        logger.warning("yt-dlp not installed. Cannot fetch channel videos.")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"yt-dlp channel fetch timed out for '{handle}'")
        return []

    if result.returncode != 0:
        logger.warning(f"yt-dlp channel fetch failed for '{handle}': {result.stderr[:200]}")
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        vid_id = entry.get("id", "")
        if not vid_id:
            continue

        videos.append({
            "video_id": vid_id,
            "title": entry.get("title", ""),
            "description": (entry.get("description") or "")[:500],
            "channel_title": entry.get("channel", entry.get("uploader", "")),
            "published_at": entry.get("upload_date", ""),
            "thumbnail_url": entry.get("thumbnail", ""),
            "url": entry.get("url", f"https://www.youtube.com/watch?v={vid_id}"),
            "view_count": entry.get("view_count", 0) or 0,
            "like_count": entry.get("like_count", 0) or 0,
            "duration": entry.get("duration_string", ""),
        })

    return videos


def fetch_channel_videos(
    handle: str, max_results: int = 5
) -> list[dict]:
    """Fetch recent uploads from a channel. Uses API if available, else yt-dlp."""
    api_key = _get_youtube_api_key()
    if api_key:
        logger.info(f"Fetching channel videos via API: '{handle}'")
        return _fetch_channel_videos_api(handle, api_key, max_results)
    else:
        logger.info(f"Fetching channel videos via yt-dlp: '{handle}'")
        return _fetch_channel_videos_fallback(handle, max_results)


def get_transcript(video_id: str) -> str | None:
    """Extract transcript from a YouTube video.

    Tries youtube-transcript-api first, falls back to yt-dlp.
    Returns the transcript text, or None if unavailable.
    """
    # Primary: youtube-transcript-api
    transcript = _get_transcript_api(video_id)
    if transcript:
        return transcript

    # Fallback: yt-dlp subtitle extraction
    return _get_transcript_ytdlp(video_id)


def _get_transcript_api(video_id: str) -> str | None:
    """Extract transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        logger.debug("youtube-transcript-api not installed")
        return None

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id)
        parts = [entry.text for entry in transcript_list]
        full_text = " ".join(parts)
        return full_text[:YOUTUBE_TRANSCRIPT_MAX_CHARS]
    except Exception as e:
        logger.debug(f"youtube-transcript-api failed for {video_id}: {e}")
        return None


def _get_transcript_ytdlp(video_id: str) -> str | None:
    """Extract transcript/subtitles using yt-dlp as fallback."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--sub-format", "json3",
                "--output", "-",
                "--print", "%(subtitles)j",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    try:
        subs_data = json.loads(result.stdout.strip())
        # Try to extract text from subtitle data
        if isinstance(subs_data, dict):
            for lang_data in subs_data.values():
                if isinstance(lang_data, list):
                    for entry in lang_data:
                        if isinstance(entry, dict) and "url" in entry:
                            # Fetch the subtitle file
                            try:
                                resp = httpx.get(entry["url"], timeout=15.0)
                                sub_json = resp.json()
                                parts = [
                                    ev.get("segs", [{}])[0].get("utf8", "")
                                    for ev in sub_json.get("events", [])
                                    if ev.get("segs")
                                ]
                                text = " ".join(p for p in parts if p.strip())
                                if text:
                                    return text[:YOUTUBE_TRANSCRIPT_MAX_CHARS]
                            except Exception:
                                continue
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def summarize_transcript(
    title: str,
    transcript: str,
    client: object | None = None,
    model: str = YOUTUBE_SUMMARIZER_MODEL,
) -> dict:
    """Summarize a YouTube video transcript using Gemini.

    Delegates to gemini_client.summarize_transcript(). The ``client`` param
    is accepted for call-site compatibility but ignored -- a Gemini client
    is created internally.

    Returns dict with:
        summary: str - 2-3 paragraph summary of key points
        key_concepts: list[str] - main concepts/technologies discussed
        mermaid_diagram: str - Mermaid diagram of architecture/concepts
        tags: list[str] - relevant tags for categorization
    """
    return gemini_summarize_transcript(
        title=title,
        transcript=transcript,
        model=model,
    )


def run_agent(dry_run: bool = False) -> str:
    """Run the YouTube trending scanner agent.

    1. Fetch recent uploads from monitored channels (YOUTUBE_CHANNELS)
    2. Search YouTube for each configured query
    3. Extract transcripts from discovered videos
    4. Summarize transcripts using Gemini Flash
    5. Assess relevance via Claude API
    6. Generate Mermaid diagrams of key concepts
    7. Write signals that pass the relevance threshold

    Returns summary string.
    """
    store = get_store()
    client = None if dry_run else get_client()

    total_found = 0
    total_new = 0
    total_written = 0
    total_transcripts = 0
    skipped_low = 0
    skipped_no_transcript = 0
    channel_found = 0

    # Collect all video batches: (label, videos_list) tuples
    video_batches: list[tuple[str, list[dict]]] = []

    # Phase 1: Monitored channels -- always fetch latest uploads
    for channel in YOUTUBE_CHANNELS:
        handle = channel.get("handle", "")
        name = channel.get("name", handle)
        if not handle:
            continue
        logger.info(f"Fetching channel: {name} ({handle})")
        ch_videos = fetch_channel_videos(
            handle, max_results=YOUTUBE_CHANNEL_MAX_VIDEOS
        )
        channel_found += len(ch_videos)
        video_batches.append((f"channel:{name}", ch_videos))
        time.sleep(1.0)  # Rate limit between channel fetches

    # Phase 2: Topic search queries
    # Alternate between relevance and date ordering to balance quality and freshness
    for i, query in enumerate(YOUTUBE_SEARCH_QUERIES):
        order = "date" if i % 2 == 0 else "relevance"
        logger.info(f"Searching YouTube: '{query}' (order={order})")
        videos = search_youtube(query, max_results=YOUTUBE_MAX_RESULTS_PER_QUERY, order=order)
        video_batches.append((f"search:{query}", videos))

    # Process all batches through the same pipeline
    try:
        for batch_label, videos in video_batches:
            total_found += len(videos)

            for video in videos:
                video_id = video["video_id"]
                signal_id = _make_signal_id(video_id)

                # Dedup check
                if signal_exists(signal_id, store=store):
                    continue

                total_new += 1

                if dry_run:
                    logger.info(f"  [DRY RUN] Would process: {video['title'][:80]}")
                    continue

                # Extract transcript
                transcript = get_transcript(video_id)
                if not transcript:
                    skipped_no_transcript += 1
                    logger.debug(
                        f"  No transcript available: {video['title'][:60]}"
                    )
                    # Still assess based on title/description if no transcript
                    transcript = None

                if transcript:
                    total_transcripts += 1

                # Summarize transcript with Gemini (if transcript available)
                summary_data: dict = {}
                if transcript:
                    summary_data = summarize_transcript(
                        title=video["title"],
                        transcript=transcript,
                        client=client,
                    )

                # Build the summary for relevance assessment
                signal_summary = summary_data.get(
                    "summary",
                    video.get("description", video["title"]),
                )

                # Assess relevance via Claude
                assessment = assess_relevance(
                    title=video["title"],
                    summary=signal_summary[:500],
                    source_context=(
                        f"YouTube video by {video.get('channel_title', 'unknown')}, "
                        f"{video.get('view_count', 0):,} views"
                    ),
                    client=client,
                )

                relevance = assessment.get("relevance", "low")
                min_level = RELEVANCE_ORDER.get(YOUTUBE_MIN_RELEVANCE, 2)
                if RELEVANCE_ORDER.get(relevance, 0) < min_level:
                    skipped_low += 1
                    logger.debug(
                        f"  Skipped (low relevance): {video['title'][:60]}"
                    )
                    continue

                # Combine tags from summarization and relevance assessment
                tags = list(
                    set(
                        assessment.get("tags", [])
                        + summary_data.get("tags", [])
                    )
                )

                # Build raw_data with Mermaid diagram and metadata
                raw_data: dict = {
                    "video_id": video_id,
                    "channel": video.get("channel_title", ""),
                    "view_count": video.get("view_count", 0),
                    "like_count": video.get("like_count", 0),
                    "duration": video.get("duration", ""),
                    "published_at": video.get("published_at", ""),
                    "key_concepts": summary_data.get("key_concepts", []),
                    "has_transcript": transcript is not None,
                }

                mermaid = summary_data.get("mermaid_diagram", "")
                if mermaid:
                    raw_data["mermaid_diagram"] = mermaid

                write_signal(
                    signal_id=signal_id,
                    source=SignalSource.YOUTUBE_SCANNER,
                    title=video["title"],
                    summary=signal_summary[:1000],
                    url=video.get("url", f"https://www.youtube.com/watch?v={video_id}"),
                    relevance=SignalRelevance(relevance),
                    relevance_rationale=assessment.get("relevance_rationale", ""),
                    tags=tags,
                    domain=assessment.get("domain"),
                    raw_data=raw_data,
                    store=store,
                )
                total_written += 1
                logger.info(f"  Wrote [{relevance}]: {video['title'][:60]}")

            # Rate limit courtesy between queries
            time.sleep(2.0)

    finally:
        store.close()

    return (
        f"Found {total_found} videos ({channel_found} from channels), "
        f"{total_new} new, "
        f"{total_transcripts} transcripts extracted, "
        f"{total_written} written, {skipped_low} skipped (low relevance), "
        f"{skipped_no_transcript} no transcript"
    )
