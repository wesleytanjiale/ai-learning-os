"""Ingest layer — fetch full text from YouTube, web articles, markdown files, and images."""

import base64
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(url: str) -> str | None:
    for pattern in [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def _youtube_title(url: str) -> str:
    try:
        req = urllib.request.Request(f"https://www.youtube.com/oembed?url={url}&format=json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("title", url)
    except Exception:
        return url


def _parse_vtt(vtt: str) -> str:
    """Extract clean text from a WebVTT subtitle file, deduplicating repeated lines."""
    lines = []
    for line in vtt.splitlines():
        line = line.strip()
        if not line or line == "WEBVTT" or "-->" in line or re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<[^>]+>", "", line)  # strip VTT inline tags
        if line:
            lines.append(line)
    # VTT repeats lines across overlapping cues — deduplicate consecutive dupes
    deduped: list[str] = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
            prev = line
    return " ".join(deduped)


def _fetch_via_ytdlp(url: str, cookies_browser: str | None = None) -> str:
    """
    Fetch transcript via yt-dlp. Tries plain first, then with browser cookies.
    cookies_browser: 'safari' | 'chrome' | 'firefox' | None
    """
    base_cmd = [
        "uv", "run", "yt-dlp",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--no-warnings",
        # storyboard format satisfies yt-dlp's format check without downloading video
        "-f", "sb0",
    ]
    if cookies_browser:
        base_cmd += ["--cookies-from-browser", cookies_browser]

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = base_cmd + ["-o", f"{tmpdir}/%(id)s.%(ext)s", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if result.returncode != 0 or not vtt_files:
            raise RuntimeError(result.stderr.strip() or "no subtitle file produced")
        return _parse_vtt(vtt_files[0].read_text(encoding="utf-8"))


def fetch_youtube(url: str) -> dict:
    """
    Fetch YouTube transcript with a three-tier fallback:
      1. youtube-transcript-api  (fast, no subprocess)
      2. yt-dlp without cookies  (handles some IP blocks)
      3. yt-dlp with Safari cookies, then Chrome, then Firefox
         (authenticated requests bypass most rate limits)
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return {"error": f"Could not extract video ID from: {url}"}

    title = _youtube_title(url)
    errors: list[str] = []

    # Tier 1: youtube-transcript-api
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(snippet.text for snippet in transcript)
        return {"title": title, "text": text, "source_type": "youtube"}
    except Exception as e:
        errors.append(f"transcript-api: {e}")

    # Tier 2 + 3: yt-dlp — no cookies, then each browser
    for browser in [None, "safari", "chrome", "firefox"]:
        label = f"yt-dlp({browser or 'no-cookies'})"
        try:
            text = _fetch_via_ytdlp(url, cookies_browser=browser)
            return {"title": title, "text": text, "source_type": "youtube"}
        except Exception as e:
            errors.append(f"{label}: {e}")

    return {
        "error": (
            "All YouTube transcript methods failed. "
            "YouTube may be rate-limiting your IP — try again in a few minutes or use a VPN. "
            f"Details: {' | '.join(errors[-2:])}"  # last 2 errors are most relevant
        )
    }


def fetch_article(url: str) -> dict:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title else url
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        return {
            "title": title,
            "text": text[:50_000],  # cap at ~50k chars; enough for most articles
            "source_type": "article",
        }
    except Exception as e:
        return {"error": str(e)}


def describe_image(image_path: str, client: OpenAI, vision_model: str) -> str:
    """Call vision LLM to describe an image file. Returns description string."""
    path = Path(image_path)
    if not path.exists():
        return f"[Image not found: {image_path}]"
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(suffix, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    response = client.chat.completions.create(
        model=vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": "Describe this image in detail, focusing on any diagrams, charts, code, or technical content. Be specific and thorough."},
            ],
        }],
        max_tokens=500,
    )
    return response.choices[0].message.content


def process_markdown(file_path: str, client: OpenAI, vision_model: str) -> dict:
    """Read a markdown file (e.g. marker output). Inline-replaces image refs with vision descriptions."""
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("-", " ").replace("_", " ").title()

    # Replace ![alt](image_path) with [Image: <description>]
    def replace_image(match):
        img_path = match.group(2)
        # Resolve relative to the markdown file's directory
        resolved = (path.parent / img_path).resolve()
        if resolved.exists():
            desc = describe_image(str(resolved), client, vision_model)
            return f"[Image: {desc}]"
        return f"[Image: {img_path} — file not found]"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_image, text)

    return {
        "title": title,
        "text": text,
        "source_type": "markdown",
    }


def process_image(file_path: str, client: OpenAI, vision_model: str) -> dict:
    """Describe a standalone image file."""
    path = Path(file_path)
    description = describe_image(file_path, client, vision_model)
    return {
        "title": path.stem.replace("-", " ").replace("_", " ").title(),
        "text": description,
        "source_type": "image",
    }


def fetch_source(source: str, client: OpenAI, vision_model: str) -> dict:
    """Route a source string to the correct ingest function."""
    s = source.strip()
    if "youtube.com" in s or "youtu.be" in s:
        return fetch_youtube(s)
    if s.startswith("http://") or s.startswith("https://"):
        return fetch_article(s)
    path = Path(s)
    if path.suffix.lower() == ".md":
        return process_markdown(s, client, vision_model)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return process_image(s, client, vision_model)
    return {"error": f"Unsupported source: {s}. Provide a YouTube URL, article URL, .md file path, or image path."}
