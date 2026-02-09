"""
agents/moltbook.py - Optional Moltbook posting after matches.

Posts match results and strategy reasoning to the gaming submolt.
Silent fail if MOLTBOOK_API_KEY is not set.
"""

import json
import logging
import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

MOLTBOOK_API_URL = "https://moltbook.com/api/v1/posts"


def post_match_result(
    winner: str,
    loser: str,
    score: str,
    match_id: str,
    reasoning: str,
    api_key: str | None = None,
) -> bool:
    """Post a match result to the gaming submolt.

    Args:
        winner: Winner address or name.
        loser: Loser address or name.
        score: Score string (e.g. "4-2").
        match_id: Match ID for linking.
        reasoning: Strategy reasoning to include.
        api_key: Moltbook API key. Falls back to MOLTBOOK_API_KEY env var.

    Returns:
        True if posted, False otherwise.
    """
    key = api_key or os.environ.get("MOLTBOOK_API_KEY")
    if not key:
        logger.debug("No MOLTBOOK_API_KEY — skipping post")
        return False

    content = (
        f"Match result: {winner[:10]}... defeats {loser[:10]}... ({score})\n"
        f"Strategy: {reasoning}\n"
        f"Match: {match_id[:16]}..."
    )

    payload = {
        "content": content,
        "submolt": "gaming",
        "tags": ["nojohns", "melee", "match-result"],
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            MOLTBOOK_API_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200 or resp.status == 201
    except (urllib.error.URLError, Exception) as e:
        logger.debug(f"Moltbook post failed: {e}")
        return False
