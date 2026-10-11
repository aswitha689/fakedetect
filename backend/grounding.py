import os
from pathlib import Path
import re
from typing import Any, Optional
import urllib.parse
import requests

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def get_api_key() -> str:
    key = os.getenv("FACTCHECK_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("FACTCHECK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


def _compute_overlap(query: str, claim: str) -> float:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
        "for", "with", "and", "that", "this", "it", "not", "by", "from", "as", "be"
    }
    q_words = {w.lower() for w in re.findall(r'\b[A-Za-z]+\b', query) if w.lower() not in stopwords}
    c_words = {w.lower() for w in re.findall(r'\b[A-Za-z]+\b', claim) if w.lower() not in stopwords}
    if not q_words or not c_words:
        return 0.0
    # Also check if claim has extra specific qualifier words (e.g. obc, corrupt, arrested, killed)
    qualifiers = c_words - q_words
    # If claim introduces major qualifier words not in the query, it's not a direct match
    if any(q in qualifiers for q in ["obc", "corrupt", "arrested", "hospitalized", "killed", "died", "resigned", "scam"]):
        return 0.20
    return len(q_words & c_words) / len(q_words)


def _check_officeholder_fact(query: str) -> Optional[dict[str, Any]]:
    # Matches patterns like: [Person] is (the) (current) [office] of [place]
    m = re.search(
        r'\b([A-Za-z\s]+?)\s+is\s+(?:the\s+)?(?:current\s+)?(president|prime minister|head|ceo|governor|king|queen)\s+of\s+([A-Za-z\s]+)',
        query,
        re.IGNORECASE
    )
    if not m:
        return None

    person = m.group(1).strip()
    office = m.group(2).strip()
    country = m.group(3).strip()

    if person.lower() in {"who", "what", "he", "she", "it", "there", "someone"}:
        return None

    try:
        # 1. Check person page on Wikipedia
        person_encoded = urllib.parse.quote(person.replace(" ", "_"))
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{person_encoded}",
            headers={"User-Agent": "FakeNewsDetector/1.0"},
            timeout=4
        )
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract", "")
            if office.lower() in extract.lower() and country.lower() in extract.lower():
                return {
                    "claim": f"{person} is the {office} of {country}",
                    "rating": f"Confirmed Truth: {extract[:180]}...",
                    "publisher": "Wikipedia Verified Knowledge",
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "is_direct_match": True,
                    "is_verified_true": True,
                    "is_verified_false": False
                }

        # 2. Check office page on Wikipedia
        office_title = f"{office.title()} of {country.title()}"
        o_encoded = urllib.parse.quote(office_title.replace(" ", "_"))
        o_resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{o_encoded}",
            headers={"User-Agent": "FakeNewsDetector/1.0"},
            timeout=4
        )
        if o_resp.status_code == 200:
            o_data = o_resp.json()
            o_extract = o_data.get("extract", "")
            if person.lower() in o_extract.lower():
                return {
                    "claim": f"{person} is the {office} of {country}",
                    "rating": f"Confirmed Truth: {o_extract[:180]}...",
                    "publisher": "Wikipedia Verified Knowledge",
                    "url": o_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "is_direct_match": True,
                    "is_verified_true": True,
                    "is_verified_false": False
                }
            else:
                return {
                    "claim": f"{office_title} (Official Record)",
                    "rating": f"Contradicted by factual record: {o_extract[:180]}...",
                    "publisher": "Wikipedia Verified Knowledge",
                    "url": o_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "is_direct_match": True,
                    "is_verified_true": False,
                    "is_verified_false": True
                }
    except Exception:
        pass
    return None


def _query_google_factcheck(query: str, api_key: str) -> list[dict[str, Any]]:
    try:
        params = {
            "query": query.strip(),
            "key": api_key,
            "pageSize": 3,
            "languageCode": "en"
        }
        resp = requests.get(API_URL, params=params, timeout=5)
        if resp.status_code != 200:
            return []

        data = resp.json()
        raw_claims = data.get("claims", [])
        matches = []

        for item in raw_claims[:3]:
            reviews = item.get("claimReview", [])
            review = reviews[0] if reviews else {}
            publisher_info = review.get("publisher", {})
            claim_text = item.get("text", "")
            rating_text = review.get("textualRating", "Unknown")
            overlap = _compute_overlap(query, claim_text)
            is_direct = overlap >= 0.50

            pub_name = publisher_info.get("name", "Google Fact Check")
            if not is_direct:
                display_pub = f"{pub_name} (Related Topic Fact-Check)"
            else:
                display_pub = pub_name

            matches.append({
                "claim": claim_text,
                "rating": rating_text,
                "publisher": display_pub,
                "url": review.get("url", ""),
                "is_direct_match": is_direct,
                "is_verified_true": False,
                "is_verified_false": is_direct and any(
                    w in rating_text.lower() for w in ["false", "pants-fire", "pants on fire", "fake", "incorrect", "hoax", "pinocchio"]
                ),
                "overlap": round(overlap, 2)
            })
        return matches
    except Exception:
        return []


def search_claims(query: str) -> dict[str, Any]:
    if not query or not query.strip():
        return {"available": False, "api_configured": False, "matches": []}

    api_key = get_api_key()
    matches = []

    # 1. High-precision officeholder fact check (e.g. Narendra Modi PM of India, Aswitha president of India)
    office_fact = _check_officeholder_fact(query)
    if office_fact:
        matches.append(office_fact)

    # 2. Google Fact Check Tools API
    if api_key:
        gfc_matches = _query_google_factcheck(query, api_key)
        if not gfc_matches:
            tokens = [w for w in re.findall(r'\b\w+\b', query) if len(w) > 2]
            if len(tokens) >= 2:
                topic = " ".join(tokens[-2:])
                gfc_matches = _query_google_factcheck(topic, api_key)
        matches.extend(gfc_matches)

    is_gfc = (api_key and any("Google Fact Check" in m.get("publisher", "") or "Related Topic" in m.get("publisher", "") for m in matches))

    return {
        "available": bool(api_key or matches),
        "api_configured": bool(api_key),
        "matches": matches,
        "source": "Google Fact Check Tools API" if is_gfc else "Knowledge Grounding"
    }


# Aliases
ground = search_claims
check_fact = search_claims
