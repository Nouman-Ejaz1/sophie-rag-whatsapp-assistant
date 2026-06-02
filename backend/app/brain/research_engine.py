from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.config import (
    JINA_API_KEY,
    SEARCH_MAX_SOURCES,
    SEARCH_PROVIDER_ORDER,
    SEARCH_REQUIRE_TRUSTED_FOR_LATEST,
    SEARXNG_URL,
)


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

LATEST_TERMS = {
    "today", "latest", "recent", "newest", "current", "now", "release",
    "released", "announcement", "announced", "launch", "launched",
}

HISTORY_TERMS = {
    "history", "timeline", "older", "previous", "archive", "past",
    "from 2024", "from 2025", "between 2024", "between 2025",
    "in 2024", "in 2025", "during 2024", "during 2025",
}

TYPO_FIXES = {
    "goolge": "google",
    "googel": "google",
    "gooogle": "google",
    "recet": "recent",
    "recnet": "recent",
    "relese": "release",
    "modle": "model",
    "bechmark": "benchmark",
    "benchmak": "benchmark",
    "leatest": "latest",
    "pirce": "price",
    "prcie": "price",
    "curreny": "currency",
    "currncy": "currency",
    "excahnge": "exchange",
    "exchnage": "exchange",
}

OFFICIAL_COMPANY_DOMAINS = {
    "anthropic.com": 38,
    "platform.claude.com": 38,
    "openai.com": 38,
    "ai.google.dev": 36,
    "blog.google": 36,
    "deepmind.google": 36,
    "developers.googleblog.com": 34,
    "x.ai": 36,
    "meta.com": 34,
    "ai.meta.com": 34,
    "mistral.ai": 34,
    "microsoft.com": 32,
    "azure.microsoft.com": 32,
    "nvidia.com": 32,
    "developer.nvidia.com": 32,
    "aws.amazon.com": 24,
    "docs.aws.amazon.com": 24,
}

OFFICIAL_BENCHMARK_DOMAINS = {
    "artificialanalysis.ai": 38,
    "arena.ai": 36,
    "lmarena.ai": 36,
    "huggingface.co": 28,
    "swebench.com": 38,
    "arcprize.org": 38,
    "livecodebench.github.io": 34,
    "aider.chat": 30,
}

REPUTABLE_PRESS_DOMAINS = {
    "apnews.com": 28,
    "bbc.com": 24,
    "bbc.co.uk": 24,
    "washingtonpost.com": 26,
    "nytimes.com": 26,
    "aljazeera.com": 22,
    "nbcnews.com": 22,
    "nbcsandiego.com": 20,
    "politico.com": 20,
    "thehill.com": 18,
    "france24.com": 18,
    "ft.com": 22,
    "reuters.com": 24,
    "cnbc.com": 22,
    "bloomberg.com": 22,
    "theverge.com": 18,
    "techcrunch.com": 18,
    "wired.com": 18,
    "arstechnica.com": 18,
    "venturebeat.com": 16,
    "zdnet.com": 16,
    "thenews.com.pk": 12,
    "mashable.com": 12,
    "indianexpress.com": 12,
}

REPUTABLE_PUBLISHER_HINTS = {
    "ap news": 28,
    "associated press": 28,
    "bbc": 24,
    "the washington post": 26,
    "washington post": 26,
    "the new york times": 26,
    "new york times": 26,
    "reuters": 24,
    "al jazeera": 22,
    "nbc news": 22,
    "nbc san diego": 20,
    "politico": 20,
    "the hill": 18,
    "france 24": 18,
    "financial times": 22,
    "cbs news": 20,
    "abc news": 18,
    "the guardian": 18,
}

OFFICIAL_FINANCE_DOMAINS = {
    "sbp.org.pk": 40,
}

FINANCE_DATA_DOMAINS = {
    "exchangerate-api.com": 26,
    "open.er-api.com": 26,
    "frankfurter.dev": 28,
    "frankfurter.app": 24,
    "xe.com": 24,
    "oanda.com": 24,
}

LOW_TRUST_DOMAINS = {
    "msn.com",
    "medium.com",
    "linkedin.com",
    "substack.com",
    "reddit.com",
    "quora.com",
    "wikipedia.org",
}

BENCHMARK_SOURCE_HINTS = {
    "artificialanalysis.ai": "Artificial Analysis Intelligence Index",
    "arena.ai": "Arena/LMArena",
    "lmarena.ai": "LMArena",
    "huggingface.co": "LMArena Hugging Face leaderboard",
    "swebench.com": "SWE-bench Verified",
    "arcprize.org": "ARC Prize / ARC-AGI",
    "livecodebench.github.io": "LiveCodeBench",
    "aider.chat": "Aider Polyglot",
}

CURRENCY_CODES = {
    "AED", "AUD", "BDT", "BHD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP",
    "HKD", "INR", "JPY", "KWD", "MYR", "NOK", "NZD", "OMR", "PKR", "QAR",
    "SAR", "SEK", "SGD", "TRY", "USD", "ZAR",
}

CURRENCY_ALIASES = [
    ("pakistani rupees", "PKR"),
    ("pakistani rupee", "PKR"),
    ("pak rupees", "PKR"),
    ("pak rupee", "PKR"),
    ("rupees", "PKR"),
    ("rupee", "PKR"),
    ("pkr", "PKR"),
    ("us dollars", "USD"),
    ("us dollar", "USD"),
    ("u.s. dollar", "USD"),
    ("dollars", "USD"),
    ("dollar", "USD"),
    ("usd", "USD"),
    ("euro", "EUR"),
    ("euros", "EUR"),
    ("eur", "EUR"),
    ("pound sterling", "GBP"),
    ("british pound", "GBP"),
    ("pounds", "GBP"),
    ("gbp", "GBP"),
    ("dirham", "AED"),
    ("aed", "AED"),
    ("riyal", "SAR"),
    ("sar", "SAR"),
    ("indian rupee", "INR"),
    ("inr", "INR"),
]


@dataclass
class SourceEvidence:
    title: str
    url: str
    snippet: str
    provider: str
    domain: str = ""
    category: str = "web"
    trust_weight: int = 8
    extracted_text: str = ""
    extraction_status: str = "snippet_only"


@dataclass
class Claim:
    claim_type: str
    value: str
    trust_score: int
    support_count: int
    source_titles: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    source_categories: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)


@dataclass
class RateObservation:
    base: str
    quote: str
    rate: float
    provider: str
    title: str
    url: str
    as_of: str
    category: str
    trust_weight: int
    note: str = ""


def clean_text(value: str) -> str:
    import html as html_lib

    text = html_lib.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def current_year() -> int:
    return datetime.now().year


def has_latest_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in LATEST_TERMS)


def has_historical_intent(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in HISTORY_TERMS)


def is_benchmark_query(query: str) -> bool:
    q = (query or "").lower()
    return any(term in q for term in [
        "benchmark", "benchmarks", "leaderboard", "arena", "lmsys",
        "lmarena", "swe-bench", "swe bench", "arc-agi", "arc agi",
        "top model", "top models", "top 3", "top three", "stat", "stats",
        "performance data",
    ])


def extract_currency_mentions(query: str) -> List[Tuple[int, str]]:
    text = (query or "").lower()
    mentions: List[Tuple[int, str]] = []
    if "$" in query:
        mentions.append((query.find("$"), "USD"))
    for alias, code in CURRENCY_ALIASES:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", text, flags=re.I):
            mentions.append((match.start(), code))
    for match in re.finditer(r"\b[A-Z]{3}\b", (query or "").upper()):
        code = match.group(0)
        if code in CURRENCY_CODES:
            mentions.append((match.start(), code))

    seen = set()
    ordered: List[Tuple[int, str]] = []
    for pos, code in sorted(mentions, key=lambda item: item[0]):
        key = (pos, code)
        if key in seen:
            continue
        seen.add(key)
        ordered.append((pos, code))
    return ordered


def extract_currency_pair(query: str, base: Optional[str] = None, quote: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    if base and quote:
        base_code = base.strip().upper()
        quote_code = quote.strip().upper()
        if base_code in CURRENCY_CODES and quote_code in CURRENCY_CODES:
            return base_code, quote_code

    mentions = extract_currency_mentions(query)
    distinct: List[str] = []
    for _, code in mentions:
        if code not in distinct:
            distinct.append(code)
    if len(distinct) >= 2:
        return distinct[0], distinct[1]
    if len(distinct) == 1 and distinct[0] != "PKR":
        q = (query or "").lower()
        if any(term in q for term in ["rate", "price", "forex", "exchange", "currency"]):
            return distinct[0], "PKR"
    return None, None


def is_exchange_rate_query(query: str) -> bool:
    q = (query or "").lower()
    mentions = extract_currency_mentions(query)
    distinct = {code for _, code in mentions}
    has_rate_word = any(term in q for term in [
        "exchange rate", "forex", "fx", "currency", "convert", "conversion",
        "rate", "price", "to pkr", "in pkr", "against pkr",
    ])
    return has_rate_word and (len(distinct) >= 2 or any(code in distinct for code in {"USD", "EUR", "GBP", "AED", "SAR"}))


def normalize_query(query: str) -> str:
    text = clean_text(query)
    for wrong, right in TYPO_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.I)
    if has_latest_intent(text) and not has_historical_intent(text):
        year = current_year()
        text = re.sub(
            r"\b20\d{2}\b",
            lambda match: match.group(0) if int(match.group(0)) >= year else " ",
            text,
        )
    text = re.sub(r"\s+", " ", text).strip()
    if has_latest_intent(text) and str(current_year()) not in text:
        text = f"{text} {current_year()}"
    return text


def detect_mode(query: str, requested_mode: str = "auto") -> str:
    if requested_mode in {"exchange_rate", "finance"}:
        return "exchange_rate"
    if is_exchange_rate_query(query):
        return "exchange_rate"
    if requested_mode in {"latest", "benchmarks", "web"}:
        return requested_mode
    if is_benchmark_query(query):
        return "benchmarks"
    if has_latest_intent(query):
        return "latest"
    return "web"


def domain_for_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def classify_source(url: str) -> Tuple[str, int]:
    """
    Returns (category, trust_score) for a URL.
    Uses TrustRegistry which loads from JSON files.
    """
    domain = domain_for_url(url)
    
    from app.brain.trust_registry import trust_registry
    score, category = trust_registry.get_score(domain)
    
    if score > 0:
        return category, score
    
    # Fallback for unknown domains — estimate based on URL patterns
    if any(gov_tld in domain for gov_tld in [".gov", ".gov.pk", ".gov.uk", ".org.pk"]):
        return "government", 70
    if domain.endswith(".edu") or domain.endswith(".ac.uk"):
        return "academic", 65
    if any(bad in domain for bad in ["medium.com", "reddit.com", "quora.com", "linkedin.com"]):
        return "low_trust", 20
    
    return "unknown", 30



def make_source(title: str, url: str, snippet: str, provider: str) -> SourceEvidence:
    category, weight = classify_source(url)
    return SourceEvidence(
        title=clean_text(title),
        url=(url or "").strip(),
        snippet=clean_text(snippet),
        provider=provider,
        domain=domain_for_url(url),
        category=category,
        trust_weight=weight,
    )


def dedupe_sources(sources: List[SourceEvidence], limit: int) -> List[SourceEvidence]:
    seen_urls = set()
    seen_titles = set()
    deduped: List[SourceEvidence] = []
    for source in sources:
        if not source.title and not source.url:
            continue
        url_key = source.url.lower().rstrip("/")
        title_key = source.title.lower()
        if url_key and url_key in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue
        if url_key:
            seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        deduped.append(source)
        if len(deduped) >= limit:
            break
    return deduped


def source_relevance_score(query: str, source: SourceEvidence) -> float:
    haystack = f"{source.title} {source.snippet} {source.url}".lower()
    query_terms = {
        term for term in re.findall(r"[a-z0-9][a-z0-9.+#/-]*", query.lower())
        if len(term) > 2 and term not in {"latest", "recent", "today", "the", "and", "for", "with"}
    }
    score = float(source.trust_weight)
    for term in query_terms:
        if term in haystack:
            score += 3.0
    if source.category in {"official_company", "official_benchmark"}:
        score += 12.0
    if source.category == "low_trust":
        score -= 8.0
    if is_benchmark_query(query) and source.category == "official_benchmark":
        score += 18.0
    if has_latest_intent(query) and any(word in haystack for word in ["released", "launch", "announced", "latest", str(current_year())]):
        score += 8.0
    return score


def ranked_sources(query: str, sources: List[SourceEvidence], limit: int) -> List[SourceEvidence]:
    ranked = sorted(sources, key=lambda source: source_relevance_score(query, source), reverse=True)
    return dedupe_sources(ranked, limit)


def searxng_search(query: str, limit: int) -> List[SourceEvidence]:
    if not SEARXNG_URL:
        return []
    endpoint = f"{SEARXNG_URL.rstrip('/')}/search"
    response = requests.get(
        endpoint,
        params={"q": query, "format": "json", "language": "en", "safesearch": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    sources = []
    for item in data.get("results", [])[:limit]:
        sources.append(make_source(
            item.get("title", ""),
            item.get("url", ""),
            item.get("content", "") or item.get("snippet", ""),
            "SearXNG",
        ))
    return sources


def google_cse_search(query: str, limit: int) -> List[SourceEvidence]:
    api_key = (
        os.getenv("GOOGLE_CSE_API_KEY")
        or os.getenv("GOOGLE_SEARCH_API_KEY")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    )
    engine_id = (
        os.getenv("GOOGLE_CSE_ID")
        or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CX")
    )
    if not api_key or not engine_id:
        return []
    params = {
        "key": api_key,
        "cx": engine_id,
        "q": query,
        "num": min(limit, 10),
    }
    if has_latest_intent(query):
        params["dateRestrict"] = "y1"
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params=params,
        timeout=12,
    )
    response.raise_for_status()
    data = response.json()
    return [
        make_source(item.get("title", ""), item.get("link", ""), item.get("snippet", ""), "Google Custom Search")
        for item in data.get("items", [])[:limit]
    ]


def check_google_cse_configured() -> bool:
    """Called at startup — warns if Google CSE is not configured."""
    key = (
        os.getenv("GOOGLE_CSE_API_KEY")
        or os.getenv("GOOGLE_SEARCH_API_KEY")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
    )
    cse = (
        os.getenv("GOOGLE_CSE_ID")
        or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")
        or os.getenv("GOOGLE_CX")
    )
    if not key or not cse:
        print("[Research] WARNING: Google CSE not configured. Add GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID to .env for Google Search.")
        return False
    return True


def brave_search(query: str, limit: int) -> List[SourceEvidence]:
    """
    Brave Search API — 2000 free calls/month, better quality than HTML scraping.
    Get key at: https://brave.com/search/api/
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not api_key:
        return []
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": min(limit, 20), "safesearch": "moderate"}
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        
        sources = []
        for item in data.get("web", {}).get("results", [])[:limit]:
            sources.append(make_source(
                item.get("title", ""),
                item.get("url", ""),
                item.get("description", ""),
                "Brave Search",
            ))
        return sources
    except Exception as e:
        print(f"[Research] Brave Search failed: {e}")
        return []


def cite_sources_for_whatsapp(claims: list, sources: list, max_citations: int = 3) -> str:
    """
    Builds a WhatsApp-formatted source citation block.
    Only shows for factual claims (prices, news, rates).
    Shows: source name, what it says, trust indicator.
    """
    if not claims and not sources:
        return ""
    
    lines = ["\n_Sources checked:_"]
    
    # Deduplicate by domain
    seen_domains = set()
    citation_count = 0
    
    for source in sources:
        # Source could be a dict or a dataclass instance
        if hasattr(source, "domain"):
            domain = getattr(source, "domain", "")
            title = getattr(source, "title", domain)[:50]
            score = getattr(source, "trust_weight", 0) # trust_weight or score
            snippet = getattr(source, "snippet", "")[:80].strip()
        else:
            domain = source.get("domain", "")
            title = source.get("title", domain)[:50]
            score = source.get("trust_score", 0) or source.get("trust_weight", 0)
            snippet = source.get("snippet", "")[:80].strip()
            
        if not domain:
            continue
            
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        
        # Trust indicator emoji
        if score >= 80:
            badge = " ✓✓"  # High trust
        elif score >= 60:
            badge = " ✓"   # Medium trust
        elif score == 0 or score == 30:
            badge = " ?"   # Unknown
        else:
            badge = ""    # Low trust — show but no badge
        
        # Friendly title
        friendly_title = domain
        if "pakwheels.com" in domain:
            friendly_title = "PakWheels"
        elif "dawn.com" in domain:
            friendly_title = "Dawn"
        elif "reuters.com" in domain:
            friendly_title = "Reuters"
        elif "apnews.com" in domain:
            friendly_title = "AP News"
        elif "bbc.com" in domain or "bbc.co.uk" in domain:
            friendly_title = "BBC"
        elif "google" in domain:
            friendly_title = "Google Official"
        elif "anthropic" in domain:
            friendly_title = "Anthropic Official"
        elif "openai" in domain:
            friendly_title = "OpenAI Official"
        elif title and len(title) > 3:
            friendly_title = title
            
        lines.append(f"• {friendly_title}{badge}")
        citation_count += 1
        if citation_count >= max_citations:
            break
            
    return "\n".join(lines) if citation_count > 0 else ""



def google_news_search(query: str, limit: int) -> List[SourceEvidence]:
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        headers={"User-Agent": USER_AGENT},
        timeout=12,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    sources = []
    for item in root.findall(".//item")[:limit]:
        sources.append(make_source(
            item.findtext("title", default=""),
            item.findtext("link", default=""),
            item.findtext("description", default=""),
            "Google News",
        ))
    return sources


def decode_bing_url(url: str) -> str:
    if not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    if params.get("url"):
        return urllib.parse.unquote(params["url"][0])
    return url


def bing_news_search(query: str, limit: int) -> List[SourceEvidence]:
    response = requests.get(
        "https://www.bing.com/news/search",
        params={"format": "rss", "q": query, "count": limit},
        headers={"User-Agent": USER_AGENT},
        timeout=12,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    sources = []
    for item in root.findall(".//item")[:limit]:
        sources.append(make_source(
            item.findtext("title", default=""),
            decode_bing_url(item.findtext("link", default="")),
            item.findtext("description", default=""),
            "Bing News",
        ))
    return sources


def parse_jina_search_text(text: str, provider: str, limit: int) -> List[SourceEvidence]:
    sources = []
    blocks = re.split(r"\n(?=Title:|\[\d+\])", text)
    for block in blocks:
        if len(sources) >= limit:
            break
        title_match = re.search(r"(?:Title:|\[\d+\]\s*)(.+)", block)
        url_match = re.search(r"(?:URL Source:|URL:)\s*(https?://\S+)", block)
        if not url_match:
            continue
        title = title_match.group(1).strip() if title_match else url_match.group(1)
        snippet = clean_text(block[:1000])
        sources.append(make_source(title, url_match.group(1).strip(), snippet, provider))
    return sources


def jina_search(query: str, limit: int) -> List[SourceEvidence]:
    if not JINA_API_KEY:
        return []
    headers = {"User-Agent": USER_AGENT}
    headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    url = "https://s.jina.ai/" + urllib.parse.quote(query)
    response = requests.get(url, headers=headers, timeout=18)
    response.raise_for_status()
    sources = parse_jina_search_text(response.text, "Jina Search", limit)
    if sources:
        return sources
    return [make_source("Jina Search results", url, response.text[:1200], "Jina Search")]


class DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current = None
        self.capture = None
        self.capture_tag = None
        self.buffer = []
        self.snippet_target = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "")
        if tag == "a" and "result__a" in classes:
            self.current = {
                "title": "",
                "url": decode_duckduckgo_url(attrs.get("href", "")),
                "snippet": "",
            }
            self.capture = "title"
            self.capture_tag = tag
            self.buffer = []
        elif "result__snippet" in classes:
            self.snippet_target = self.current or (self.results[-1] if self.results else None)
            if self.snippet_target is not None:
                self.capture = "snippet"
                self.capture_tag = tag
                self.buffer = []

    def handle_data(self, data):
        if self.capture:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if not self.capture or tag != self.capture_tag:
            return
        text = clean_text(" ".join(self.buffer))
        if self.capture == "title" and self.current:
            self.current["title"] = text
            self.results.append(self.current)
            self.current = None
        elif self.capture == "snippet" and self.snippet_target:
            self.snippet_target["snippet"] = text
        self.capture = None
        self.capture_tag = None
        self.buffer = []
        self.snippet_target = None


def decode_duckduckgo_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/l/"):
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("uddg"):
            return urllib.parse.unquote(params["uddg"][0])
    return url


def duckduckgo_search(query: str, limit: int) -> List[SourceEvidence]:
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=12,
    )
    response.raise_for_status()
    parser = DuckDuckGoParser()
    parser.feed(response.text)
    return [
        make_source(item.get("title", ""), item.get("url", ""), item.get("snippet", ""), "DuckDuckGo")
        for item in parser.results[:limit]
    ]


def provider_functions() -> Dict[str, object]:
    return {
        "google_cse": google_cse_search,
        "brave": brave_search,
        "google_news": google_news_search,
        "bing_news": bing_news_search,
        "searxng": searxng_search,
        "jina_search": jina_search,
        "duckduckgo": duckduckgo_search,
    }


def benchmark_queries(query: str) -> List[str]:
    year = current_year()
    return [
        f"Artificial Analysis Intelligence Index top AI models {year} benchmark scores",
        f"Arena LMArena leaderboard top language models {year}",
        f"SWE-bench Verified leaderboard top AI models {year}",
        f"ARC Prize ARC-AGI leaderboard top AI models {year}",
        f"{normalize_query(query)} official benchmark leaderboard stats {year}",
    ]


def latest_queries(query: str) -> List[str]:
    normalized = normalize_query(query)
    q = normalized.lower()
    queries = [normalized]
    if (
        "islamabad" in q
        and ("peace talk" in q or "peace talks" in q or "talks" in q)
        and ("america" in q or "u.s." in q or " us " in f" {q} " or "united states" in q)
    ):
        queries.insert(0, "US official sent to Islamabad peace talks Pakistan Iran name")
        queries.insert(1, "US envoys Islamabad Pakistan Iran talks White House")
    if "claude" in q or "anthropic" in q:
        queries.insert(0, f"Anthropic official latest Claude model release {current_year()} Claude Opus")
        queries.append(f"site:anthropic.com/news Claude latest model release {current_year()}")
    if "openai" in q or "gpt" in q:
        queries.append(f"site:openai.com OpenAI latest model release {current_year()}")
    if "gemini" in q or "google" in q:
        queries.append(f"site:blog.google Gemini latest model release {current_year()}")
    return queries


def trusted_seed_sources(query: str, mode: str) -> List[SourceEvidence]:
    q = query.lower()
    seeds: List[SourceEvidence] = []
    if mode == "benchmarks":
        seeds.extend([
            make_source(
                "Artificial Analysis LLM Leaderboard",
                "https://artificialanalysis.ai/leaderboards/models",
                "Official Artificial Analysis model leaderboard for intelligence, speed, price, and benchmark data.",
                "Trusted Benchmark Seed",
            ),
            make_source(
                "Arena/LMArena leaderboard",
                "https://arena.ai/leaderboard",
                "Official Arena leaderboard comparing frontier AI models across arenas.",
                "Trusted Benchmark Seed",
            ),
            make_source(
                "SWE-bench leaderboards",
                "https://www.swebench.com/",
                "Official SWE-bench leaderboards for software engineering benchmark results.",
                "Trusted Benchmark Seed",
            ),
            make_source(
                "ARC Prize leaderboard",
                "https://arcprize.org/leaderboard",
                "Official ARC Prize leaderboard for ARC-AGI benchmark results.",
                "Trusted Benchmark Seed",
            ),
            make_source(
                "LiveCodeBench leaderboard",
                "https://livecodebench.github.io/leaderboard.html",
                "Official LiveCodeBench leaderboard for code generation performance.",
                "Trusted Benchmark Seed",
            ),
        ])
    if mode == "latest" and ("claude" in q or "anthropic" in q):
        seeds.extend([
            make_source(
                "Anthropic news",
                "https://www.anthropic.com/news",
                "Official Anthropic news and product announcements.",
                "Trusted Company Seed",
            ),
            make_source(
                "Claude Opus model page",
                "https://www.anthropic.com/claude/opus",
                "Official Claude Opus model page.",
                "Trusted Company Seed",
            ),
            make_source(
                "Claude release notes",
                "https://support.claude.com/en/articles/12138966-release-notes",
                "Official Claude release notes.",
                "Trusted Company Seed",
            ),
        ])
    return seeds


def format_rate_value(rate: float) -> str:
    return f"{rate:,.4f}".rstrip("0").rstrip(".")


def parse_rate_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def make_rate_observation(
    base: str,
    quote: str,
    rate: float,
    provider: str,
    title: str,
    url: str,
    as_of: str,
    note: str = "",
) -> Tuple[RateObservation, SourceEvidence]:
    snippet = f"1 {base} = {format_rate_value(rate)} {quote}"
    if as_of:
        snippet += f"; as of {as_of}"
    if note:
        snippet += f"; {note}"
    source = make_source(title, url, snippet, provider)
    observation = RateObservation(
        base=base,
        quote=quote,
        rate=rate,
        provider=provider,
        title=title,
        url=url,
        as_of=as_of,
        category=source.category,
        trust_weight=source.trust_weight,
        note=note,
    )
    return observation, source


def normalize_rate_direction(observation: RateObservation, base: str, quote: str) -> Optional[RateObservation]:
    if observation.base == base and observation.quote == quote:
        return observation
    if observation.base == quote and observation.quote == base and observation.rate:
        return RateObservation(
            base=base,
            quote=quote,
            rate=1 / observation.rate,
            provider=observation.provider,
            title=observation.title,
            url=observation.url,
            as_of=observation.as_of,
            category=observation.category,
            trust_weight=observation.trust_weight,
            note=observation.note,
        )
    return None


def fetch_exchangerate_api_rate(base: str, quote: str) -> Tuple[List[RateObservation], List[SourceEvidence]]:
    url = f"https://open.er-api.com/v6/latest/{urllib.parse.quote(base)}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
    response.raise_for_status()
    data = response.json()
    rate = parse_rate_float((data.get("rates") or {}).get(quote))
    if rate is None:
        return [], []
    as_of = clean_text(data.get("time_last_update_utc") or data.get("time_last_update_unix") or "")
    observation, source = make_rate_observation(
        base,
        quote,
        rate,
        "ExchangeRate-API",
        f"ExchangeRate-API latest {base}/{quote}",
        url,
        as_of,
        "free open endpoint",
    )
    return [observation], [source]


def fetch_frankfurter_rate(base: str, quote: str) -> Tuple[List[RateObservation], List[SourceEvidence]]:
    attempts = [
        (
            "https://api.frankfurter.dev/v1/latest",
            {"base": base, "symbols": quote},
        ),
        (
            "https://api.frankfurter.app/latest",
            {"from": base, "to": quote},
        ),
    ]
    for url, params in attempts:
        try:
            response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
            response.raise_for_status()
            data = response.json()
            rate = parse_rate_float((data.get("rates") or {}).get(quote))
            if rate is None:
                continue
            as_of = clean_text(data.get("date") or "")
            observation, source = make_rate_observation(
                base,
                quote,
                rate,
                "Frankfurter",
                f"Frankfurter latest {base}/{quote}",
                response.url,
                as_of,
                "daily institutional-source rate",
            )
            return [observation], [source]
        except Exception:
            continue
    return [], []


def fetch_sbp_usd_pkr_rate(base: str, quote: str) -> Tuple[List[RateObservation], List[SourceEvidence]]:
    if {base, quote} != {"USD", "PKR"}:
        return [], []
    url = "https://www.sbp.org.pk/ecodata/rates/m2m/m2m-current.asp"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=12)
    response.raise_for_status()
    text = strip_html_text(response.text)
    date_match = re.search(r"\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b", text)
    as_of = date_match.group(0) if date_match else ""
    rate_match = re.search(r"M2M\s+Revaluation\s+Rate\s*[:\-]?\s*(\d{2,4}(?:\.\d+)?)", text, flags=re.I)
    if not rate_match:
        rate_match = re.search(r"(?:US\s+Dollar|USD)\s+(?:USD\s+)?(\d{2,4}(?:\.\d+)?)", text, flags=re.I)
    rate = parse_rate_float(rate_match.group(1) if rate_match else None)
    if rate is None:
        return [], []
    obs_base, obs_quote, obs_rate = "USD", "PKR", rate
    if base == "PKR" and quote == "USD":
        obs_base, obs_quote, obs_rate = "PKR", "USD", 1 / rate
    observation, source = make_rate_observation(
        obs_base,
        obs_quote,
        obs_rate,
        "State Bank of Pakistan",
        "State Bank of Pakistan mark-to-market exchange rate",
        url,
        as_of,
        "official/reference M2M rate",
    )
    return [observation], [source]


def collect_exchange_rate_sources(base: str, quote: str) -> Tuple[List[RateObservation], List[SourceEvidence], List[str]]:
    observations: List[RateObservation] = []
    sources: List[SourceEvidence] = []
    notes: List[str] = []
    providers = [
        ("ExchangeRate-API", fetch_exchangerate_api_rate),
        ("Frankfurter", fetch_frankfurter_rate),
        ("State Bank of Pakistan", fetch_sbp_usd_pkr_rate),
    ]
    for provider_name, provider_func in providers:
        try:
            provider_observations, provider_sources = provider_func(base, quote)
            normalized_observations = []
            for observation in provider_observations:
                normalized = normalize_rate_direction(observation, base, quote)
                if normalized:
                    normalized_observations.append(normalized)
            if normalized_observations:
                observations.extend(normalized_observations)
                sources.extend(provider_sources)
            else:
                notes.append(f"{provider_name} returned no {base}/{quote} rate.")
        except Exception as exc:
            notes.append(f"{provider_name} failed: {exc}")
    return observations, sources, notes


def build_exchange_rate_claims(base: str, quote: str, observations: List[RateObservation]) -> List[Claim]:
    if not observations:
        return []

    rates = sorted(observation.rate for observation in observations)
    median_rate = rates[len(rates) // 2]
    min_rate = min(rates)
    max_rate = max(rates)
    average_rate = sum(rates) / len(rates)
    spread_pct = ((max_rate - min_rate) / average_rate * 100) if average_rate else 0.0
    has_official = any(observation.category == "official_finance" for observation in observations)
    support_count = len({observation.provider for observation in observations})

    if spread_pct <= 0.75:
        value = f"1 {base} = about {format_rate_value(median_rate)} {quote}"
    else:
        value = f"1 {base} = {format_rate_value(min_rate)}-{format_rate_value(max_rate)} {quote}"

    details = [
        f"{observation.provider}: 1 {base} = {format_rate_value(observation.rate)} {quote}"
        + (f" (as of {observation.as_of})" if observation.as_of else "")
        + (f" - {observation.note}" if observation.note else "")
        for observation in observations
    ]
    categories = list(dict.fromkeys(observation.category for observation in observations))
    source_titles = list(dict.fromkeys(observation.title for observation in observations))
    source_urls = list(dict.fromkeys(observation.url for observation in observations))

    trust_score = min(96, 48 + support_count * 12 + (18 if has_official else 0) + max(observation.trust_weight for observation in observations) // 3)
    if support_count == 1 and not has_official:
        trust_score = min(trust_score, 70)
    if spread_pct > 1.5:
        trust_score = max(45, trust_score - 18)

    return [Claim(
        claim_type="exchange_rate",
        value=value,
        trust_score=trust_score,
        support_count=support_count,
        source_titles=source_titles,
        source_urls=source_urls,
        source_categories=categories,
        details=details,
    )]


def collect_sources(query: str, mode: str, max_sources: int) -> Tuple[str, List[SourceEvidence], List[str]]:
    normalized = normalize_query(query)
    query_set = benchmark_queries(query) if mode == "benchmarks" else latest_queries(query)
    providers = [item.strip() for item in SEARCH_PROVIDER_ORDER.split(",") if item.strip()]
    functions = provider_functions()
    notes = []
    sources: List[SourceEvidence] = trusted_seed_sources(normalized, mode)
    per_call_limit = max(4, min(10, max_sources))

    for search_query in query_set:
        for provider in providers:
            func = functions.get(provider)
            if not func:
                notes.append(f"Unknown provider skipped: {provider}")
                continue
            try:
                found = func(search_query, per_call_limit)
                if found:
                    sources.extend(found)
                else:
                    notes.append(f"{provider} returned no results for '{search_query}'")
            except Exception as exc:
                notes.append(f"{provider} failed for '{search_query}': {exc}")
            if len(dedupe_sources(sources, max_sources)) >= max_sources:
                break
        if len(dedupe_sources(sources, max_sources)) >= max_sources:
            break

    return normalized, ranked_sources(normalized, sources, max_sources), notes


def strip_html_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    return clean_text(text)


def read_with_jina(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    reader_url = "https://r.jina.ai/" + url
    response = requests.get(reader_url, headers=headers, timeout=18)
    response.raise_for_status()
    return clean_text(response.text)


def extract_url_text(url: str) -> Tuple[str, str]:
    if not url or not url.startswith("http"):
        return "", "no_url"
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=8,
            allow_redirects=True,
        )
        response.raise_for_status()
        html = response.text
        extracted = ""
        try:
            import trafilatura  # type: ignore
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            ) or ""
        except Exception:
            extracted = ""
        if not extracted:
            extracted = strip_html_text(html)
        extracted = clean_text(extracted)
        if len(extracted) >= 300:
            return extracted[:8000], "trafilatura_or_html"
    except Exception:
        pass
    try:
        text = read_with_jina(url)
        if text:
            return text[:8000], "jina_reader"
    except Exception:
        pass
    return "", "snippet_only"


def enrich_sources(sources: List[SourceEvidence], max_extract: int = 10) -> List[SourceEvidence]:
    candidates = [source for source in sources if source.url.startswith("http")][:max_extract]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_url_text, source.url): source for source in candidates}
        for future in as_completed(futures):
            source = futures[future]
            try:
                text, status = future.result()
                source.extracted_text = text
                source.extraction_status = status
            except Exception:
                source.extracted_text = ""
                source.extraction_status = "extract_failed"
    return sources


MODEL_PATTERNS = [
    r"\bClaude\s+(?:Opus|Sonnet|Haiku|Mythos)(?:\s+Preview)?\s+\d+(?:\.\d+)?\b",
    r"\bGPT[-\s]?\d+(?:\.\d+)?(?:[-\s]?(?:Instant|Codex|Thinking|Pro|xhigh|high))?\b",
    r"\bGemini\s+\d+(?:\.\d+)?(?:\s+(?:Pro|Flash|Ultra|Deep Think|Preview))?\b",
    r"\bGrok\s+\d+(?:\.\d+)?(?:[-\s]?\w+)?\b",
    r"\bLlama\s+\d+(?:\.\d+)?\b",
    r"\bDeepSeek\s+(?:V|R)\d+(?:\.\d+)?(?:[-\s]?\w+)?\b",
    r"\bMistral\s+(?:Medium|Large|Small)?\s*\d+(?:\.\d+)?\b",
    r"\bKimi\s+K\d+(?:\.\d+)?\b",
    r"\bQwen\d+(?:\.\d+)?(?:[-\w]+)?\b",
]


def canonical_model_name(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("GPT ", "GPT-")
    return text.strip(" .,:;")


def model_entity_allowed(query: str, model_name: str) -> bool:
    q = query.lower()
    name = model_name.lower()
    entity_filters = {
        "claude": "claude",
        "anthropic": "claude",
        "openai": "gpt",
        "gpt": "gpt",
        "gemini": "gemini",
        "google": "gemini",
        "grok": "grok",
        "xai": "grok",
        "llama": "llama",
        "meta": "llama",
        "mistral": "mistral",
        "deepseek": "deepseek",
    }
    matched = [expected for token, expected in entity_filters.items() if token in q]
    if not matched or is_benchmark_query(query):
        return True
    return any(expected in name for expected in matched)


def find_model_names(text: str) -> List[str]:
    names = []
    for pattern in MODEL_PATTERNS:
        for match in re.findall(pattern, text, flags=re.I):
            name = canonical_model_name(match)
            if name and name.lower() not in {item.lower() for item in names}:
                names.append(name)
    return names


def version_score(model_name: str) -> float:
    nums = re.findall(r"\d+(?:\.\d+)?", model_name)
    if not nums:
        return 0.0
    try:
        return max(float(num) for num in nums)
    except Exception:
        return 0.0


def source_text(source: SourceEvidence) -> str:
    return clean_text(f"{source.title}\n{source.snippet}\n{source.extracted_text}")


def publisher_hint_for_source(source: SourceEvidence) -> Tuple[str, int]:
    haystack = f"{source.title} {source.snippet} {source.domain}".lower()
    for publisher, weight in REPUTABLE_PUBLISHER_HINTS.items():
        if publisher in haystack:
            return publisher.title(), weight
    return "", 0


def source_categories_with_publisher(source: SourceEvidence) -> List[str]:
    categories = [source.category]
    reputable_categories = {
        "international_news", "news_english", "news_urdu", "finance_global",
        "technology", "science_health", "specialized"
    }
    if source.category in reputable_categories and "reputable_press" not in categories:
        categories.append("reputable_press")
    _, publisher_weight = publisher_hint_for_source(source)
    if publisher_weight and "reputable_press" not in categories:
        categories.append("reputable_press")
    return categories


def generic_news_relevance_hits(query: str, text: str) -> int:
    terms = {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9.+#/-]*", (query or "").lower())
        if len(term) > 2 and term not in {"tell", "about", "what", "happened", "latest", "recent", "today", "the", "and", "for", "with", "from"}
    }
    lower = (text or "").lower()
    return sum(1 for term in terms if term in lower)


PERSON_LOOKUP_WORDS = {
    "who", "name", "guy", "person", "official", "representative", "envoy",
    "delegate", "delegation", "sent", "send", "travel", "trip", "headed",
}

PERSON_STOP_NAMES = {
    "United States", "White House", "New York", "The New", "The Washington",
    "Washington Post", "New York Times", "The Hill", "Financial Times",
    "Al Jazeera", "Associated Press", "Business Recorder", "France 24",
    "CBS News", "BBC News", "Fox News", "Iran Peace", "Pakistan Iran",
    "US Iran", "U S", "Donald Trump", "President Trump",
}

PERSON_BAD_PARTS = {
    "And", "Calls", "Call", "Off", "Peace", "Talks", "Talk", "Travel", "Trip",
    "Go", "Going", "Headed", "Sending", "Send", "Sends", "Sent", "Pakistan",
    "Islamabad", "Iran", "America", "United", "States", "State", "White",
    "House", "News", "Times", "Post", "Hill", "France", "Home", "Shows",
    "Newsfeed", "Settings", "Sign", "Menu", "Close", "Ukraine", "Mideast",
    "Middle", "East", "Strait", "Hormuz", "Increased", "By",
    "President", "Minister", "Foreign", "Deputy", "Prime", "Senator", "FM",
    "Last", "Minute", "We", "Have", "All", "Time", "Magazine", "Oregon",
    "Public", "Broadcasting", "OPB", "WATCH", "PBS", "MSN",
}


def is_person_lookup_query(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in PERSON_LOOKUP_WORDS) and any(
        topic in q for topic in ["islamabad", "pakistan", "america", "u.s.", " us ", "united states", "peace talk", "peace talks", "talks"]
    )


def extract_person_names(text: str) -> List[str]:
    names: List[str] = []
    cleaned = clean_text(text or "")
    for match in re.finditer(r"\b(?:[A-Z][a-z]+|[A-Z]{2})(?:\s+[A-Z][a-z]+){1,2}\b", cleaned):
        name = clean_text(match.group(0))
        if name in PERSON_STOP_NAMES:
            continue
        parts = name.replace(".", "").split()
        if any(part in PERSON_BAD_PARTS for part in parts):
            continue
        if len(parts) >= 3 and parts[0] in {"The", "A", "An"}:
            continue
        if len(parts) == 2 and all(len(part) <= 3 for part in parts):
            continue
        if name.lower() not in {item.lower() for item in names}:
            names.append(name)
    return names[:8]


def extract_surname_pair(text: str) -> str:
    cleaned = clean_text(text or "")
    for first, second in re.findall(r"\b([A-Z][a-z]{3,}),\s+([A-Z][a-z]{3,})\b", cleaned):
        if first in PERSON_BAD_PARTS or second in PERSON_BAD_PARTS:
            continue
        if first in {"Trump", "Biden"}:
            continue
        return f"{first} and {second}"
    return ""


def person_lookup_action_score(query: str, text: str) -> int:
    lower = (text or "").lower()
    score = 0
    if any(word in lower for word in ["travel", "travels", "fly", "headed", "head to", "trip", "delegation", "envoy", "sent", "send"]):
        score += 12
    if "islamabad" in lower or "pakistan" in lower:
        score += 8
    if "peace talks" in lower or "talks" in lower:
        score += 6
    if "white house" in lower or "u.s." in lower or " us " in f" {lower} ":
        score += 4
    return score


def strip_publisher_suffix(title: str) -> str:
    parts = re.split(r"\s+-\s+", clean_text(title or ""))
    if len(parts) > 1 and len(parts[-1]) <= 42:
        return " - ".join(parts[:-1]).strip()
    return clean_text(title or "")


def remove_leading_publisher(text: str) -> str:
    cleaned = clean_text(text or "")
    lower = cleaned.lower()
    for publisher in sorted(REPUTABLE_PUBLISHER_HINTS, key=len, reverse=True):
        if lower.startswith(publisher):
            return cleaned[len(publisher):].lstrip(" :-")
    return cleaned


def concise_news_value(source: SourceEvidence) -> str:
    title = strip_publisher_suffix(source.title)
    snippet = clean_text(source.snippet)
    if title and snippet.lower().startswith(title.lower()):
        snippet = snippet[len(title):].lstrip(" :-")
    snippet = remove_leading_publisher(snippet)
    if snippet and title and snippet.lower() not in title.lower() and len(snippet) > 35:
        combined = f"{title}: {snippet}"
    else:
        combined = title or snippet
    return combined[:320]


def build_generic_news_claims(query: str, sources: List[SourceEvidence], mode: str) -> List[Claim]:
    if mode not in {"latest", "web"}:
        return []
    claims: List[Claim] = []
    for source in sources:
        text = source_text(source)
        value = concise_news_value(source)
        if not value:
            continue
        hits = generic_news_relevance_hits(query, text)
        if hits < 2 and source.category not in {"reputable_press", "news_aggregator"}:
            continue
        publisher, publisher_weight = publisher_hint_for_source(source)
        categories = source_categories_with_publisher(source)
        score = min(92, 34 + source.trust_weight + publisher_weight + hits * 3)
        if "shooting" in (query or "").lower() and re.search(r"\b(killed|dead|active shooter|suspects?|victims?)\b", text, flags=re.I):
            score = min(92, score + 6)
        if source.category == "news_aggregator" and not publisher_weight:
            score = min(score, 48)
        if source.category == "low_trust":
            score = min(score, 42)
        detail = source.snippet or source.title
        if publisher:
            detail = f"{publisher}: {detail}"
        claims.append(Claim(
            claim_type="news_report",
            value=value,
            trust_score=score,
            support_count=1,
            source_titles=[source.title or source.domain],
            source_urls=[source.url],
            source_categories=categories,
            details=[clean_text(detail)[:420]],
        ))
    claims.sort(key=lambda claim: (claim.trust_score, claim.support_count), reverse=True)
    return claims[:6]


def build_person_lookup_claims(query: str, sources: List[SourceEvidence], mode: str) -> List[Claim]:
    if mode not in {"latest", "web"} or not is_person_lookup_query(query):
        return []
    grouped: Dict[str, Dict[str, object]] = {}
    for source in sources:
        title = strip_publisher_suffix(source.title)
        text = clean_text(f"{title}. {source.snippet}")
        pair = extract_surname_pair(text)
        if pair:
            names = [pair]
        else:
            names = extract_person_names(text)
        if not names:
            continue
        action_score = person_lookup_action_score(query, text)
        if action_score < 8:
            continue
        publisher, publisher_weight = publisher_hint_for_source(source)
        categories = source_categories_with_publisher(source)
        relevant_names = names[:3]
        # For "envoys/delegation" results, the useful answer can be a pair.
        if len(relevant_names) >= 2 and re.search(r"\b(envoys?|delegation|witkoff|kushner|travel|trip|headed)\b", text, flags=re.I):
            value = " and ".join(relevant_names[:2])
        else:
            value = relevant_names[0]
        key = value.lower()
        entry = grouped.setdefault(key, {
            "value": value,
            "weight": 0,
            "sources": [],
            "urls": [],
            "categories": [],
            "details": [],
        })
        entry["weight"] = int(entry["weight"]) + source.trust_weight + publisher_weight + action_score
        entry["sources"].append(source.title or source.domain)
        entry["urls"].append(source.url)
        for category in categories:
            entry["categories"].append(category)
        detail = concise_news_value(source)
        if publisher:
            detail = f"{publisher}: {detail}"
        entry["details"].append(detail)

    claims: List[Claim] = []
    for entry in grouped.values():
        source_titles = list(dict.fromkeys(entry["sources"]))
        source_urls = list(dict.fromkeys(entry["urls"]))
        categories = list(dict.fromkeys(entry["categories"]))
        details = list(dict.fromkeys(entry["details"]))
        support_count = len(source_urls)
        score = min(95, 36 + int(entry["weight"]) // max(1, support_count) + support_count * 7)
        if "reputable_press" not in categories:
            score = min(score, 58)
        claims.append(Claim(
            claim_type="person_lookup",
            value=str(entry["value"]),
            trust_score=score,
            support_count=support_count,
            source_titles=source_titles[:6],
            source_urls=source_urls[:6],
            source_categories=categories,
            details=details[:4],
        ))
    claims.sort(key=lambda claim: (claim.trust_score, claim.support_count), reverse=True)
    return claims[:4]


def nearby_stats(text: str, model_name: str) -> List[str]:
    stats = []
    lower = text.lower()
    idx = lower.find(model_name.lower())
    windows = []
    if idx >= 0:
        windows.append(text[max(0, idx - 240):idx + len(model_name) + 260])
    windows.append(text[:1200])
    for window in windows:
        for pct in re.findall(r"\b\d{1,3}(?:\.\d+)?%", window):
            if pct not in stats:
                stats.append(pct)
        for score in re.findall(r"\b(?:score|index|elo|rating)\s*(?:of|:|=)?\s*\d{2,4}(?:\.\d+)?\b", window, flags=re.I):
            clean = clean_text(score)
            if clean not in stats:
                stats.append(clean)
    return stats[:5]


def benchmark_name_for(source: SourceEvidence, text: str) -> str:
    domain = source.domain
    for known, name in BENCHMARK_SOURCE_HINTS.items():
        if domain == known or domain.endswith(f".{known}"):
            return name
    lower = text.lower()
    if "artificial analysis" in lower:
        return "Artificial Analysis Intelligence Index"
    if "swe-bench" in lower or "swe bench" in lower:
        return "SWE-bench"
    if "arc-agi" in lower or "arc agi" in lower:
        return "ARC-AGI"
    if "arena" in lower or "lmarena" in lower or "lmsys" in lower:
        return "Arena/LMArena"
    return source.provider


def build_model_claims(query: str, sources: List[SourceEvidence], mode: str) -> List[Claim]:
    grouped: Dict[str, Dict[str, object]] = {}
    for source in sources:
        text = source_text(source)
        for model in find_model_names(text):
            if not model_entity_allowed(query, model):
                continue
            key = model.lower()
            entry = grouped.setdefault(key, {
                "value": model,
                "weight": 0,
                "sources": [],
                "urls": [],
                "categories": [],
                "details": [],
            })
            entry["weight"] = int(entry["weight"]) + source.trust_weight
            entry["sources"].append(source.title or source.domain)
            entry["urls"].append(source.url)
            entry["categories"].append(source.category)
            if mode == "benchmarks":
                benchmark = benchmark_name_for(source, text)
                stats = nearby_stats(text, model)
                detail = benchmark
                if stats:
                    detail += f": {', '.join(stats)}"
                if detail not in entry["details"]:
                    entry["details"].append(detail)
    claims = []
    for entry in grouped.values():
        source_titles = list(dict.fromkeys(entry["sources"]))
        source_urls = list(dict.fromkeys(entry["urls"]))
        categories = list(dict.fromkeys(entry["categories"]))
        details = list(dict.fromkeys(entry["details"]))
        support_count = len(source_urls)
        official_bonus = 22 if any(cat in {"official_company", "official_benchmark"} for cat in categories) else 0
        benchmark_bonus = 14 if mode == "benchmarks" and any(cat == "official_benchmark" for cat in categories) else 0
        score = min(98, 25 + int(entry["weight"]) + support_count * 6 + official_bonus + benchmark_bonus)
        if support_count == 1 and "official_company" not in categories and "official_benchmark" not in categories:
            score = min(score, 58)
        if mode == "benchmarks" and support_count == 1:
            score = min(score, 72)
        if mode == "benchmarks" and not details:
            score = min(score, 62)
        claims.append(Claim(
            claim_type="benchmark_model" if mode == "benchmarks" else "model_release",
            value=str(entry["value"]),
            trust_score=score,
            support_count=support_count,
            source_titles=source_titles[:6],
            source_urls=source_urls[:6],
            source_categories=categories,
            details=details[:5],
        ))
    claims.sort(key=lambda claim: (claim.trust_score, version_score(claim.value), claim.support_count), reverse=True)
    if mode == "latest" and claims:
        top_version_by_family: Dict[str, float] = {}
        for claim in claims:
            family = claim.value.split()[0].lower()
            top_version_by_family[family] = max(top_version_by_family.get(family, 0.0), version_score(claim.value))
        for claim in claims:
            family = claim.value.split()[0].lower()
            if version_score(claim.value) < top_version_by_family.get(family, 0.0):
                claim.trust_score = min(claim.trust_score, 68)
        claims.sort(key=lambda claim: (claim.trust_score, version_score(claim.value), claim.support_count), reverse=True)
    return claims


def build_warnings(query: str, mode: str, sources: List[SourceEvidence], claims: List[Claim], notes: List[str]) -> List[str]:
    warnings = []
    if has_latest_intent(query) and not has_historical_intent(query):
        stale_years = re.findall(r"\b20\d{2}\b", query)
        stale_years = [year for year in stale_years if int(year) < current_year()]
        if stale_years:
            warnings.append(f"Stale year filters removed for latest/current query: {', '.join(sorted(set(stale_years)))}")
    if SEARCH_REQUIRE_TRUSTED_FOR_LATEST and mode == "latest":
        has_reputable_claim = any(
            "reputable_press" in claim.source_categories or "official_company" in claim.source_categories
            for claim in claims
        )
        if not has_reputable_claim and not any(source.category in {"official_company", "reputable_press"} for source in sources):
            warnings.append("No official/reputable source was found; treat latest answer as low confidence.")
    if mode == "benchmarks":
        official_count = sum(1 for source in sources if source.category == "official_benchmark")
        if official_count == 0:
            warnings.append("No official benchmark leaderboard page was extracted; use only low-confidence wording.")
        elif claims and not any("official_benchmark" in claim.source_categories for claim in claims):
            warnings.append("Official benchmark pages were checked, but exact model rows came from secondary sources; label benchmark claims as mixed-confidence.")
        if not claims:
            warnings.append("No exact benchmark model claims were extracted from the checked sources.")
        numeric_details = [
            detail
            for claim in claims
            for detail in claim.details
            if re.search(r"\d+(?:\.\d+)?%|\b(?:score|index|elo|rating)\s*(?:of|:|=)?\s*\d{2,4}", detail, flags=re.I)
        ]
        if len(numeric_details) < 5:
            warnings.append("Fewer than five exact numeric benchmark stats were extracted; provide a partial sourced answer instead of inventing missing scores.")
        warnings.append("Benchmark leaderboards can disagree by task; do not present a single universal winner without naming the benchmark.")
    if mode == "exchange_rate":
        if claims:
            warnings.append("Exchange rates can move during the day; state the source/date shown in details.")
            if not any(source.category == "official_finance" for source in sources):
                warnings.append("No official central-bank source was found; label the value as a market/data-provider estimate.")
        else:
            warnings.append("No numeric exchange-rate claim was extracted; do not invent a rate.")
    if notes:
        warnings.extend(notes[:5])
    return warnings


def result_to_dict(
    query: str,
    normalized_query: str,
    mode: str,
    sources: List[SourceEvidence],
    claims: List[Claim],
    warnings: List[str],
    elapsed_ms: float,
) -> Dict[str, object]:
    top_trust = max((claim.trust_score for claim in claims), default=0)
    answerable = bool(claims and (top_trust >= 55 or mode != "latest"))
    return {
        "answerable": answerable,
        "mode": mode,
        "original_query": query,
        "normalized_query": normalized_query,
        "sources_checked": len(sources),
        "trust_score": top_trust,
        "freshness": datetime.now().strftime("%Y-%m-%d"),
        "claims": [
            {
                "claim_type": claim.claim_type,
                "value": claim.value,
                "trust_score": claim.trust_score,
                "support_count": claim.support_count,
                "source_titles": claim.source_titles[:3],
                "source_urls": claim.source_urls[:2],
                "source_categories": claim.source_categories,
                "details": claim.details[:3],
            }
            for claim in claims[:4]
        ],
        "citations": [
            {
                "title": source.title,
                "url": source.url,
                "provider": source.provider,
                "domain": source.domain,
                "category": source.category,
                "snippet": source.snippet[:180],
            }
            for source in sources[:6]
        ],
        "warnings": warnings,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def research_query(query: str, mode: str = "auto", max_sources: Optional[int] = None) -> Dict[str, object]:
    started = time.perf_counter()
    resolved_mode = detect_mode(query, mode)
    max_count = max(5, min(max_sources or SEARCH_MAX_SOURCES, 20))
    if resolved_mode == "exchange_rate":
        normalized_query = normalize_query(query)
        base, quote = extract_currency_pair(normalized_query)
        if not base or not quote:
            elapsed_ms = (time.perf_counter() - started) * 1000
            warnings = ["Could not identify a currency pair. Ask for a pair like USD to PKR."]
            return result_to_dict(query, normalized_query, resolved_mode, [], [], warnings, elapsed_ms)
        observations, sources, notes = collect_exchange_rate_sources(base, quote)
        claims = build_exchange_rate_claims(base, quote, observations)
        warnings = build_warnings(query, resolved_mode, sources, claims, notes)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return result_to_dict(query, normalized_query, resolved_mode, sources, claims, warnings, elapsed_ms)

    normalized_query, sources, notes = collect_sources(query, resolved_mode, max_count)
    sources = enrich_sources(sources, max_extract=min(6, max_count))
    claims = build_model_claims(normalized_query, sources, resolved_mode)
    if not claims and resolved_mode in {"latest", "web"}:
        claims = build_person_lookup_claims(normalized_query, sources, resolved_mode)
    if not claims and resolved_mode in {"latest", "web"}:
        claims = build_generic_news_claims(normalized_query, sources, resolved_mode)
    warnings = build_warnings(query, resolved_mode, sources, claims, notes)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result_to_dict(query, normalized_query, resolved_mode, sources, claims, warnings, elapsed_ms)


def format_research_result(result: Dict[str, object]) -> str:
    compact_result = dict(result)
    compact_result["warnings"] = list(compact_result.get("warnings", []))[:6]
    compact = json.dumps(compact_result, ensure_ascii=True, indent=2)
    lines = [
        "STRUCTURED_RESEARCH_RESULT",
        compact,
        "",
        "SOPHIE_RULES:",
        "- Use only the claims and citations above for factual answers.",
        "- Include trust percentages when answering latest/model/benchmark questions.",
        "- For exchange_rate claims, answer with the numeric rate in claims and include the source/date from details.",
        "- For latest model questions, answer with the highest-trust current model claim first; do not list older models as equally latest.",
        "- For news_report claims, summarize only the reported facts in value/details and name the cited publishers.",
        "- If claims are missing or warnings say sources disagree, say that clearly.",
        "- Do not invent exchange rates, model names, benchmark scores, dates, or rankings not present in claims/citations.",
    ]
    return "\n".join(lines)
