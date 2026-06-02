from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python always has this in our runtime.
    ZoneInfo = None  # type: ignore


TimeTarget = Dict[str, object]


def _target(label: str, abbr: str, offset_hours: float, zone: str = "") -> TimeTarget:
    return {
        "label": label,
        "abbr": abbr,
        "offset_hours": offset_hours,
        "zone": zone,
    }


TIME_TARGETS: Dict[str, TimeTarget] = {
    "pakistan": _target("Pakistan", "PKT", 5, "Asia/Karachi"),
    "pk": _target("Pakistan", "PKT", 5, "Asia/Karachi"),
    "pkt": _target("Pakistan", "PKT", 5, "Asia/Karachi"),
    "karachi": _target("Karachi, Pakistan", "PKT", 5, "Asia/Karachi"),
    "lahore": _target("Lahore, Pakistan", "PKT", 5, "Asia/Karachi"),
    "islamabad": _target("Islamabad, Pakistan", "PKT", 5, "Asia/Karachi"),
    "dubai": _target("Dubai, UAE", "GST", 4, "Asia/Dubai"),
    "uae": _target("United Arab Emirates", "GST", 4, "Asia/Dubai"),
    "emirates": _target("United Arab Emirates", "GST", 4, "Asia/Dubai"),
    "abu dhabi": _target("Abu Dhabi, UAE", "GST", 4, "Asia/Dubai"),
    "asia dubai": _target("Dubai, UAE", "GST", 4, "Asia/Dubai"),
    "china": _target("China", "CST", 8, "Asia/Shanghai"),
    "beijing": _target("Beijing, China", "CST", 8, "Asia/Shanghai"),
    "shanghai": _target("Shanghai, China", "CST", 8, "Asia/Shanghai"),
    "india": _target("India", "IST", 5.5, "Asia/Kolkata"),
    "delhi": _target("New Delhi, India", "IST", 5.5, "Asia/Kolkata"),
    "new delhi": _target("New Delhi, India", "IST", 5.5, "Asia/Kolkata"),
    "london": _target("London, UK", "GMT/BST", 0, "Europe/London"),
    "uk": _target("United Kingdom", "GMT/BST", 0, "Europe/London"),
    "united kingdom": _target("United Kingdom", "GMT/BST", 0, "Europe/London"),
    "new york": _target("New York, USA", "ET", -5, "America/New_York"),
    "nyc": _target("New York, USA", "ET", -5, "America/New_York"),
    "eastern": _target("New York, USA", "ET", -5, "America/New_York"),
    "los angeles": _target("Los Angeles, USA", "PT", -8, "America/Los_Angeles"),
    "la": _target("Los Angeles, USA", "PT", -8, "America/Los_Angeles"),
    "pacific": _target("Los Angeles, USA", "PT", -8, "America/Los_Angeles"),
    "utc": _target("UTC", "UTC", 0, "UTC"),
    "gmt": _target("GMT", "GMT", 0, "Etc/GMT"),
}

US_REPRESENTATIVE_TARGETS: List[TimeTarget] = [
    _target("New York, USA", "ET", -5, "America/New_York"),
    _target("Los Angeles, USA", "PT", -8, "America/Los_Angeles"),
]

MULTI_TARGET_ALIASES: Dict[str, List[TimeTarget]] = {
    "america": US_REPRESENTATIVE_TARGETS,
    "us": US_REPRESENTATIVE_TARGETS,
    "usa": US_REPRESENTATIVE_TARGETS,
    "united states": US_REPRESENTATIVE_TARGETS,
}


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9+\-:\s/]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", normalized.replace("/", " ")).strip()


def _contains_key(query: str, tokens: set[str], key: str) -> bool:
    key = _normalize_text(key)
    if not key:
        return False
    if " " in key:
        return f" {key} " in f" {query} "
    return key in tokens


def _dedupe_targets(targets: List[TimeTarget]) -> List[TimeTarget]:
    seen: set[str] = set()
    deduped: List[TimeTarget] = []
    for target in targets:
        identity = str(target.get("zone") or target.get("label") or target.get("abbr"))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(target)
    return deduped


def _utc_offset_target(query: str) -> Optional[TimeTarget]:
    offset_match = re.search(r"\butc\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?\b", query)
    if not offset_match:
        return None
    sign = 1 if offset_match.group(1) == "+" else -1
    hours = int(offset_match.group(2))
    minutes = int(offset_match.group(3) or "0")
    offset_hours = sign * (hours + minutes / 60)
    label = f"UTC{offset_match.group(1)}{hours:02d}:{minutes:02d}"
    return _target(label, label, offset_hours)


def infer_time_targets(message: str) -> List[TimeTarget]:
    query = _normalize_text(message)
    tokens = set(query.split())
    targets: List[TimeTarget] = []

    for alias, alias_targets in MULTI_TARGET_ALIASES.items():
        if _contains_key(query, tokens, alias):
            targets.extend(alias_targets)

    for key, target in TIME_TARGETS.items():
        if _contains_key(query, tokens, key):
            targets.append(target)

    offset_target = _utc_offset_target(query)
    if offset_target:
        targets.append(offset_target)

    return _dedupe_targets(targets)


def infer_time_target(message: str) -> Optional[TimeTarget]:
    targets = infer_time_targets(message)
    return targets[0] if targets else None


def is_current_time_query(message: str) -> bool:
    query = _normalize_text(message)
    if "time" not in query and "clock" not in query:
        return False
    if any(
        word in query
        for word in [
            "start time",
            "startof",
            "start of",
            "event time",
            "schedule",
            "when is",
            "google i o",
            "google io",
        ]
    ):
        return False
    return bool(infer_time_targets(message)) or any(
        phrase in query
        for phrase in [
            "what time is it",
            "current time",
            "time now",
            "local time",
            "time in",
            "time of",
            "time for",
        ]
    )


def _timezone_for_target(target: TimeTarget) -> timezone:
    zone_name = str(target.get("zone") or "")
    if zone_name and ZoneInfo is not None:
        try:
            return ZoneInfo(zone_name)  # type: ignore[return-value]
        except Exception:
            pass
    return timezone(timedelta(hours=float(target["offset_hours"])), str(target["abbr"]))


def _format_time_target(target: TimeTarget) -> Dict[str, str]:
    tz = _timezone_for_target(target)
    now = datetime.now(timezone.utc).astimezone(tz)
    abbr = now.tzname() or str(target["abbr"])
    if abbr.startswith(("+", "-")):
        abbr = str(target["abbr"])
    return {
        "label": str(target["label"]),
        "abbr": abbr,
        "iso": now.isoformat(timespec="seconds"),
        "time_12h": now.strftime("%I:%M %p").lstrip("0"),
        "time_24h": now.strftime("%H:%M"),
        "date": now.strftime("%A, %d %B %Y"),
        "utc_offset": now.strftime("%z"),
    }


def get_current_time_results(message: str = "", location: str = "") -> List[Dict[str, str]]:
    targets = infer_time_targets(location or message)
    if not targets:
        targets = [_target("UTC", "UTC", 0, "UTC")]
    return [_format_time_target(target) for target in targets]


def get_current_time_data(message: str = "", location: str = "") -> Dict[str, str]:
    return get_current_time_results(message=message, location=location)[0]


def current_time_reply(message: str = "", location: str = "") -> str:
    results = get_current_time_results(message=message, location=location)
    if len(results) == 1:
        data = results[0]
        return f"{data['label']} time is {data['time_12h']} {data['abbr']}, {data['date']}."

    labels = {item["label"] for item in results}
    has_us_representatives = {"New York, USA", "Los Angeles, USA"}.issubset(labels)
    prefix = "America has multiple time zones. Right now:" if has_us_representatives else "Right now:"
    lines = [
        f"- {item['label']}: {item['time_12h']} {item['abbr']}, {item['date']}."
        for item in results
    ]
    return "\n".join([prefix, *lines])
