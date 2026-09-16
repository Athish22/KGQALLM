"""Temporal expression parser: NL time phrases -> normalized ISO 8601
range(s). Deliberately a standalone, testable sub-component, not folded
into general entity extraction -- every example query in the reference KG
that filters at all does so on `tm:hasStartTime`/`tm:hasFinishTime`.

Covers: relative dates (today/yesterday/last N days), shift/period names
(this week/month), explicit timestamps (ISO date or datetime, with or
without an explicit "between X and Y"/"from X to Y"), and open-ended
ranges (since/before/after X). Needs its own eval set of NL phrase ->
expected range; this implementation is the seed to test against, not a
finished parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


@dataclass
class TemporalRange:
    start: str | None  # ISO 8601, or None for open start
    end: str | None  # ISO 8601, or None for open end
    matched_text: str


_ISO_DATETIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:?\d{2})?\b"
)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_LAST_N_UNIT = re.compile(r"\b(?:last|past)\s+(\d+)\s+(day|hour|week|month)s?\b", re.I)
_BETWEEN = re.compile(
    r"\bbetween\s+(?P<start>.+?)\s+and\s+(?P<end>.+)$", re.I
)
_FROM_TO = re.compile(r"\bfrom\s+(?P<start>.+?)\s+to\s+(?P<end>.+)$", re.I)
_SINCE = re.compile(r"\bsince\s+(?P<start>.+)$", re.I)
_BEFORE = re.compile(r"\bbefore\s+(?P<end>.+)$", re.I)
_AFTER = re.compile(r"\bafter\s+(?P<start>.+)$", re.I)
_ON = re.compile(r"\bon\s+(?P<day>.+)$", re.I)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _day_range(d: date) -> tuple[str, str]:
    start = datetime(d.year, d.month, d.day, 0, 0, 0)
    end = datetime(d.year, d.month, d.day, 23, 59, 59)
    return _iso(start), _iso(end)


def _parse_atom(text: str, now: datetime) -> datetime | None:
    """Parse a single time expression (not a range) to a datetime."""
    text = text.strip().rstrip(".,")
    m = _ISO_DATETIME.search(text)
    if m:
        raw = m.group(0).replace(" ", "T").rstrip("Z")
        raw = raw[:19]  # drop any offset for simplicity of stdlib parsing
        return datetime.fromisoformat(raw)
    m = _ISO_DATE.search(text)
    if m:
        return datetime.strptime(m.group(0), "%Y-%m-%d")

    low = text.lower()
    if "today" in low:
        return now
    if "yesterday" in low:
        return now - timedelta(days=1)
    if "tomorrow" in low:
        return now + timedelta(days=1)
    return None


class TemporalParser:
    def parse(self, text: str, now: datetime | None = None) -> TemporalRange | None:
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)

        for pattern in (_BETWEEN, _FROM_TO):
            m = pattern.search(text)
            if m:
                start_dt = _parse_atom(m.group("start"), now)
                end_dt = _parse_atom(m.group("end"), now)
                if start_dt or end_dt:
                    return TemporalRange(
                        start=_iso(start_dt) if start_dt else None,
                        end=_iso(end_dt) if end_dt else None,
                        matched_text=m.group(0),
                    )

        m = _LAST_N_UNIT.search(text)
        if m:
            n, unit = int(m.group(1)), m.group(2).lower()
            delta = {
                "day": timedelta(days=n),
                "hour": timedelta(hours=n),
                "week": timedelta(weeks=n),
                "month": timedelta(days=30 * n),
            }[unit]
            return TemporalRange(start=_iso(now - delta), end=_iso(now), matched_text=m.group(0))

        low = text.lower()
        if "this week" in low:
            start = now - timedelta(days=now.weekday())
            return TemporalRange(
                start=_iso(datetime(start.year, start.month, start.day)),
                end=_iso(now),
                matched_text="this week",
            )
        if "this month" in low:
            start = datetime(now.year, now.month, 1)
            return TemporalRange(start=_iso(start), end=_iso(now), matched_text="this month")

        m = _SINCE.search(text)
        if m:
            start_dt = _parse_atom(m.group("start"), now)
            if start_dt:
                return TemporalRange(start=_iso(start_dt), end=None, matched_text=m.group(0))

        m = _AFTER.search(text)
        if m:
            start_dt = _parse_atom(m.group("start"), now)
            if start_dt:
                return TemporalRange(start=_iso(start_dt), end=None, matched_text=m.group(0))

        m = _BEFORE.search(text)
        if m:
            end_dt = _parse_atom(m.group("end"), now)
            if end_dt:
                return TemporalRange(start=None, end=_iso(end_dt), matched_text=m.group(0))

        # Bare "today"/"yesterday"/an ISO date with no relational word ->
        # treat as a whole-day range rather than a single instant.
        for label in ("today", "yesterday", "tomorrow"):
            if label in low:
                anchor = _parse_atom(label, now)
                s, e = _day_range(anchor.date())
                return TemporalRange(start=s, end=e, matched_text=label)

        m = _ISO_DATE.search(text)
        if m and not _ISO_DATETIME.search(text):
            d = datetime.strptime(m.group(0), "%Y-%m-%d").date()
            s, e = _day_range(d)
            return TemporalRange(start=s, end=e, matched_text=m.group(0))

        m = _ISO_DATETIME.search(text)
        if m:
            dt = _parse_atom(m.group(0), now)
            return TemporalRange(start=_iso(dt), end=_iso(dt), matched_text=m.group(0))

        return None
