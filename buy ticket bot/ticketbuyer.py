import argparse
import difflib
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Iterable


DEFAULT_URL = "https://golive-asia.thaiticketmajor.com"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0"
)

try:
    from selenium import webdriver
    from selenium.common.exceptions import (
        ElementClickInterceptedException,
        NoSuchElementException,
        StaleElementReferenceException,
        WebDriverException,
    )
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
except ImportError:  # pragma: no cover - optional dependency
    webdriver = None
    ElementClickInterceptedException = Exception
    NoSuchElementException = Exception
    StaleElementReferenceException = Exception
    WebDriverException = Exception
    EdgeOptions = None
    By = None


STATUS_PATTERNS = {
    "available": [
        r"\bavailable\b",
        r"\bopen\b",
        r"\bcan\s+buy\b",
        r"\bready\b",
        r"\bbook\s+now\b",
    ],
    "limited": [
        r"\blimited\b",
        r"\balmost\s+gone\b",
        r"\bfew\s+left\b",
        r"\blow\s+availability\b",
    ],
    "blocked": [
        r"\blocked\b",
        r"\btemporarily\s+unavailable\b",
        r"\bnot\s+available\b",
        r"\bunavailable\b",
        r"\bbusy\b",
        r"\btry\s+again\b",
        r"\bseat\s+isn.?t\s+avail",
        r"\balready\s+taken\b",
        r"\bnot\s+eligible\b",
    ],
    "sold_out": [
        r"\bsold\s*out\b",
        r"\bfully\s+booked\b",
        r"\bout\s+of\s+stock\b",
    ],
}

KEYWORDS = (
    "seat",
    "zone",
    "section",
    "row",
    "ticket",
    "available",
    "unavailable",
    "locked",
    "open",
    "sold out",
    "book",
    "reserve",
    "price",
    "block",
)


@dataclass
class Finding:
    status: str
    line_number: int
    text: str


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int | None
    reason: str
    html: str
    accessible: bool
    access_note: str


@dataclass(frozen=True)
class SeatCandidate:
    section: str
    row: str
    seat_number: int
    source_text: str


@dataclass
class ActionCandidate:
    description: str
    locator: str
    text: str


@dataclass(frozen=True)
class SectionCandidate:
    section: str
    status: str
    source_text: str


def strip_html(html_text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html_text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(div|p|li|tr|td|th|section|article|h[1-6])>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def classify_line(line: str) -> str | None:
    lowered = line.lower()
    for status in ("blocked", "sold_out", "limited", "available"):
        patterns = STATUS_PATTERNS[status]
        if any(re.search(pattern, lowered) for pattern in patterns):
            return status
    return None


def iter_relevant_lines(text: str) -> Iterable[tuple[int, str]]:
    for idx, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            yield idx, line


def analyze_text(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in iter_relevant_lines(text):
        status = classify_line(line)
        if status:
            findings.append(Finding(status=status, line_number=line_number, text=line))
    return findings


def summarize(findings: list[Finding]) -> str:
    counts = Counter(f.status for f in findings)
    if not counts:
        return "No clearly classified seat or status lines were found."
    ordered = ["available", "limited", "blocked", "sold_out"]
    parts = [f"{status}={counts[status]}" for status in ordered if counts[status]]
    return "Summary: " + ", ".join(parts)


def normalize_label(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    cleaned = re.sub(r"\s+", " ", value).strip(" -:#")
    return cleaned or fallback


def extract_current_section_label_from_html(html_text: str) -> str | None:
    patterns = (
        re.compile(
            r"(?is)<span[^>]*class=\"[^\"]*\bzone-name\b[^\"]*\"[^>]*>.*?"
            r"<small[^>]*>\s*Section\s*</small>.*?<strong[^>]*>\s*([^<]+?)\s*</strong>"
        ),
        re.compile(
            r"(?is)<span[^>]*class=\"[^\"]*\blabel\b[^\"]*\"[^>]*>\s*Section\s*</span>.*?"
            r"<span[^>]*class=\"[^\"]*\bresult\b[^\"]*\"[^>]*>\s*([^<]+?)\s*</span>"
        ),
        re.compile(r"(?i)\bzoneDesc\b[^>]*\bvalue=\"([^\"]+)\""),
        re.compile(r"(?i)\bname=\"zone\"\b[^>]*\bvalue=\"([^\"]+)\""),
    )
    for pattern in patterns:
        match = pattern.search(html_text)
        if match:
            return normalize_label(match.group(1), "")
    return None


def line_looks_accessible(line: str) -> bool:
    lowered = line.lower()
    positive = (
        "available",
        "open",
        "free",
        "vacant",
        "can buy",
        "book now",
        "select",
    )
    negative = (
        "sold out",
        "unavailable",
        "blocked",
        "busy",
        "taken",
        "locked",
        "reserved",
        "not available",
        "not eligible",
    )
    if any(token in lowered for token in negative):
        return False
    return any(token in lowered for token in positive)


def extract_seat_candidates(text: str) -> list[SeatCandidate]:
    seat_patterns = [
        re.compile(
            r"(?i)\b(?:section|zone)\s*[:#-]?\s*(?P<section>[A-Za-z0-9_-]+).*?"
            r"\brow\s*[:#-]?\s*(?P<row>[A-Za-z0-9_-]+).*?"
            r"\bseat\s*[:#-]?\s*(?P<seat>\d{1,3})\b"
        ),
        re.compile(
            r"(?i)\brow\s*[:#-]?\s*(?P<row>[A-Za-z0-9_-]+).*?"
            r"\bseat\s*[:#-]?\s*(?P<seat>\d{1,3})\b"
        ),
        re.compile(
            r"(?i)\bseat\s*[:#-]?\s*(?P<seat>\d{1,3})\b.*?"
            r"\brow\s*[:#-]?\s*(?P<row>[A-Za-z0-9_-]+)\b"
        ),
    ]

    candidates: list[SeatCandidate] = []
    seen: set[tuple[str, str, int]] = set()
    current_section = "Unknown Section"
    current_row = "Unknown Row"

    for line in text.splitlines():
        lowered = line.lower()
        section_match = re.search(r"(?i)\b(?:section|zone)\s*[:#-]?\s*([A-Za-z0-9_-]+)\b", line)
        if section_match:
            current_section = normalize_label(section_match.group(1), current_section)
        row_match = re.search(r"(?i)\brow\s*[:#-]?\s*([A-Za-z0-9_-]+)\b", line)
        if row_match:
            current_row = normalize_label(row_match.group(1), current_row)

        if not line_looks_accessible(line):
            continue

        for pattern in seat_patterns:
            for match in pattern.finditer(line):
                section = normalize_label(match.groupdict().get("section"), current_section)
                row = normalize_label(match.groupdict().get("row"), current_row)
                seat_number = int(match.group("seat"))
                key = (section, row, seat_number)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SeatCandidate(
                        section=section,
                        row=row,
                        seat_number=seat_number,
                        source_text=line,
                    )
                )
    return candidates


def extract_seat_candidates_from_html(html_text: str, fallback_section: str = "Unknown Section") -> list[SeatCandidate]:
    candidates: list[SeatCandidate] = []
    seen: set[tuple[str, str, int]] = set()
    resolved_fallback_section = extract_current_section_label_from_html(html_text) or fallback_section

    seat_pattern = re.compile(
        r'(?is)<td[^>]*title="(?P<row>[A-Z]+)-(?P<seat>\d{1,3})"[^>]*>\s*'
        r'<div[^>]*class="[^"]*\bseatuncheck\b[^"]*"[^>]*data-seat="(?P<data>[^"]+)"'
    )
    for match in seat_pattern.finditer(html_text):
        row = normalize_label(match.group("row"), "Unknown Row")
        seat_number = int(match.group("seat"))
        data_seat = match.group("data")
        section_match = re.search(r"-P\*(?P<section>\d+)", data_seat)
        section = normalize_label(
            resolved_fallback_section,
            resolved_fallback_section,
        )
        key = (section, row, seat_number)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            SeatCandidate(
                section=section,
                row=row,
                seat_number=seat_number,
                source_text=f"{row}-{seat_number:02d}",
            )
        )
    return candidates


def extract_section_candidates(text: str) -> list[SectionCandidate]:
    candidates: list[SectionCandidate] = []
    seen: set[tuple[str, str]] = set()
    patterns = [
        re.compile(r"(?i)\bsection\s*[:#-]?\s*(?P<section>\d{3}|ST|TL|TS)\b.*?\b(?P<status>available|open|limited)\b"),
        re.compile(r"(?i)\b(?P<section>\d{3}|ST|TL|TS)\b\s+(?P<status>available|open|limited)\b"),
    ]

    for line in text.splitlines():
        for pattern in patterns:
            for match in pattern.finditer(line):
                section = normalize_label(match.group("section"), "Unknown Section")
                status = match.group("status").lower()
                key = (section, status)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    SectionCandidate(
                        section=section,
                        status=status,
                        source_text=line,
                    )
                )
    return candidates


def find_adjacent_pairs(candidates: list[SeatCandidate], group_size: int = 2) -> list[list[SeatCandidate]]:
    grouped: dict[tuple[str, str], dict[int, SeatCandidate]] = {}
    for candidate in candidates:
        key = (candidate.section, candidate.row)
        grouped.setdefault(key, {})[candidate.seat_number] = candidate

    pairs: list[list[SeatCandidate]] = []
    for (_, _), seats_by_number in sorted(grouped.items()):
        seat_numbers = sorted(seats_by_number)
        for seat_number in seat_numbers:
            window = [seat_number + offset for offset in range(group_size)]
            if all(number in seats_by_number for number in window):
                pairs.append([seats_by_number[number] for number in window])
    return pairs


def row_label_to_index(row: str) -> int:
    value = 0
    cleaned = re.sub(r"[^A-Z]", "", row.upper())
    if not cleaned:
        return 10_000
    for char in cleaned:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def find_nearby_pairs(candidates: list[SeatCandidate], group_size: int = 2) -> list[list[SeatCandidate]]:
    if group_size != 2:
        return []

    ordered = sorted(
        candidates,
        key=lambda seat: (seat.section, row_label_to_index(seat.row), seat.seat_number),
    )
    pairs: list[tuple[tuple[int, int, int, str, int], list[SeatCandidate]]] = []
    seen: set[tuple[str, str, int, str, int]] = set()

    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if first.section != second.section:
                continue
            row_gap = abs(row_label_to_index(first.row) - row_label_to_index(second.row))
            seat_gap = abs(first.seat_number - second.seat_number)
            if row_gap == 0 and seat_gap == 1:
                continue
            if row_gap <= 1 and seat_gap <= 1:
                key = (
                    first.section,
                    first.row,
                    first.seat_number,
                    second.row,
                    second.seat_number,
                )
                if key in seen:
                    continue
                seen.add(key)
                score = (row_gap + seat_gap, row_gap, seat_gap, first.row, first.seat_number)
                pairs.append((score, [first, second]))
                break

    pairs.sort(key=lambda item: item[0])
    return [pair for _, pair in pairs]


def print_pairs(candidates: list[SeatCandidate], group_size: int, show_empty_message: bool = True) -> None:
    pairs = find_adjacent_pairs(candidates, group_size=group_size)
    if not candidates:
        if show_empty_message:
            print(f"No accessible seat candidates were detected for groups of {group_size}.")
            print("Tip: use an event or seat-map URL, or save a snapshot from the actual booking page.")
        return

    print(
        f"Accessible seat candidates: {len(candidates)} "
        f"across {len({(seat.section, seat.row) for seat in candidates})} row groups."
    )

    if not pairs:
        nearby_pairs = find_nearby_pairs(candidates, group_size=group_size)
        if nearby_pairs:
            print(f"No side-by-side groups of {group_size} were detected. Closest nearby groups:")
            for pair in nearby_pairs[:10]:
                seats = " + ".join(f"{seat.row}-{seat.seat_number}" for seat in pair)
                print(f"- Section {pair[0].section}, Seats {seats}")
            return
        print(f"No side-by-side groups of {group_size} were detected in the parsed data.")
        return

    print(f"Side-by-side groups of {group_size}:")
    for pair in pairs[:20]:
        section = pair[0].section
        row = pair[0].row
        seats = ", ".join(str(seat.seat_number) for seat in pair)
        print(f"- Section {section}, Row {row}, Seats {seats}")


def print_sections(sections: list[SectionCandidate]) -> None:
    if not sections:
        print("No available sections were detected.")
        return
    labels = ", ".join(section.section for section in sections[:20])
    print(f"Available sections detected: {labels}")


def serialize_pairs(candidates: list[SeatCandidate], group_size: int) -> tuple[str, ...]:
    serialized: list[str] = []
    for pair in find_adjacent_pairs(candidates, group_size=group_size):
        section = pair[0].section
        row = pair[0].row
        seats = ",".join(str(seat.seat_number) for seat in pair)
        serialized.append(f"{section}|{row}|{seats}")
    return tuple(serialized)


def sanitize_for_xpath(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join(f"'{part}'" for part in parts) + ")"


def collect_browser_text(driver) -> str:
    script = """
const chunks = [];
const pushValue = (value) => {
  if (!value) return;
  const cleaned = String(value).replace(/\\s+/g, ' ').trim();
  if (!cleaned) return;
  chunks.push(cleaned);
};

pushValue(document.body ? document.body.innerText : '');

for (const el of document.querySelectorAll('*')) {
  pushValue(el.getAttribute('aria-label'));
  pushValue(el.getAttribute('title'));
  pushValue(el.getAttribute('alt'));
  const attrs = ['data-seat', 'data-row', 'data-section', 'data-zone', 'data-status', 'data-name'];
  const bits = [];
  for (const attr of attrs) {
    const value = el.getAttribute(attr);
    if (value) bits.push(`${attr}=${value}`);
  }
  if (bits.length) pushValue(bits.join(' '));
}

return chunks.join('\\n');
"""
    return driver.execute_script(script)


def collect_browser_snapshot(driver) -> str:
    chunks = [driver.page_source, collect_browser_text(driver)]
    frames = driver.find_elements("css selector", "iframe, frame")
    for index, frame in enumerate(frames, start=1):
        try:
            driver.switch_to.frame(frame)
            chunks.append(f"\nFRAME {index}\n")
            chunks.append(driver.page_source)
            chunks.append(collect_browser_text(driver))
        except WebDriverException:
            continue
        finally:
            driver.switch_to.default_content()
    return "\n".join(chunk for chunk in chunks if chunk)


def is_queue_page(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    markers = (
        "queue",
        "waiting room",
        "please wait",
        "line up",
        "queue-it",
        "lineup",
        "in line",
    )
    return any(marker in lowered for marker in markers)


def looks_like_payment_page(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    markers = (
        "payment",
        "credit card",
        "card number",
        "promptpay",
        "billing",
        "checkout",
        "pay now",
    )
    return any(marker in lowered for marker in markers)


def looks_like_confirmation_page(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    markers = (
        "confirm",
        "review order",
        "booking summary",
        "order summary",
        "reservation details",
    )
    return any(marker in lowered for marker in markers)


def looks_like_thanks_page(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    markers = (
        "thank you",
        "thanks for your purchase",
        "booking confirmed",
        "order confirmed",
        "payment successful",
        "transaction successful",
        "reservation confirmed",
        "your order has been placed",
    )
    return any(marker in lowered for marker in markers)


def looks_like_zone_retry_error(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    url_lower = url.lower()
    if "zones.php" in url_lower:
        return False

    strong_markers = (
        "please select date again from zone page",
        "select date again from zone page",
        "status not available",
    )
    if any(marker in lowered for marker in strong_markers):
        return True

    # Avoid false positives from the normal seat-map legend that also contains
    # the text "Not Available".
    if "sorry" in lowered and "select seat" not in lowered:
        return True

    return False


def looks_like_holding_status(text: str, url: str) -> bool:
    lowered = f"{text}\n{url}".lower()
    markers = (
        "holding status",
        "on hold",
        "being held",
        "currently held",
        "temporarily unavailable",
        "ticket might be unavailable",
        "tickets might be unavailable",
        "seat might be unavailable",
        "seats might be unavailable",
        "this section is unavailable",
    )
    return any(marker in lowered for marker in markers)


def looks_like_booking_url(url: str) -> bool:
    lowered = (url or "").lower()
    markers = (
        "zones.php",
        "fixed.php",
        "festival.php",
        "/booking/prww/",
        "golive-asia.thaiticketmajor.com/booking/",
    )
    return any(marker in lowered for marker in markers)


def get_element_text_blob(element) -> str:
    parts = [
        element.text,
        element.get_attribute("innerText"),
        element.get_attribute("aria-label"),
        element.get_attribute("title"),
        element.get_attribute("alt"),
        element.get_attribute("data-seat"),
        element.get_attribute("data-row"),
        element.get_attribute("data-section"),
        element.get_attribute("data-zone"),
        element.get_attribute("data-status"),
        element.get_attribute("data-name"),
        element.get_attribute("class"),
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def parse_seat_candidate_from_text(text: str) -> SeatCandidate | None:
    candidates = extract_seat_candidates(text)
    if not candidates:
        return None
    return candidates[0]


def collect_clickable_seat_candidates_fast(driver) -> list[dict[str, object]]:
    script = """
const seats = [];
const elements = document.querySelectorAll('#tableseats div.seatuncheck, #tableseats div[data-seat]');
for (let index = 0; index < elements.length; index += 1) {
  const el = elements[index];
  if (!el || !el.offsetParent) continue;
  const className = String(el.className || '');
  const dataSeat = String(el.getAttribute('data-seat') || '');
  const title = String(el.getAttribute('title') || '');
  const raw = `${dataSeat} ${title} ${className}`;

  let row = '';
  let seat = '';
  let section = '';

  let match = dataSeat.match(/([A-Z]+)-(\\d{1,3})/i) || title.match(/([A-Z]+)-(\\d{1,3})/i);
  if (match) {
    row = match[1];
    seat = match[2];
  }

  match = dataSeat.match(/-P\\*(\\d+)/i) || title.match(/(?:section|zone)\\s*[:#-]?\\s*([A-Za-z0-9_-]+)/i);
  if (match) {
    section = match[1];
  }

  if (!row || !seat) continue;
  if (!/seatuncheck/i.test(className) && !/available|open|free|vacant|select/i.test(raw)) continue;
  if (/sold out|unavailable|blocked|busy|taken|locked|reserved|not available|not eligible/i.test(raw)) continue;

  seats.push({
    dom_index: index,
    section: section || 'Unknown Section',
    row,
    seat_number: parseInt(seat, 10),
    source_text: raw.trim(),
  });
}
return seats;
"""
    try:
        result = driver.execute_script(script)
    except WebDriverException:
        return []
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


def collect_clickable_seat_elements(driver) -> list[tuple[SeatCandidate, object]]:
    clickable_candidates: list[tuple[SeatCandidate, object]] = []
    seen: set[tuple[str, str, int]] = set()
    selectors = (
        "#tableseats div.seatuncheck, "
        "#tableseats div[data-seat], "
        "button, a, [role='button'], [onclick], [aria-label], [data-seat], [data-row], [data-section], [data-zone]"
    )

    for element in driver.find_elements(By.CSS_SELECTOR, selectors):
        try:
            if not element.is_displayed():
                continue
            text_blob = get_element_text_blob(element)
        except (StaleElementReferenceException, WebDriverException):
            continue

        candidate = None
        try:
            data_seat = element.get_attribute("data-seat") or ""
            if data_seat:
                match = re.search(r"(?P<row>[A-Z]+)-(?P<seat>\d{1,3})", data_seat)
                section_match = re.search(r"-P\*(?P<section>\d+)", data_seat)
                if match:
                    candidate = SeatCandidate(
                        section=normalize_label(
                            section_match.group("section") if section_match else "",
                            "Unknown Section",
                        ),
                        row=normalize_label(match.group("row"), "Unknown Row"),
                        seat_number=int(match.group("seat")),
                        source_text=data_seat,
                    )
        except (StaleElementReferenceException, WebDriverException):
            candidate = None

        if candidate is None:
            candidate = parse_seat_candidate_from_text(text_blob)
        if candidate is None:
            continue
        if "seatuncheck" not in text_blob.lower() and not line_looks_accessible(text_blob):
            continue

        key = (candidate.section, candidate.row, candidate.seat_number)
        if key in seen:
            continue
        seen.add(key)
        clickable_candidates.append((candidate, element))

    return clickable_candidates


def collect_clickable_seat_elements_fast(driver) -> list[tuple[SeatCandidate, int]]:
    clickable_candidates: list[tuple[SeatCandidate, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in collect_clickable_seat_candidates_fast(driver):
        try:
            candidate = SeatCandidate(
                section=normalize_label(str(item.get("section") or ""), "Unknown Section"),
                row=normalize_label(str(item.get("row") or ""), "Unknown Row"),
                seat_number=int(item.get("seat_number")),
                source_text=str(item.get("source_text") or ""),
            )
            dom_index = int(item.get("dom_index"))
        except (TypeError, ValueError):
            continue
        key = (candidate.section, candidate.row, candidate.seat_number)
        if key in seen:
            continue
        seen.add(key)
        clickable_candidates.append((candidate, dom_index))
    return clickable_candidates


def find_adjacent_clickable_pair(
    clickable_candidates: list[tuple[SeatCandidate, object]], group_size: int
) -> list[tuple[SeatCandidate, object]] | None:
    return find_adjacent_clickable_pair_with_exclusions(clickable_candidates, group_size, set())


def serialize_pair_key(pair: list[tuple[SeatCandidate, object]]) -> tuple[tuple[str, str, int], ...]:
    return tuple((seat.section, seat.row, seat.seat_number) for seat, _ in pair)


def find_adjacent_clickable_pair_with_exclusions(
    clickable_candidates: list[tuple[SeatCandidate, object]],
    group_size: int,
    excluded_pairs: set[tuple[tuple[str, str, int], ...]],
) -> list[tuple[SeatCandidate, object]] | None:
    grouped: dict[tuple[str, str], dict[int, tuple[SeatCandidate, object]]] = {}
    for candidate, element in clickable_candidates:
        grouped.setdefault((candidate.section, candidate.row), {})[candidate.seat_number] = (
            candidate,
            element,
        )

    for (_, _), seats_by_number in sorted(grouped.items()):
        seat_numbers = sorted(seats_by_number)
        for seat_number in seat_numbers:
            window = [seat_number + offset for offset in range(group_size)]
            if all(number in seats_by_number for number in window):
                pair = [seats_by_number[number] for number in window]
                if serialize_pair_key(pair) not in excluded_pairs:
                    return pair
    if group_size == 2:
        ordered = sorted(
            clickable_candidates,
            key=lambda item: (item[0].section, row_label_to_index(item[0].row), item[0].seat_number),
        )
        best_pair: list[tuple[SeatCandidate, object]] | None = None
        best_score: tuple[int, int, str, int] | None = None
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                first_seat = first[0]
                second_seat = second[0]
                if first_seat.section != second_seat.section:
                    continue
                row_gap = abs(row_label_to_index(first_seat.row) - row_label_to_index(second_seat.row))
                seat_gap = abs(first_seat.seat_number - second_seat.seat_number)
                if row_gap <= 1 and seat_gap <= 1 and not (row_gap == 0 and seat_gap == 0):
                    score = (row_gap + seat_gap, row_gap, first_seat.row, first_seat.seat_number)
                    if best_score is None or score < best_score:
                        candidate_pair = [first, second]
                        if serialize_pair_key(candidate_pair) not in excluded_pairs:
                            best_score = score
                            best_pair = candidate_pair
                    break
        return best_pair
    return None


def collect_clickable_section_elements(driver) -> dict[str, object]:
    section_elements: dict[str, object] = {}

    for area in driver.find_elements(By.CSS_SELECTOR, "area[href*='#fixed.php#'], area[href*='#festival.php#']"):
        try:
            href = area.get_attribute("href") or ""
        except (StaleElementReferenceException, WebDriverException):
            continue

        match = re.search(r"#(?:fixed|festival)\.php#([A-Za-z0-9_-]+)$", href)
        if not match:
            continue
        section = normalize_label(match.group(1), "")
        if section and section not in section_elements:
            section_elements[section] = area

    selectors = "button, a, [role='button'], [onclick]"
    for element in driver.find_elements(By.CSS_SELECTOR, selectors):
        try:
            text_blob = get_element_text_blob(element)
        except (StaleElementReferenceException, WebDriverException):
            continue

        match = re.search(r"(?i)\b(section\s*)?(?P<section>\d{3}|ST|TL|TS)\b", text_blob)
        if not match:
            continue
        section = normalize_label(match.group("section"), "")
        if section and section not in section_elements:
            section_elements[section] = element

    return section_elements


def describe_clickable_sections(driver) -> list[str]:
    script = """
const labels = [];
const seen = new Set();
for (const area of document.querySelectorAll('area[href*="#fixed.php#"], area[href*="#festival.php#"]')) {
  const href = area.getAttribute('href') || '';
  const m = href.match(/#(?:fixed|festival)\\.php#([A-Za-z0-9_-]+)$/);
  if (!m) continue;
  const label = String(m[1]).trim();
  if (!label || seen.has(label)) continue;
  seen.add(label);
  labels.push(label);
}
return labels;
"""
    try:
        result = driver.execute_script(script)
    except WebDriverException:
        return []
    if not isinstance(result, list):
        return []
    return [normalize_label(str(item), "") for item in result if str(item).strip()]


def click_available_section(
    driver,
    available_sections: list[SectionCandidate],
    allowed_sections: set[str] | None = None,
) -> SectionCandidate | None:
    clickable_sections = collect_clickable_section_elements(driver)
    for section_candidate in available_sections:
        if allowed_sections is not None and section_candidate.section not in allowed_sections:
            continue
        element = clickable_sections.get(section_candidate.section)
        if element is None:
            continue
        if click_element(driver, element):
            return section_candidate
    return None


def click_any_available_section_fast(
    driver,
    excluded_sections: set[str] | None = None,
    allowed_sections: set[str] | None = None,
) -> str | None:
    excluded = {normalize_label(section, "") for section in (excluded_sections or set()) if section}
    allowed = {normalize_label(section, "") for section in (allowed_sections or set()) if section}
    script = """
const areas = Array.from(document.querySelectorAll('area[href*="#fixed.php#"], area[href*="#festival.php#"]'));
const excluded = new Set(arguments[0] || []);
const allowed = new Set(arguments[1] || []);
for (const area of areas) {
  const href = area.getAttribute('href') || '';
  const m = href.match(/#(?:fixed|festival)\\.php#([A-Za-z0-9_-]+)$/);
  if (!m) continue;
  const section = String(m[1]).trim();
  if (allowed.size && !allowed.has(section)) continue;
  if (excluded.has(section)) continue;
  area.click();
  return section;
}
return null;
"""
    try:
        result = driver.execute_script(script, list(excluded), list(allowed))
        return str(result) if result else None
    except WebDriverException:
        return None


def click_preferred_section_fast(driver, preferred_sections: list[str]) -> str | None:
    normalized_sections = [normalize_label(section, "") for section in preferred_sections if section]
    script = """
const preferred = arguments[0] || [];
const areas = Array.from(document.querySelectorAll('area[href*="#fixed.php#"], area[href*="#festival.php#"]'));
for (const target of preferred) {
  for (const area of areas) {
    const href = area.getAttribute('href') || '';
    const m = href.match(/#(?:fixed|festival)\\.php#([A-Za-z0-9_-]+)$/);
    if (!m) continue;
    if (String(m[1]).trim() !== String(target).trim()) continue;
    area.click();
    return target;
  }
}
return null;
"""
    try:
        result = driver.execute_script(script, normalized_sections)
        return str(result) if result else None
    except WebDriverException:
        return None


def click_preferred_section(driver, preferred_sections: list[str]) -> str | None:
    fast_result = click_preferred_section_fast(driver, preferred_sections)
    if fast_result is not None:
        return fast_result
    clickable_sections = collect_clickable_section_elements(driver)
    for section in preferred_sections:
        normalized_section = normalize_label(section, "")
        element = clickable_sections.get(normalized_section)
        if element is None:
            continue
        if click_element(driver, element):
            return normalized_section
    return None


def go_back_to_zone_page(driver) -> bool:
    try:
        selectors = (
            "a[href*='zones.php?query']",
            "a.btn-action",
            "a[href='javascript:void(0);']",
            "button",
        )
        for selector in selectors:
            for element in driver.find_elements(By.CSS_SELECTOR, selector):
                try:
                    text_blob = get_element_text_blob(element).lower()
                    href = (element.get_attribute("href") or "").lower()
                except (StaleElementReferenceException, WebDriverException):
                    continue
                if "select another section" in text_blob or "zones.php?query" in href:
                    if click_element(driver, element):
                        return True
    except WebDriverException:
        pass

    try:
        driver.back()
        return True
    except WebDriverException:
        return False


def rotate_preferred_sections(preferred_sections: list[str], start_index: int) -> list[str]:
    if not preferred_sections:
        return []
    index = start_index % len(preferred_sections)
    return preferred_sections[index:] + preferred_sections[:index]


def choose_next_section(
    driver,
    preferred_sections: list[str],
    preferred_index: int,
    exhausted_preferred_sections: set[str],
    only_preferred_sections: bool = False,
    allowed_sections: set[str] | None = None,
) -> tuple[str | None, int, str]:
    allowed_sections = (
        {normalize_label(section, "") for section in allowed_sections if section}
        if allowed_sections
        else None
    )
    rotated = rotate_preferred_sections(preferred_sections, preferred_index)
    for offset, section in enumerate(rotated):
        normalized_section = normalize_label(section, "")
        if allowed_sections is not None and normalized_section not in allowed_sections:
            continue
        if not normalized_section or normalized_section in exhausted_preferred_sections:
            continue
        chosen = click_preferred_section(driver, [normalized_section])
        if chosen is not None:
            next_index = (
                (preferred_sections.index(chosen) + 1) % len(preferred_sections)
                if chosen in preferred_sections
                else preferred_index
            )
            return chosen, next_index, "preferred"

    if only_preferred_sections:
        return None, preferred_index, "none"

    excluded_sections = set(exhausted_preferred_sections)
    fallback = click_any_available_section_fast(
        driver,
        excluded_sections=excluded_sections,
        allowed_sections=allowed_sections,
    )
    if fallback is None and allowed_sections and exhausted_preferred_sections:
        exhausted_preferred_sections.clear()
        fallback = click_any_available_section_fast(
            driver,
            excluded_sections=set(),
            allowed_sections=allowed_sections,
        )
    if fallback is not None:
        return fallback, preferred_index, "fallback"

    return None, preferred_index, "none"


def wait_for_zone_page(driver, timeout_seconds: float = 1.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if "zones.php" in driver.current_url.lower():
                return True
        except WebDriverException:
            pass
        time.sleep(0.01)
    return False


def wait_for_page_mode(driver, target_mode: str, timeout_seconds: float = 1.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if get_page_mode(driver.current_url, driver.page_source) == target_mode:
                return True
        except WebDriverException:
            pass
        time.sleep(0.01)
    return False


def wait_for_url_change(driver, previous_url: str, timeout_seconds: float = 0.5) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            if driver.current_url != previous_url:
                return True
        except WebDriverException:
            pass
        time.sleep(0.01)
    return False


def wait_for_section_transition_result(
    driver,
    previous_url: str,
    timeout_seconds: float = 0.35,
) -> str:
    deadline = time.time() + timeout_seconds
    saw_url_change = False
    while time.time() < deadline:
        try:
            current_url = driver.current_url
            page_source = driver.page_source
        except WebDriverException:
            time.sleep(0.01)
            continue

        if current_url != previous_url:
            saw_url_change = True

        page_mode = get_page_mode(current_url, page_source)
        if page_mode == "seat":
            return "seat"
        if page_mode == "zone" and saw_url_change:
            return "zone"

        normalized = strip_html(page_source)
        if looks_like_holding_status(normalized, current_url):
            return "holding"
        if looks_like_zone_retry_error(normalized, current_url):
            return "retry"

        time.sleep(0.01)
    return "unknown"


def wait_for_seat_selection_result(
    driver,
    expected_count: int,
    timeout_seconds: float = 0.25,
) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            current_url = driver.current_url
            page_source = driver.page_source
        except WebDriverException:
            time.sleep(0.01)
            continue

        selected_count = get_selected_seat_count(driver)
        if selected_count >= expected_count:
            return "selected"

        normalized = strip_html(page_source)
        if looks_like_holding_status(normalized, current_url):
            return "holding"
        if looks_like_zone_retry_error(normalized, current_url):
            return "retry"

        time.sleep(0.01)
    return "unknown"


def maybe_switch_to_booking_tab(driver) -> bool:
    try:
        current_url = driver.current_url
        if looks_like_booking_url(current_url):
            return True
        targets = driver.execute_cdp_cmd("Target.getTargets", {})
    except WebDriverException:
        return False

    target_infos = targets.get("targetInfos", []) if isinstance(targets, dict) else []
    booking_target = next(
        (
            info
            for info in target_infos
            if info.get("type") == "page" and looks_like_booking_url(str(info.get("url") or ""))
        ),
        None,
    )
    if booking_target is None:
        return False

    target_id = str(booking_target.get("targetId") or "")
    if not target_id:
        return False

    try:
        driver.switch_to.window(target_id)
        return True
    except WebDriverException:
        return False


def wait_for_manual_start_page(driver, require_confirm: bool) -> str:
    print("Use the Edge window to pass any queue/login flow.")
    print("Waiting for the current tab to become the zone page or seat page...")
    last_mode = ""
    last_switch_attempt = 0.0
    while True:
        try:
            now = time.perf_counter()
            current_url = driver.current_url
            if looks_like_booking_url(current_url) or now - last_switch_attempt >= 0.5:
                maybe_switch_to_booking_tab(driver)
                last_switch_attempt = now
            current_url = driver.current_url
            page_mode = get_page_mode(current_url, driver.page_source)
        except WebDriverException:
            time.sleep(0.02)
            continue

        if page_mode != last_mode:
            if page_mode == "zone":
                print("Zone page detected.")
            elif page_mode == "seat":
                print("Seat page detected.")
            last_mode = page_mode

        if page_mode in {"zone", "seat"}:
            if require_confirm:
                print("Press Enter to start scanning and auto actions.")
                input()
            return page_mode

        time.sleep(0.02)


def click_element(driver, element) -> bool:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        element.click()
        return True
    except (ElementClickInterceptedException, WebDriverException):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except WebDriverException:
            return False


def click_adjacent_pair(driver, group_size: int) -> list[SeatCandidate] | None:
    return click_adjacent_pair_with_exclusions(driver, group_size, set())


def click_seat_elements_by_dom_index(driver, dom_indices: list[int]) -> bool:
    if not dom_indices:
        return False
    script = """
const indices = arguments[0];
const elements = document.querySelectorAll('#tableseats div.seatuncheck, #tableseats div[data-seat]');
for (const index of indices) {
  const el = elements[index];
  if (!el) return false;
  el.scrollIntoView({block: 'center'});
  el.click();
}
return true;
"""
    try:
        return bool(driver.execute_script(script, dom_indices))
    except WebDriverException:
        return False


def wait_and_click_adjacent_pair_with_exclusions(
    driver,
    group_size: int,
    excluded_pairs: set[tuple[tuple[str, str, int], ...]],
    timeout_seconds: float = 0.75,
) -> list[SeatCandidate] | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        selected = click_adjacent_pair_with_exclusions(driver, group_size, excluded_pairs)
        if selected:
            return selected
        time.sleep(0.01)
    return None


def click_adjacent_pair_with_exclusions(
    driver,
    group_size: int,
    excluded_pairs: set[tuple[tuple[str, str, int], ...]],
) -> list[SeatCandidate] | None:
    clickable_candidates = collect_clickable_seat_elements_fast(driver)
    if clickable_candidates:
        pair = find_adjacent_clickable_pair_with_exclusions(
            clickable_candidates,
            group_size,
            excluded_pairs,
        )
        if pair is not None:
            selected = [candidate for candidate, _ in pair]
            dom_indices = [dom_index for _, dom_index in pair]
            if click_seat_elements_by_dom_index(driver, dom_indices):
                return selected

    clickable_candidates = collect_clickable_seat_elements(driver)
    pair = find_adjacent_clickable_pair_with_exclusions(clickable_candidates, group_size, excluded_pairs)
    if pair is None:
        return None

    selected: list[SeatCandidate] = []
    for candidate, element in pair:
        if not click_element(driver, element):
            return None
        selected.append(candidate)
        time.sleep(0.01)
    return selected


def find_action_buttons(driver, labels: tuple[str, ...]) -> list[tuple[ActionCandidate, object]]:
    matches: list[tuple[ActionCandidate, object]] = []
    seen: set[str] = set()
    lowered_labels = tuple(label.lower() for label in labels)

    selectors = "button, input[type='button'], input[type='submit'], a, [role='button']"
    for element in driver.find_elements(By.CSS_SELECTOR, selectors):
        try:
            if not element.is_displayed() or not element.is_enabled():
                continue
            text_blob = get_element_text_blob(element)
        except (StaleElementReferenceException, WebDriverException):
            continue

        lowered = text_blob.lower()
        if not any(label in lowered for label in lowered_labels):
            continue

        try:
            locator = element.get_attribute("outerHTML")[:120]
        except WebDriverException:
            locator = "<unavailable>"
        if locator in seen:
            continue
        seen.add(locator)
        matches.append(
            (
                ActionCandidate(
                    description="matching action button",
                    locator=locator,
                    text=text_blob[:120],
                ),
                element,
            )
        )
    return matches


def click_best_action_button(driver, labels: tuple[str, ...]) -> ActionCandidate | None:
    for candidate, element in find_action_buttons(driver, labels):
        if click_element(driver, element):
            return candidate
    return None


def get_selected_seat_count(driver) -> int:
    script = """
const selected = document.querySelectorAll('#tableseats .seatcheck, #tableseats .seatchecked, #tableseats .selected');
if (selected.length) return selected.length;
const seatList = document.querySelector('#seatlist');
if (seatList && seatList.value) {
  return seatList.value.split(',').map(s => s.trim()).filter(Boolean).length;
}
const totalSelected = document.querySelector('#total_selected');
if (totalSelected) {
  const match = String(totalSelected.textContent || '').match(/\\d+/);
  if (match) return parseInt(match[0], 10);
}
return 0;
"""
    try:
        return int(driver.execute_script(script) or 0)
    except WebDriverException:
        return 0


def get_selected_seat_labels(driver) -> list[str]:
    script = """
const selected = Array.from(document.querySelectorAll('#tableseats .seatcheck, #tableseats .seatchecked, #tableseats .selected'));
const labels = [];
for (const el of selected) {
  const parts = [];
  const dataSeat = String(el.getAttribute('data-seat') || '').trim();
  const title = String(el.getAttribute('title') || '').trim();
  const aria = String(el.getAttribute('aria-label') || '').trim();
  const text = String(el.textContent || '').replace(/\\s+/g, ' ').trim();
  if (dataSeat) parts.push(dataSeat);
  if (title) parts.push(title);
  if (aria) parts.push(aria);
  if (!parts.length && text) parts.push(text);
  const label = parts.join(' | ').trim();
  if (label) labels.push(label);
}
return labels;
"""
    try:
        result = driver.execute_script(script)
    except WebDriverException:
        return []
    if not isinstance(result, list):
        return []
    return [str(item).strip() for item in result if str(item).strip()]


def clear_selected_seats(driver, timeout_seconds: float = 0.4) -> bool:
    deadline = time.time() + timeout_seconds
    cleared_any = False
    previous_count: int | None = None

    script = """
const selected = Array.from(document.querySelectorAll('#tableseats .seatcheck, #tableseats .seatchecked, #tableseats .selected'));
if (!selected.length) return 0;
for (const el of selected) {
  el.scrollIntoView({block: 'center'});
  el.click();
}
return selected.length;
"""
    while time.time() < deadline:
        selected_count = get_selected_seat_count(driver)
        if selected_count <= 0:
            return cleared_any
        if previous_count is not None and selected_count >= previous_count:
            return cleared_any
        previous_count = selected_count
        try:
            clicked_count = int(driver.execute_script(script) or 0)
        except WebDriverException:
            return cleared_any
        if clicked_count > 0:
            cleared_any = True
        time.sleep(0.03)
    return cleared_any


def click_book_now(driver) -> bool:
    script = """
const button = document.querySelector('#booknow') || document.querySelector('#bookmnow');
if (!button) return false;
button.scrollIntoView({block: 'center'});
button.click();
return true;
"""
    try:
        return bool(driver.execute_script(script))
    except WebDriverException:
        return False


def get_page_mode(current_url: str, page_source: str) -> str:
    lowered_url = current_url.lower()
    lowered_source = page_source.lower()
    if 'id="tableseats"' in lowered_source or "seatuncheck" in lowered_source or "data-seat=" in lowered_source:
        return "seat"
    if "zones.php" in lowered_url or "select-zone" in lowered_source:
        return "zone"
    if "fixed.php" in lowered_url or "festival.php" in lowered_url:
        return "seat"
    return "other"


def open_section_availability_popup(driver) -> bool:
    popup_button = click_best_action_button(driver, ("seats available",))
    if popup_button is None:
        return False
    time.sleep(1.0)
    return True


def extract_available_sections_from_live_dom(driver) -> list[SectionCandidate]:
    sources: list[str] = [driver.page_source]
    try:
        body_text = collect_browser_text(driver)
        if body_text:
            sources.append(body_text)
    except WebDriverException:
        pass
    return extract_section_candidates("\n".join(sources))


def fetch_zone_availability(driver) -> list[SectionCandidate]:
    script = """
const done = arguments[0];
const roundSelect = document.querySelector('#rdId');
const roundInput = document.querySelector('input[name="rdId"]');
const round = (roundSelect && roundSelect.value) || (roundInput && roundInput.value) || '';
if (!round) {
  done({ok: false, reason: 'no-round'});
  return;
}
fetch(`zonesavail.php?round=${encodeURIComponent(round)}`, {
  credentials: 'same-origin',
  cache: 'no-store'
})
  .then((res) => res.text())
  .then((html) => done({ok: true, html}))
  .catch((err) => done({ok: false, reason: String(err)}));
"""
    try:
        result = driver.execute_async_script(script)
    except WebDriverException:
        return []

    if not isinstance(result, dict) or not result.get("ok"):
        return []

    html = str(result.get("html") or "")
    text = strip_html(html)
    return extract_section_candidates(text)


def select_best_section_candidate(
    preferred_sections: list[str],
    preferred_index: int,
    available_sections: list[SectionCandidate],
) -> str | None:
    available_lookup = {section.section for section in available_sections}
    if preferred_sections:
        for section in rotate_preferred_sections(preferred_sections, preferred_index):
            normalized = normalize_label(section, "")
            if normalized in available_lookup:
                return normalized
        # If the live availability panel is stale or incomplete, still allow
        # probing preferred sections in order before giving up.
        for section in rotate_preferred_sections(preferred_sections, preferred_index):
            normalized = normalize_label(section, "")
            if normalized:
                return normalized
    if available_sections:
        return available_sections[0].section
    return None


def read_snapshot(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def fetch_live_page(url: str, timeout: float) -> FetchResult:
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            html = body.decode(charset, errors="replace")
            final_url = response.geturl()
            status_code = getattr(response, "status", None)
            reason = getattr(response, "reason", "OK")
    except HTTPError as exc:
        body = exc.read()
        charset = exc.headers.get_content_charset() or "utf-8"
        html = body.decode(charset, errors="replace")
        return FetchResult(
            url=url,
            final_url=exc.geturl(),
            status_code=exc.code,
            reason=exc.reason,
            html=html,
            accessible=False,
            access_note=f"HTTP error {exc.code} ({exc.reason})",
        )
    except URLError as exc:
        return FetchResult(
            url=url,
            final_url=url,
            status_code=None,
            reason=str(exc.reason),
            html="",
            accessible=False,
            access_note=f"Network error: {exc.reason}",
        )

    lowered = html.lower()
    challenge_patterns = (
        "access denied",
        "forbidden",
        "captcha",
        "cloudflare",
        "attention required",
        "temporarily unavailable",
        "service unavailable",
        "bot detection",
    )
    final_url_lower = final_url.lower()
    blocked_by_page = any(pattern in lowered for pattern in challenge_patterns)
    error_redirect = "error.php" in final_url_lower or "errcode=" in final_url_lower
    meta_refresh_home = bool(
        re.search(r'(?is)<meta[^>]+http-equiv=["\']refresh["\'][^>]+content=["\'][^"\']*url=/', html)
    )
    accessible = (
        bool(status_code and 200 <= status_code < 400)
        and not blocked_by_page
        and not error_redirect
        and not meta_refresh_home
    )
    if error_redirect:
        access_note = f"Booking URL redirected to an error page: {final_url}"
    elif meta_refresh_home:
        access_note = "Page immediately redirects to the site root, so the booking state is not usable."
    elif blocked_by_page:
        access_note = "Page responded, but content suggests access is blocked or challenged."
    elif accessible:
        access_note = "Page responded with a successful status and no obvious block page markers."
    else:
        access_note = "Page did not return a clearly accessible response."

    return FetchResult(
        url=url,
        final_url=final_url,
        status_code=status_code,
        reason=str(reason),
        html=html,
        accessible=accessible,
        access_note=access_note,
    )


def command_analyze(path: Path, show_all: bool, pair_size: int) -> int:
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    raw_text = read_snapshot(path)
    text = strip_html(raw_text)
    findings = analyze_text(text)
    sections = extract_section_candidates(text)
    seat_candidates = extract_seat_candidates(text) or extract_seat_candidates_from_html(raw_text)

    print(f"Snapshot: {path}")
    print(summarize(findings))
    print_sections(sections)
    print_pairs(seat_candidates, pair_size)

    if findings:
        print("\nMatches:")
        for finding in findings:
            print(f"[{finding.status}] line {finding.line_number}: {finding.text}")
    elif show_all:
        print("\nRelevant lines:")
        for line_number, line in iter_relevant_lines(text):
            print(f"line {line_number}: {line}")

    return 0


def command_compare(old_path: Path, new_path: Path) -> int:
    if not old_path.exists():
        print(f"File not found: {old_path}")
        return 1
    if not new_path.exists():
        print(f"File not found: {new_path}")
        return 1

    old_text = strip_html(read_snapshot(old_path)).splitlines()
    new_text = strip_html(read_snapshot(new_path)).splitlines()

    diff = list(
        difflib.unified_diff(
            old_text,
            new_text,
            fromfile=str(old_path),
            tofile=str(new_path),
            n=1,
            lineterm="",
        )
    )

    relevant = [
        line
        for line in diff
        if line.startswith(("---", "+++", "@@"))
        or any(keyword in line.lower() for keyword in KEYWORDS)
    ]

    print(f"Comparing {old_path} -> {new_path}")
    if relevant:
        print("\nRelevant changes:")
        for line in relevant:
            print(line)
    else:
        print("No seat-related text changes detected in the saved snapshots.")

    old_findings = analyze_text("\n".join(old_text))
    new_findings = analyze_text("\n".join(new_text))
    print()
    print("Before:", summarize(old_findings))
    print("After: ", summarize(new_findings))
    return 0


def command_live(
    url: str,
    timeout: float,
    save_to: Path | None,
    show_all: bool,
    pair_size: int,
) -> int:
    result = fetch_live_page(url, timeout)
    print(f"Requested URL: {result.url}")
    print(f"Final URL: {result.final_url}")
    if result.final_url.rstrip("/") != result.url.rstrip("/"):
        print("Redirected: yes")
    else:
        print("Redirected: no")
    if result.status_code is None:
        print(f"HTTP status: unavailable ({result.reason})")
    else:
        print(f"HTTP status: {result.status_code} {result.reason}")
    print(f"Accessible: {'yes' if result.accessible else 'no'}")
    print(f"Assessment: {result.access_note}")

    if save_to is not None and result.html:
        save_to.write_text(result.html, encoding="utf-8")
        print(f"Saved snapshot: {save_to}")

    if not result.html:
        print("\nNo HTML body was returned.")
        return 1

    text = strip_html(result.html)
    findings = analyze_text(text)
    candidates = extract_seat_candidates(text) or extract_seat_candidates_from_html(result.html)
    sections = extract_section_candidates(text)
    print()
    print(summarize(findings))
    print_sections(sections)
    print_pairs(candidates, pair_size)

    if findings:
        print("\nMatches:")
        for finding in findings:
            print(f"[{finding.status}] line {finding.line_number}: {finding.text}")
    elif show_all:
        print("\nRelevant lines:")
        for line_number, line in iter_relevant_lines(text):
            print(f"line {line_number}: {line}")

    return 0 if result.accessible else 1


def command_browser_watch(
    url: str | None,
    pair_size: int,
    poll_seconds: float,
    max_cycles: int,
    save_to: Path | None,
    headless: bool,
    user_data_dir: Path | None,
    lean_browser: bool,
    confirm_start: bool,
    auto_select: bool,
    auto_advance: bool,
    continue_to_payment: bool,
    debugger_address: str | None,
    keep_browser_open: bool,
    preferred_sections: list[str],
    only_preferred_sections: bool,
    allowed_sections: list[str],
    manual_section_fallback: bool,
) -> int:
    if webdriver is None or EdgeOptions is None:
        print("Selenium is not installed. Run: python -m pip install selenium")
        return 1

    options = EdgeOptions()
    effective_keep_browser_open = keep_browser_open or debugger_address is not None
    options.add_argument(f"--user-agent={DEFAULT_USER_AGENT}")
    options.add_argument("--start-maximized")
    if headless and debugger_address is None:
        options.add_argument("--headless=new")
    if user_data_dir is not None:
        options.add_argument(f"--user-data-dir={user_data_dir.resolve()}")
    if lean_browser:
        # Keep the Edge session light for ticketing when Selenium launches it.
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--disable-component-update")
        options.add_argument("--disable-default-apps")
        options.add_argument("--disable-features=MediaRouter,OptimizationHints,AutofillServerCommunication")
        options.add_argument("--disable-sync")
        options.add_argument("--metrics-recording-only")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
    if debugger_address is not None:
        options.debugger_address = debugger_address

    try:
        if debugger_address is not None:
            print(f"Attaching to existing Edge at {debugger_address}...")
            if lean_browser:
                print("Lean browser mode only affects Edge sessions launched by the script, not an already running Edge attached with --debugger-address.")
            driver = webdriver.Edge(options=options)
            print("Attached to existing Edge session.")
            if url:
                print(f"Navigating attached browser to: {url}")
                driver.get(url)
            else:
                print("Using the current tab in your existing Edge session.")
        else:
            print("Launching Edge browser session...")
            driver = webdriver.Edge(options=options)
            if url:
                driver.get(url)
    except WebDriverException as exc:
        print(f"Could not start or attach to Edge: {exc}")
        return 1

    print("Browser opened.")
    wait_for_manual_start_page(driver, require_confirm=confirm_start)

    if auto_select and not auto_advance:
        print("Auto-select is on. It will click seats only.")
        print("Add --auto-advance if you want it to click Book Now too.")
    if preferred_sections and only_preferred_sections:
        print(f"Only preferred sections will be used: {', '.join(preferred_sections)}")
    if preferred_sections and manual_section_fallback:
        print(
            "Manual section fallback is on. The bot will try preferred sections first, "
            "then wait for you to choose sections manually."
        )
    normalized_allowed_sections = {
        normalize_label(section, "") for section in allowed_sections if section
    }
    if normalized_allowed_sections:
        print(f"Fallback sections are limited to: {', '.join(sorted(normalized_allowed_sections))}")
    section_auto_click_is_preferred_only = only_preferred_sections or manual_section_fallback
    automatic_fallback_sections_enabled = not section_auto_click_is_preferred_only
    can_exhaust_active_section = not only_preferred_sections

    last_signature: tuple[str, ...] | None = None
    last_url = ""
    exit_code = 1
    preferred_index = 0
    exhausted_preferred_sections: set[str] = set()
    active_section_name: str | None = None
    attempted_pairs_by_page: dict[str, set[tuple[tuple[str, str, int], ...]]] = {}
    last_zone_section_signature: tuple[str, ...] | None = None

    try:
        for cycle in range(1, max_cycles + 1):
            current_url = driver.current_url
            page_source = driver.page_source
            page_mode = get_page_mode(current_url, page_source)
            selected_seat_labels: list[str] = []
            clickable_zone_sections: list[str] = []
            if page_mode == "seat":
                normalized = strip_html(page_source)
                candidates = extract_seat_candidates_from_html(page_source)
                sections = []
                selected_seat_labels = get_selected_seat_labels(driver)
            elif page_mode == "zone":
                normalized = ""
                candidates = []
                sections = []
            else:
                combined_text = collect_browser_snapshot(driver)
                normalized = strip_html(combined_text)
                candidates = extract_seat_candidates(normalized) or extract_seat_candidates_from_html(
                    page_source
                )
                sections = extract_section_candidates(normalized)
            signature = serialize_pairs(candidates, pair_size)
            zone_retry_error = looks_like_zone_retry_error(normalized, current_url)

            should_print = signature != last_signature or current_url != last_url or cycle == 1
            if should_print:
                print()
                print(f"[scan {cycle}] URL: {current_url}")
                if normalized and is_queue_page(normalized, current_url):
                    print("Queue detected. Waiting for the booking page to become available.")
                if page_mode == "seat":
                    print_pairs(candidates, pair_size, show_empty_message=False)
                    if selected_seat_labels:
                        print("Currently selected seats:")
                        for label in selected_seat_labels[:10]:
                            print(f"- {label}")
                elif page_mode == "zone":
                    clickable_zone_sections = describe_clickable_sections(driver)
                    print("Zone page ready.")
                    if clickable_zone_sections:
                        labels = ", ".join(clickable_zone_sections[:20])
                        print(f"Clickable sections detected: {labels}")
                    else:
                        print("No clickable section IDs detected on the zone map yet.")
                    normalized_preferred_sections = [
                        normalize_label(section, "") for section in preferred_sections if section
                    ]
                    matching_preferred_sections = [
                        section
                        for section in normalized_preferred_sections
                        if section in clickable_zone_sections
                    ]
                    if normalized_preferred_sections and not matching_preferred_sections:
                        joined = ", ".join(normalized_preferred_sections)
                        print(
                            f"Preferred sections did not match the current zone map: {joined}"
                        )
                else:
                    print_sections(sections)
                    print_pairs(candidates, pair_size, show_empty_message=False)
                if save_to is not None:
                    save_to.write_text(page_source, encoding="utf-8")
                    print(f"Saved snapshot: {save_to}")
                if zone_retry_error:
                    print("Section page returned a retry error. Moving back to the zone page.")

            if signature:
                exit_code = 0

            if auto_select and not (normalized and is_queue_page(normalized, current_url)):
                if page_mode == "seat":
                    if len(selected_seat_labels) >= pair_size:
                        print("Seats are already selected on the seat page. Not auto-clicking again.")
                        last_signature = signature
                        last_url = current_url
                        if cycle < max_cycles:
                            time.sleep(poll_seconds)
                        continue
                    if selected_seat_labels and clear_selected_seats(driver):
                        print("Cleared leftover partial seat selection before retrying.")
                        continue
                    attempted_pairs = attempted_pairs_by_page.setdefault(current_url, set())
                    selected = wait_and_click_adjacent_pair_with_exclusions(
                        driver,
                        pair_size,
                        attempted_pairs,
                        timeout_seconds=0.75,
                    )
                    if selected:
                        seats = ", ".join(
                            f"{seat.section}/{seat.row}/{seat.seat_number}" for seat in selected
                        )
                        attempted_pairs.add(
                            tuple((seat.section, seat.row, seat.seat_number) for seat in selected)
                        )
                        selection_result = wait_for_seat_selection_result(
                            driver,
                            expected_count=pair_size,
                            timeout_seconds=0.25,
                        )
                        if selection_result == "selected":
                            print(f"Selected seats: {seats}")
                        elif selection_result == "holding":
                            clear_selected_seats(driver)
                            print("Seat click triggered a holding/unavailable message. Trying another pair.")
                            continue
                        elif selection_result == "retry":
                            clear_selected_seats(driver)
                            print("Seat click was rejected as unavailable. Trying another pair.")
                            continue
                        else:
                            clear_selected_seats(driver)
                            print("Seat click did not stick. Trying another pair.")
                            continue
                        if auto_advance:
                            if click_book_now(driver) or (time.sleep(0.03) is None and click_book_now(driver)):
                                print("Clicked Book Now.")
                                time.sleep(0.05)
                                current_url = driver.current_url
                                current_source = driver.page_source
                                current_snapshot = strip_html(current_source)
                                if looks_like_payment_page(current_snapshot, current_url) and not continue_to_payment:
                                    print("Reached payment step. Stopping before payment.")
                                    return 0
                            else:
                                print("Seats selected, but Book Now was not clicked.")
                                continue
                            if not continue_to_payment:
                                return 0
                elif page_mode == "zone":
                    chosen_section_name, preferred_index, selection_kind = choose_next_section(
                        driver,
                        preferred_sections,
                        preferred_index,
                        exhausted_preferred_sections,
                        section_auto_click_is_preferred_only,
                        normalized_allowed_sections,
                    )
                    if chosen_section_name is not None:
                        active_section_name = chosen_section_name
                        if selection_kind == "preferred":
                            print(f"Selected preferred section: {chosen_section_name}")
                        else:
                            print(f"Selected fallback section: {chosen_section_name}")
                        transition_result = wait_for_section_transition_result(
                            driver,
                            current_url,
                            timeout_seconds=0.3,
                        )
                        if transition_result == "holding":
                            print("Holding-status popup detected. Trying the next section quickly.")
                            active_section_name = None
                            continue
                        elif transition_result == "retry":
                            if only_preferred_sections:
                                print("Preferred section immediately reported unavailable. Waiting for it to open.")
                            else:
                                print("Section immediately reported unavailable. Removing it from priority.")
                            if (
                                can_exhaust_active_section
                                and (
                                    active_section_name in preferred_sections
                                    or active_section_name in normalized_allowed_sections
                                )
                            ):
                                exhausted_preferred_sections.add(active_section_name)
                            active_section_name = None
                            continue
                    else:
                        print("Zone page detected. Waiting for a section click opportunity.")
                        current_zone_section_signature = tuple(clickable_zone_sections)
                        if current_zone_section_signature and current_zone_section_signature != last_zone_section_signature:
                            last_zone_section_signature = current_zone_section_signature
                    last_signature = signature
                    last_url = current_url
                    if cycle < max_cycles:
                        time.sleep(poll_seconds)
                    continue
                elif zone_retry_error:
                    if go_back_to_zone_page(driver):
                        if (
                            can_exhaust_active_section
                            and (
                                active_section_name in preferred_sections
                                or active_section_name in normalized_allowed_sections
                            )
                        ):
                            exhausted_preferred_sections.add(active_section_name)
                        active_section_name = None
                        if wait_for_zone_page(driver, 0.6):
                            chosen_section_name, preferred_index, selection_kind = choose_next_section(
                                driver,
                                preferred_sections,
                                preferred_index,
                                exhausted_preferred_sections,
                                section_auto_click_is_preferred_only,
                                normalized_allowed_sections,
                            )
                            if chosen_section_name is not None:
                                active_section_name = chosen_section_name
                                if selection_kind == "preferred":
                                    print(f"Selected preferred section: {chosen_section_name}")
                                else:
                                    print(f"Selected fallback section: {chosen_section_name}")
                                continue
                        time.sleep(0.01)
                        continue

                if (
                    not candidates
                    and "zones.php" in current_url
                    and not sections
                ):
                    chosen_section_name, preferred_index, selection_kind = choose_next_section(
                        driver,
                        preferred_sections,
                        preferred_index,
                        exhausted_preferred_sections,
                        section_auto_click_is_preferred_only,
                        normalized_allowed_sections,
                    )
                    if chosen_section_name is not None:
                        active_section_name = chosen_section_name
                        if selection_kind == "preferred":
                            print(f"Selected preferred section: {chosen_section_name}")
                        else:
                            print(f"Selected fallback section: {chosen_section_name}")
                        transition_result = wait_for_section_transition_result(
                            driver,
                            current_url,
                            timeout_seconds=0.3,
                        )
                        if transition_result == "holding":
                            print("Holding-status popup detected. Trying the next section quickly.")
                            active_section_name = None
                            continue
                        elif transition_result == "retry":
                            if only_preferred_sections:
                                print("Preferred section immediately reported unavailable. Waiting for it to open.")
                            else:
                                print("Section immediately reported unavailable. Removing it from priority.")
                            if (
                                can_exhaust_active_section
                                and (
                                    active_section_name in preferred_sections
                                    or active_section_name in normalized_allowed_sections
                                )
                            ):
                                exhausted_preferred_sections.add(active_section_name)
                            active_section_name = None
                        continue

                if not candidates and preferred_sections:
                    if "zones.php" in current_url:
                        chosen_section_name, preferred_index, selection_kind = choose_next_section(
                            driver,
                            preferred_sections,
                            preferred_index,
                            exhausted_preferred_sections,
                            section_auto_click_is_preferred_only,
                            normalized_allowed_sections,
                        )
                        if chosen_section_name is not None:
                            active_section_name = chosen_section_name
                            if selection_kind == "preferred":
                                print(f"Selected preferred section: {chosen_section_name}")
                            else:
                                print(f"Selected fallback section: {chosen_section_name}")
                            transition_result = wait_for_section_transition_result(
                                driver,
                                current_url,
                                timeout_seconds=0.3,
                            )
                            if transition_result == "holding":
                                print("Holding-status popup detected. Trying the next section quickly.")
                                active_section_name = None
                                continue
                            elif transition_result == "retry":
                                if only_preferred_sections:
                                    print("Preferred section immediately reported unavailable. Waiting for it to open.")
                                else:
                                    print("Section immediately reported unavailable. Removing it from priority.")
                                if (
                                    can_exhaust_active_section
                                    and (
                                        active_section_name in preferred_sections
                                        or active_section_name in normalized_allowed_sections
                                    )
                                ):
                                    exhausted_preferred_sections.add(active_section_name)
                                active_section_name = None
                            continue
                if not candidates and sections and automatic_fallback_sections_enabled:
                    chosen_section = click_available_section(
                        driver,
                        sections,
                        allowed_sections=normalized_allowed_sections or None,
                    )
                    if chosen_section is not None:
                        print(f"Selected section: {chosen_section.section}")
                        active_section_name = chosen_section.section
                        if preferred_sections and chosen_section.section in preferred_sections:
                            preferred_index = preferred_sections.index(chosen_section.section)
                        wait_for_section_transition_result(
                            driver,
                            current_url,
                            timeout_seconds=0.3,
                        )
                        continue
                if not candidates and "zones.php" in current_url and automatic_fallback_sections_enabled:
                    fallback_section = click_any_available_section_fast(
                        driver,
                        excluded_sections=exhausted_preferred_sections,
                        allowed_sections=normalized_allowed_sections or None,
                    )
                    if fallback_section is not None:
                        print(f"Selected fallback section: {fallback_section}")
                        active_section_name = fallback_section
                        transition_result = wait_for_section_transition_result(
                            driver,
                            current_url,
                            timeout_seconds=0.3,
                        )
                        if transition_result == "holding":
                            print("Holding-status popup detected. Trying another section.")
                            active_section_name = None
                            continue
                        elif transition_result == "retry":
                            print("Fallback section immediately reported unavailable. Trying another section.")
                            if fallback_section in normalized_allowed_sections:
                                exhausted_preferred_sections.add(fallback_section)
                            active_section_name = None
                            continue
                        continue
                if page_mode == "zone":
                    last_signature = signature
                    last_url = current_url
                    if cycle < max_cycles:
                        time.sleep(poll_seconds)
                    continue
                if 0 < get_selected_seat_count(driver) < pair_size and clear_selected_seats(driver):
                    print("Cleared leftover partial seat selection before retrying.")
                    continue
                selected = wait_and_click_adjacent_pair_with_exclusions(
                    driver,
                    pair_size,
                    set(),
                    timeout_seconds=0.75,
                )
                if selected:
                    seats = ", ".join(
                        f"{seat.section}/{seat.row}/{seat.seat_number}" for seat in selected
                    )
                    selection_result = wait_for_seat_selection_result(
                        driver,
                        expected_count=pair_size,
                        timeout_seconds=0.25,
                    )
                    if selection_result == "selected":
                        print(f"Selected seats: {seats}")
                    elif selection_result == "holding":
                        clear_selected_seats(driver)
                        print("Seat click triggered a holding/unavailable message. Trying another pair.")
                        continue
                    elif selection_result == "retry":
                        clear_selected_seats(driver)
                        print("Seat click was rejected as unavailable. Trying another pair.")
                        continue
                    else:
                        clear_selected_seats(driver)
                        print("Seat click did not stick. Trying another pair.")
                        continue
                    if auto_advance:
                        if click_book_now(driver) or (time.sleep(0.03) is None and click_book_now(driver)):
                            print("Clicked Book Now.")
                            time.sleep(0.08)
                            current_url = driver.current_url
                            current_snapshot = strip_html(collect_browser_snapshot(driver))
                            if looks_like_payment_page(current_snapshot, current_url) and not continue_to_payment:
                                print("Reached payment step. Stopping before payment.")
                                return 0

                        advance_labels = ("next", "continue", "confirm", "reserve", "book")
                        while True:
                            current_url = driver.current_url
                            combined_text = collect_browser_snapshot(driver)
                            normalized = strip_html(combined_text)

                            if looks_like_thanks_page(normalized, current_url):
                                print("Reached thank-you page.")
                                return 0
                            if looks_like_payment_page(normalized, current_url) and not continue_to_payment:
                                print("Reached payment step. Stopping before payment.")
                                return 0

                            button = click_best_action_button(driver, advance_labels)
                            if button is None:
                                print("No obvious next/confirm button found after seat selection.")
                                return 0

                            print(f"Clicked action button: {button.text or button.description}")
                            time.sleep(0.1)

                            if continue_to_payment and looks_like_payment_page(
                                strip_html(collect_browser_snapshot(driver)),
                                driver.current_url,
                            ):
                                print("Reached payment page.")
                                return 0
                            if looks_like_thanks_page(
                                strip_html(collect_browser_snapshot(driver)),
                                driver.current_url,
                            ):
                                print("Reached thank-you page.")
                                return 0

                    print("Seat selection completed.")
                    return 0
                elif page_mode == "seat" and preferred_sections:
                    should_exhaust_active_section = not candidates
                    if (
                        should_exhaust_active_section
                        and can_exhaust_active_section
                        and (
                            active_section_name in preferred_sections
                            or active_section_name in normalized_allowed_sections
                        )
                    ):
                        exhausted_preferred_sections.add(active_section_name)
                    attempted_pairs_by_page.pop(current_url, None)
                    if go_back_to_zone_page(driver):
                        if should_exhaust_active_section:
                            print("Section appears unavailable. Removing it from priority and trying the next section.")
                        else:
                            print("No usable adjacent seats found. Trying the next preferred section for now.")
                        if wait_for_zone_page(driver, 0.3):
                            active_section_name = None
                            chosen_section_name, preferred_index, selection_kind = choose_next_section(
                                driver,
                                preferred_sections,
                                preferred_index,
                                exhausted_preferred_sections,
                                section_auto_click_is_preferred_only,
                                normalized_allowed_sections,
                            )
                            if chosen_section_name is not None:
                                active_section_name = chosen_section_name
                                if selection_kind == "preferred":
                                    print(f"Selected preferred section: {chosen_section_name}")
                                else:
                                    print(f"Selected fallback section: {chosen_section_name}")
                                wait_for_section_transition_result(
                                    driver,
                                    driver.current_url,
                                    timeout_seconds=0.3,
                                )
                                continue
                        time.sleep(0.01)
                        continue

            if zone_retry_error and page_mode != "seat":
                if go_back_to_zone_page(driver):
                    if (
                        can_exhaust_active_section
                        and (
                            active_section_name in preferred_sections
                            or active_section_name in normalized_allowed_sections
                        )
                    ):
                        exhausted_preferred_sections.add(active_section_name)
                    active_section_name = None
                    if wait_for_zone_page(driver, 0.3):
                        chosen_section_name, preferred_index, selection_kind = choose_next_section(
                            driver,
                            preferred_sections,
                            preferred_index,
                            exhausted_preferred_sections,
                            section_auto_click_is_preferred_only,
                            normalized_allowed_sections,
                        )
                        if chosen_section_name is not None:
                            active_section_name = chosen_section_name
                            if selection_kind == "preferred":
                                print(f"Selected preferred section: {chosen_section_name}")
                            else:
                                print(f"Selected fallback section: {chosen_section_name}")
                            continue
                    time.sleep(0.01)
                    continue

            last_signature = signature
            last_url = current_url

            if cycle < max_cycles:
                time.sleep(poll_seconds)
    finally:
        print()
        if effective_keep_browser_open:
            print("Leaving browser open.")
        else:
            print("Closing browser...")
            driver.quit()

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze saved ticket page HTML snapshots for seat/status text."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze one saved HTML snapshot."
    )
    analyze_parser.add_argument("snapshot", type=Path, help="Path to saved HTML file.")
    analyze_parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show relevant lines even if they are not classified.",
    )
    analyze_parser.add_argument(
        "--pair-size",
        type=int,
        default=2,
        help="Find side-by-side groups of this size. Default: 2.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare two saved HTML snapshots."
    )
    compare_parser.add_argument("old_snapshot", type=Path, help="Older HTML snapshot.")
    compare_parser.add_argument("new_snapshot", type=Path, help="Newer HTML snapshot.")

    live_parser = subparsers.add_parser(
        "live", help="Fetch and analyze a live ticket page."
    )
    live_parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL to fetch. Default: {DEFAULT_URL}",
    )
    live_parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Request timeout in seconds. Default: 15.",
    )
    live_parser.add_argument(
        "--save-to",
        type=Path,
        help="Optional path to save the fetched HTML snapshot.",
    )
    live_parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show relevant lines even if they are not classified.",
    )
    live_parser.add_argument(
        "--pair-size",
        type=int,
        default=2,
        help="Find side-by-side groups of this size. Default: 2.",
    )

    browser_parser = subparsers.add_parser(
        "browser-watch",
        help="Open Edge, let you reach the seat page, then scan the loaded DOM for adjacent seats.",
    )
    browser_parser.add_argument(
        "--url",
        help=(
            "Optional URL to open in Edge. "
            "If omitted while using --debugger-address, the current tab is left as-is."
        ),
    )
    browser_parser.add_argument(
        "--pair-size",
        type=int,
        default=2,
        help="Find side-by-side groups of this size. Default: 2.",
    )
    browser_parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.4,
        help="Seconds between scans after the booking tab reaches the zone or seat page. Default: 0.4.",
    )
    browser_parser.add_argument(
        "--max-cycles",
        type=int,
        default=40,
        help="Maximum number of scans before exiting. Default: 40.",
    )
    browser_parser.add_argument(
        "--save-to",
        type=Path,
        help="Optional path to save the current page HTML on each scan.",
    )
    browser_parser.add_argument(
        "--user-data-dir",
        type=Path,
        help="Optional Edge user data directory for keeping session state between runs.",
    )
    browser_parser.add_argument(
        "--lean-browser",
        action="store_true",
        help="When launching Edge from the script, disable extensions and background browser features to reduce overhead.",
    )
    browser_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Edge headlessly. Usually leave this off for manual login/queue flows.",
    )
    browser_parser.add_argument(
        "--confirm-start",
        action="store_true",
        help="After the booking tab reaches the zone or seat page, wait for Enter before scanning.",
    )
    browser_parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Try to click the first detected side-by-side group automatically.",
    )
    browser_parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="After selecting seats, try to click visible next/confirm buttons automatically.",
    )
    browser_parser.add_argument(
        "--continue-to-payment",
        action="store_true",
        help="Allow auto-advance to continue onto the payment step. Default behavior stops before payment.",
    )
    browser_parser.add_argument(
        "--debugger-address",
        help="Attach to an already running Edge started with --remote-debugging-port, for example 127.0.0.1:9222.",
    )
    browser_parser.add_argument(
        "--keep-browser-open",
        action="store_true",
        help="Leave the browser running when the script exits.",
    )
    browser_parser.add_argument(
        "--preferred-sections",
        help="Comma-separated section IDs to try first on the zone map, for example 301,302,303,420.",
    )
    browser_parser.add_argument(
        "--allowed-sections",
        help=(
            "Comma-separated section IDs that fallback is allowed to use. "
            "Use this to stay inside one ticket category, for example Cat 5."
        ),
    )
    browser_parser.add_argument(
        "--only-preferred-sections",
        action="store_true",
        help="Do not fall back to any other section when --preferred-sections is set.",
    )
    browser_parser.add_argument(
        "--manual-section-fallback",
        action="store_true",
        help="Try preferred sections first, then wait for you to click other sections manually.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return command_analyze(args.snapshot, args.show_all, args.pair_size)
    if args.command == "compare":
        return command_compare(args.old_snapshot, args.new_snapshot)
    if args.command == "live":
        return command_live(
            args.url,
            args.timeout,
            args.save_to,
            args.show_all,
            args.pair_size,
        )
    if args.command == "browser-watch":
        return command_browser_watch(
            args.url,
            args.pair_size,
            args.poll_seconds,
            args.max_cycles,
            args.save_to,
            args.headless,
            args.user_data_dir,
            args.lean_browser,
            args.confirm_start,
            args.auto_select,
            args.auto_advance,
            args.continue_to_payment,
            args.debugger_address,
            args.keep_browser_open,
            [
                part.strip()
                for part in (args.preferred_sections or "").split(",")
                if part.strip()
            ],
            args.only_preferred_sections,
            [
                part.strip()
                for part in (args.allowed_sections or "").split(",")
                if part.strip()
            ],
            args.manual_section_fallback,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
