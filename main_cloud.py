import os
import re
import json
import time
import html
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# استفاده از curl_cffi برای دور زدن Cloudflare و خطای 403 سایت‌ها
try:
    from curl_cffi import requests as cf_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

from google import genai
from google.genai import types


# ==============================================================================
# CONFIG
# ==============================================================================

TELEGRAM_CHANNELS = [
    "expertapply",
    "ApplyIR2UK",
    "applyforfree",
    "pargarwiki",
    "applyclub",
    "ApplyDaily",
    "computer_phd_apply",
    "EuropeanPhD",
    "PargarPositions",
]

SEARCH_QUERIES = [
    "Visual Inertial Odometry",
    "Visual SLAM",
    "Robot Localization",
    "Autonomous Navigation",
    "Sensor Fusion Robotics",
    "LiDAR SLAM",
    "GNSS Navigation",
    "State Estimation Robotics",
]

SEEN_FILE = "seen_professors.json"

# ارسال پیام گزارش وضعیت حتی در روزهایی که پوزیشن جدیدی یافت نمی‌شود
SEND_EMPTY_RUN_SUMMARY = True

# ----------------------------------------------------------------------
# Gemini Configuration (با پشتیبانی از مدل جایگزین خودکار)
# ----------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PRIMARY_MODEL = os.getenv("GEMINI_EVAL_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
ACTIVE_GEMINI_MODEL = PRIMARY_MODEL

# ----------------------------------------------------------------------
# Limits & Windows
# ----------------------------------------------------------------------

GEMINI_BATCH_SIZE = 10
MAX_TELEGRAM_PAGES_PER_CHANNEL = 3  # بررسی تا ۳ صفحه به عقب در هر کانال تلگرام
MAX_TELEGRAM_CANDIDATES_PER_RUN = 40
MAX_WEB_CANDIDATES_PER_RUN = 45

TELEGRAM_DAYS_BACK = 45
MIN_RELEVANT_SCORE = 65

GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 4

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
}


# ==============================================================================
# HTTP FETCHER (ضد تحریم و Cloudflare)
# ==============================================================================

def smart_get(url, timeout=25, use_impersonation=True):
    """
    Uses curl_cffi browser impersonation first to bypass Cloudflare (403),
    falling back to standard requests if unavailable.
    """
    if use_impersonation and HAS_CURL_CFFI:
        for browser in ("chrome", "safari"):
            try:
                resp = cf_requests.get(
                    url,
                    headers=REQUEST_HEADERS,
                    impersonate=browser,
                    timeout=timeout,
                    allow_redirects=True,
                )
                if resp.status_code == 200 and len(resp.text) > 200:
                    return resp.status_code, resp.text
            except Exception:
                continue

    try:
        resp = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        return resp.status_code, resp.text
    except Exception as exc:
        print(f"   ⚠️ HTTP request error for {url[:65]}: {exc}")
        return 0, ""


# ==============================================================================
# LOCAL FILTERS (پشتیبانی کامل انگلیسی و فارسی)
# ==============================================================================

PHD_PATTERNS = [
    r"\bph\.?\s*d\.?\b",
    r"\bphd\b",
    r"\bdoctoral\b",
    r"\bdoctorate\b",
    r"\bphd position\b",
    r"\bphd studentship\b",
    r"\bfunded phd\b",
    r"\bfully funded phd\b",
    r"\bphd vacancy\b",
    r"\bphd opening\b",
    r"\bphd candidate\b",
    r"دکتری",
    r"دکترا",
    r"پوزیشن",
    r"فول[- ]?فاند",
    r"بورسیه",
    r"دانشجوی دکتری",
    r"موقعیت دکتری",
    r"فرصت دکتری",
]

STRONG_RESEARCH_PATTERNS = [
    # VIO / SLAM
    r"\bvisual inertial\b",
    r"\bvisual-inertial\b",
    r"\bvisual odometry\b",
    r"\binertial odometry\b",
    r"\bvisual navigation\b",
    r"\bvisual localization\b",
    r"\bvisual localisation\b",
    r"\bslam\b",
    r"\bvins\b",
    r"\bodometry\b",

    # Localization / Navigation / Estimation
    r"\brobot localization\b",
    r"\brobot localisation\b",
    r"\bstate estimation\b",
    r"\bpose estimation\b",
    r"\bfactor graph\b",
    r"\bkalman\b",
    r"\bpositioning\b",
    r"\blocalization\b",
    r"\blocalisation\b",
    r"\bnavigation\b",

    # Sensors / Fusion
    r"\bgnss\b",
    r"\bgps\b",
    r"\bins\b",
    r"\bgnss/ins\b",
    r"\bsensor fusion\b",
    r"\bmulti[- ]sensor\b",
    r"\blidar\b",
    r"\bpoint cloud\b",
    r"\bcamera[- ]imu\b",
    r"\bevent camera\b",

    # Robotics / Autonomous Systems
    r"\brobotics\b",
    r"\brobotic\b",
    r"\bmobile robot\b",
    r"\bautonomous robot\b",
    r"\bautonomous navigation\b",
    r"\bautonomous driving\b",
    r"\bautonomous vehicle\b",
    r"\bautonomous systems\b",
    r"\buav\b",
    r"\bdrone\b",

    # فارسی
    r"رباتیک",
    r"بینایی ماشین",
    r"بینایی کامپیوتر",
    r"ناوبری",
    r"تخمین وضعیت",
    r"تخمین موقعیت",
    r"مکان[- ]?یابی",
    r"سیستم‌های خودران",
    r"خودروهای خودران",
    r"خودران",
    r"سنسور فیوژن",
    r"ترکیب سنسور",
    r"پهپاد",
]

ADJACENT_RESEARCH_PATTERNS = [
    r"\bcomputer vision\b",
    r"\b3d vision\b",
    r"\b3d reconstruction\b",
    r"\bdepth estimation\b",
    r"\bstereo vision\b",
    r"\brobot perception\b",
    r"\bperception\b",
    r"\bembedded ai\b",
    r"\bedge ai\b",
    r"\bdeep learning\b",
    r"\bimage processing\b",
    r"پردازش تصویر",
    r"یادگیری عمیق",
    r"هوش مصنوعی",
]

# رشته‌های کاملاً غیرمرتبط
UNRELATED_FIELD_PATTERNS = [
    r"\bchemistry\b",
    r"\bchemical engineering\b",
    r"\bmolecular biology\b",
    r"\bvirology\b",
    r"\bimmunology\b",
    r"\bsociology\b",
    r"\bliterature\b",
]

NON_PHD_ONLY_PATTERNS = [
    r"\bpostdoc only\b",
    r"\bpostdoctoral fellow\b",
    r"\bundergraduate internship\b",
]


def regex_any(text, patterns):
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def count_matches(text, patterns):
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def local_candidate_filter(title, description, assume_phd=False):
    text = f"{title}\n{description}".lower()

    if regex_any(text, UNRELATED_FIELD_PATTERNS):
        return False

    has_phd = assume_phd or regex_any(text, PHD_PATTERNS)
    if not has_phd:
        return False

    if regex_any(text, NON_PHD_ONLY_PATTERNS) and not regex_any(title.lower(), PHD_PATTERNS):
        return False

    strong_count = count_matches(text, STRONG_RESEARCH_PATTERNS)
    adjacent_count = count_matches(text, ADJACENT_RESEARCH_PATTERNS)

    return strong_count >= 1 or adjacent_count >= 2


# ==============================================================================
# UTILS
# ==============================================================================

def normalize_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_uid(url, source=""):
    """Generate a stable UID based primarily on canonical URL so minor text edits don't duplicate."""
    clean_url = url.split("#")[0].rstrip("/").strip().lower()
    raw = f"{source}|{clean_url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_seen():
    if not os.path.exists(SEEN_FILE):
        return set()
    try:
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        if isinstance(data, dict):
            return set(data.keys())
    except Exception as e:
        print(f"⚠️ Could not load seen DB: {e}")
    return set()


def save_seen(seen):
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(seen), f, ensure_ascii=False, indent=2)
        print(f"💾 Seen DB saved: {len(seen)} entries")
    except Exception as e:
        print(f"⚠️ Could not save seen DB: {e}")


def is_recent(dt, days=TELEGRAM_DAYS_BACK):
    if not dt:
        return True
    try:
        if dt.endswith("Z"):
            dt = dt[:-1] + "+00:00"
        parsed = datetime.fromisoformat(dt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return parsed >= cutoff
    except Exception:
        return True


# ==============================================================================
# 1. TELEGRAM SCRAPER (با BeautifulSoup و صفحه‌بندی به عقب)
# ==============================================================================

def scrape_telegram_channel(channel):
    results = []
    before_id = None

    for _page_num in range(MAX_TELEGRAM_PAGES_PER_CHANNEL):
        url = f"https://t.me/s/{channel}"
        if before_id:
            url += f"?before={before_id}"

        status, page_html = smart_get(url, timeout=20, use_impersonation=False)
        if status != 200 or not page_html:
            break

        soup = BeautifulSoup(page_html, "html.parser")
        message_nodes = soup.select("div.tgme_widget_message")
        if not message_nodes:
            break

        oldest_id_on_page = None
        reached_old_posts = False

        for msg in message_nodes:
            data_post = msg.get("data-post", "")  # e.g. "expertapply/15420"
            if not data_post or "/" not in data_post:
                continue

            post_id_str = data_post.split("/")[-1]
            if post_id_str.isdigit():
                post_id = int(post_id_str)
                if oldest_id_on_page is None or post_id < oldest_id_on_page:
                    oldest_id_on_page = post_id

            post_url = f"https://t.me/{data_post}"

            # استخراج تاریخ پست
            time_tag = msg.select_one("a.tgme_widget_message_date time") or msg.select_one("time")
            post_date = time_tag.get("datetime") if time_tag else None

            if not is_recent(post_date):
                reached_old_posts = True
                continue

            # استخراج متن اصلی پیام (و نادیده گرفتن متن ریپلای‌شده در بالای پیام)
            body_Bubble = msg.select_one("div.tgme_widget_message_bubble") or msg
            text_divs = body_Bubble.select("div.js-message_text")
            if not text_divs:
                continue

            # آخرین js-message_text مربوط به بدنه اصلی پست است (اولی ممکن است ریپلای باشد)
            main_text_div = text_divs[-1]
            for br in main_text_div.find_all("br"):
                br.replace_with("\n")

            # استخراج لینک‌های خارجی موجود در متن پست (مثل لینک دانشگاه یا استاد)
            external_links = []
            for a_tag in main_text_div.find_all("a", href=True):
                href = a_tag["href"].strip()
                if href.startswith("http") and "t.me/" not in href:
                    external_links.append(href)

            raw_text = main_text_div.get_text(separator=" ")
            lines = [normalize_text(line) for line in main_text_div.get_text().splitlines() if normalize_text(line)]
            text = "\n".join(lines)

            if not text or len(text) < 40:
                continue

            title = lines[0][:220] if lines else text[:220]

            if not local_candidate_filter(title, text, assume_phd=False):
                continue

            full_desc = text[:4500]
            if external_links:
                full_desc += "\nExternal Links: " + ", ".join(external_links[:3])

            uid = make_uid(post_url, "telegram")
            results.append({
                "uid": uid,
                "title": title,
                "description": full_desc,
                "url": post_url,
                "source": f"Telegram @{channel}",
                "type": "telegram",
            })

        if reached_old_posts or not oldest_id_on_page:
            break
        before_id = oldest_id_on_page
        time.sleep(0.4)

    return results


def collect_telegram_candidates(seen):
    candidates = []
    print("\n================ TELEGRAM CHANNELS ================\n")

    for channel in TELEGRAM_CHANNELS:
        print(f"📡 Scanning @{channel}...")
        channel_results = scrape_telegram_channel(channel)
        new_results = [
            item for item in channel_results
            if item["uid"] not in seen
        ]
        print(f"   ✓ {len(channel_results)} matched filter | {len(new_results)} new")
        candidates.extend(new_results)

    unique = {item["uid"]: item for item in candidates}
    return list(unique.values())[:MAX_TELEGRAM_CANDIDATES_PER_RUN]


# ==============================================================================
# 2. MULTI-SOURCE WEB SCRAPER (FindAPhD + EURAXESS + ScholarshipDB)
# ==============================================================================

def scrape_findaphd_keyword(keyword):
    encoded = quote(keyword)
    url = f"https://www.findaphd.com/phds/?Keywords={encoded}"

    status, page = smart_get(url, timeout=25, use_impersonation=True)
    if status != 200 or not page:
        print(f"   ⚠️ FindAPhD HTTP {status} for '{keyword}'")
        return []

    soup = BeautifulSoup(page, "html.parser")
    results = []

    # پشتیبانی از ساختارهای مختلف کارت‌های FindAPhD
    for a_tag in soup.select('a[href*="/phds/project/"]'):
        href = a_tag.get("href", "").strip()
        if not href:
            continue

        title = normalize_text(a_tag.get_text(" "))
        if len(title) < 15 or title.lower() in ("read more", "view project", "apply now", "more info"):
            continue

        full_url = href if href.startswith("http") else f"https://www.findaphd.com{href}"
        full_url = full_url.split("?")[0].rstrip("/")

        # تلاش برای استخراج توضیحات کارت پروژه
        card = a_tag.find_parent("div", class_=re.compile(r"phd-result|resultsRow|card", re.I))
        snippet = normalize_text(card.get_text(" ")) if card else ""
        desc = f"FindAPhD Project ({keyword}): {title}. {snippet[:1200]}"

        if not local_candidate_filter(title, desc, assume_phd=True):
            continue

        uid = make_uid(full_url, "findaphd")
        results.append({
            "uid": uid,
            "title": title[:240],
            "description": desc,
            "url": full_url,
            "source": "FindAPhD",
            "type": "web",
        })

    return results


def scrape_euraxess_keyword(keyword):
    """
    EURAXESS (Official European Commission Research Job Portal)
    Openly accessible from GitHub Actions IPs without Cloudflare blocks.
    """
    encoded = quote(keyword)
    # stage[]=1 corresponds to First Stage Researcher (R1 - PhD candidates)
    url = f"https://euraxess.ec.europa.eu/jobs/search?keywords={encoded}&research_profile%5B0%5D=437"

    status, page = smart_get(url, timeout=25, use_impersonation=False)
    if status != 200 or not page:
        return []

    soup = BeautifulSoup(page, "html.parser")
    results = []

    for article in soup.select("article, div.views-row, li.ecl-content-item-block__item"):
        link_tag = article.select_one('a[href*="/jobs/"]')
        if not link_tag:
            continue

        href = link_tag.get("href", "").strip()
        if not re.search(r"/jobs/\d+", href):
            continue

        title = normalize_text(link_tag.get_text(" "))
        if len(title) < 12:
            continue

        full_url = href if href.startswith("http") else f"https://euraxess.ec.europa.eu{href}"
        snippet = normalize_text(article.get_text(" "))
        desc = f"EURAXESS European PhD/R1 Position ({keyword}): {title}. {snippet[:1500]}"

        if not local_candidate_filter(title, desc, assume_phd=True):
            continue

        uid = make_uid(full_url, "euraxess")
        results.append({
            "uid": uid,
            "title": title[:240],
            "description": desc,
            "url": full_url,
            "source": "EURAXESS Europe",
            "type": "web",
        })

    return results


def scrape_scholarshipdb_keyword(keyword):
    """
    ScholarshipDB aggregates global PhD openings. Uses curl_cffi to bypass Cloudflare.
    """
    encoded = quote(f'"{keyword}" PhD')
    url = f"https://scholarshipdb.net/scholarships?q={encoded}&l="

    status, page = smart_get(url, timeout=25, use_impersonation=True)
    if status != 200 or not page:
        return []

    soup = BeautifulSoup(page, "html.parser")
    results = []

    for li in soup.select("ul.list-unstyled li, div.job-listing"):
        a_tag = li.select_one('h4 a[href*="/jobs-in-"], a[href*="-Phd-"]') or li.select_one("h4 a")
        if not a_tag:
            continue

        href = a_tag.get("href", "").strip()
        title = normalize_text(a_tag.get_text(" "))
        if not href or len(title) < 12:
            continue

        full_url = href if href.startswith("http") else f"https://scholarshipdb.net{href}"
        snippet = normalize_text(li.get_text(" "))
        desc = f"ScholarshipDB PhD Listing ({keyword}): {title}. {snippet[:1200]}"

        if not local_candidate_filter(title, desc, assume_phd=True):
            continue

        uid = make_uid(full_url, "scholarshipdb")
        results.append({
            "uid": uid,
            "title": title[:240],
            "description": desc,
            "url": full_url,
            "source": "ScholarshipDB",
            "type": "web",
        })

    return results


def collect_web_candidates(seen):
    candidates = []
    print("\n================ ACADEMIC WEB PORTALS ================\n")
    if not HAS_CURL_CFFI:
        print("ℹ️ Note: Install 'curl_cffi' in workflow to bypass Cloudflare on FindAPhD/ScholarshipDB.")

    for kw in SEARCH_QUERIES:
        print(f"🌐 Searching portals for: '{kw}'...")

        f_items = scrape_findaphd_keyword(kw)
        e_items = scrape_euraxess_keyword(kw)
        s_items = scrape_scholarshipdb_keyword(kw)

        combined = f_items + e_items + s_items
        new_items = [it for it in combined if it["uid"] not in seen]

        print(
            f"   ✓ Found: FindAPhD={len(f_items)}, EURAXESS={len(e_items)}, "
            f"ScholarshipDB={len(s_items)} | New={len(new_items)}"
        )
        candidates.extend(new_items)
        time.sleep(0.8)

    unique = {item["uid"]: item for item in candidates}
    return list(unique.values())[:MAX_WEB_CANDIDATES_PER_RUN]


# ==============================================================================
# GEMINI EVALUATION (با فال‌بک خودکار مدل و پارسینگ امن JSON)
# ==============================================================================

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is not configured.")
    raise SystemExit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_QUOTA_EXHAUSTED = False


def gemini_generate(contents, config=None, retries=GEMINI_MAX_RETRIES):
    global GEMINI_QUOTA_EXHAUSTED, ACTIVE_GEMINI_MODEL

    if GEMINI_QUOTA_EXHAUSTED:
        return None

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=ACTIVE_GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            error_text = str(e)
            lower = error_text.lower()
            print(f"   ⚠️ Gemini error [{ACTIVE_GEMINI_MODEL}] (attempt {attempt + 1}/{retries + 1}): {error_text[:200]}")

            # اگر نام مدل در دسترس نبود یا کلید محدودیت مدل داشت، مدل جایگزین را تست کن
            if "404" in lower or "not_found" in lower or "not supported" in lower:
                for candidate_model in FALLBACK_MODELS:
                    if candidate_model != ACTIVE_GEMINI_MODEL:
                        print(f"   🔄 Switching Gemini model to fallback: {candidate_model}")
                        ACTIVE_GEMINI_MODEL = candidate_model
                        break
                continue

            if "429" in lower or "resource_exhausted" in lower or "quota" in lower:
                GEMINI_QUOTA_EXHAUSTED = True
                print("\n🛑 Gemini quota exhausted during evaluation.")
                return None

            if attempt < retries:
                time.sleep(GEMINI_RETRY_DELAY * (attempt + 1))

    return None


def test_gemini_api():
    print("\n================ GEMINI API TEST ================\n")
    for model_name in dict.fromkeys(FALLBACK_MODELS):
        global ACTIVE_GEMINI_MODEL
        ACTIVE_GEMINI_MODEL = model_name
        response = gemini_generate(
            contents="Reply with: GEMINI_OK",
            retries=1,
        )
        if response is not None and getattr(response, "text", ""):
            print(f"✅ Gemini API test PASSED using model '{ACTIVE_GEMINI_MODEL}'.")
            return True

    print("❌ All Gemini models failed API test.")
    return False


EVALUATION_SYSTEM = """
You are evaluating PhD opportunities for a student specializing in:
- Visual-Inertial Odometry (VIO) & Visual SLAM / LiDAR SLAM
- Robot Localization & Autonomous Navigation (UGV, UAV, Autonomous Vehicles)
- State Estimation, Factor Graphs, Sensor Fusion (Camera, IMU, GNSS, LiDAR, Radar)
- 3D Computer Vision, Robot Perception & Real-Time Embedded/Edge AI for Robotics

Target: Funded PhD positions (including combined announcements that recruit both PhD and Postdoc).
Reject: Positions that are strictly Postdoc-only, Master's-only without PhD track, or unrelated fields.

Return a JSON object matching this exact schema:
{
  "results": [
    {
      "index": 0,
      "relevant": true,
      "score": 85,
      "reason": "Short explanation in English (max 25 words)"
    }
  ]
}

Scoring guide:
- 90-100: Direct match (VIO, SLAM, Inertial/GNSS Navigation, Robot Localization, Sensor Fusion)
- 75-89: Strong fit (Robot Perception, 3D Vision for Robotics, Autonomous Driving/UAV Navigation)
- 65-74: Good adjacent fit (Computer Vision, Deep Learning for Robotics/Edge AI, Control & Estimation)
- 0-64: Weak or irrelevant (set relevant=false)
"""


def parse_gemini_json(raw_text):
    """Safely parse Gemini JSON whether it returns a dict {'results': [...]} or a direct list [...]."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    if isinstance(data, dict):
        for key in ("results", "items", "candidates", "evaluations"):
            if isinstance(data.get(key), list):
                return data[key]
        return []
    if isinstance(data, list):
        return data
    return []


def evaluate_batch(batch):
    if GEMINI_QUOTA_EXHAUSTED:
        return None

    items_text = []
    for i, item in enumerate(batch):
        items_text.append(
            f"INDEX: {i}\nTITLE: {item.get('title', '')}\nSOURCE: {item.get('source', '')}\n"
            f"DESCRIPTION:\n{item.get('description', '')[:2800]}"
        )

    prompt = f"{EVALUATION_SYSTEM}\n\nCANDIDATES:\n" + "\n---\n".join(items_text)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
    )

    response = gemini_generate(
        contents=prompt,
        config=config,
    )

    if not response or not getattr(response, "text", ""):
        return None

    try:
        return parse_gemini_json(response.text)
    except Exception as e:
        print(f"   ⚠️ JSON Parse Error: {e} | Raw snippet: {response.text[:180]}")
        return None


def evaluate_candidates(candidates):
    if not candidates:
        return []

    print("\n================ GEMINI EVALUATION ================\n")
    evaluated = []

    for start in range(0, len(candidates), GEMINI_BATCH_SIZE):
        if GEMINI_QUOTA_EXHAUSTED:
            break

        batch = candidates[start:start + GEMINI_BATCH_SIZE]
        print(f"🧠 Evaluating batch {start + 1}-{start + len(batch)} of {len(candidates)}")

        results = evaluate_batch(batch)
        if results is None:
            print("   ⚠️ Batch evaluation failed; leaving these candidates unseen for next run.")
            continue

        for res in results:
            if not isinstance(res, dict):
                continue
            try:
                idx = int(res.get("index"))
                if 0 <= idx < len(batch):
                    cand = batch[idx].copy()
                    score = int(res.get("score", 0))
                    is_rel = bool(res.get("relevant", False)) and (score >= MIN_RELEVANT_SCORE)
                    cand["relevant"] = is_rel
                    cand["score"] = score
                    cand["reason"] = str(res.get("reason", ""))
                    evaluated.append(cand)
            except Exception:
                continue

        time.sleep(1.2)

    return evaluated


# ==============================================================================
# TELEGRAM NOTIFICATIONS
# ==============================================================================

def telegram_send_message(text):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("⚠️ Telegram bot credentials not configured.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        res = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                " disable_web_page_preview": True,
            },
            timeout=20,
        )
        if not res.ok:
            print(f"⚠️ Telegram API HTTP {res.status_code}: {res.text[:250]}")
        return res.ok
    except Exception as e:
        print(f"⚠️ Telegram notification failed: {e}")
        return False


def format_candidate(item):
    return (
        f"🎓 {item.get('title', 'Untitled')}\n\n"
        f"⭐ Score: {item.get('score', 0)}/100\n"
        f"📌 Source: {item.get('source', '')}\n"
        f"🧠 {item.get('reason', '')}\n\n"
        f"🔗 {item.get('url', '')}"
    )


def send_notifications(evaluated_results, total_scanned):
    relevant = [x for x in evaluated_results if x.get("relevant")]
    relevant.sort(key=lambda x: x.get("score", 0), reverse=True)

    if relevant:
        header = (
            f"🎯 PhD Opportunity Finder\n"
            f"Found {len(relevant)} relevant PhD opening(s) (out of {len(evaluated_results)} new candidates):\n"
        )
        telegram_send_message(header)
        time.sleep(0.4)

        for item in relevant:
            telegram_send_message(format_candidate(item))
            time.sleep(0.5)
    elif SEND_EMPTY_RUN_SUMMARY:
        summary = (
            f"ℹ️ PhD Opportunity Finder Status:\n"
            f"Run completed successfully.\n"
            f"• New candidates evaluated today: {len(evaluated_results)}\n"
            f"• Matches above score {MIN_RELEVANT_SCORE}: 0"
        )
        telegram_send_message(summary)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(
        """
====================================================
   PhD Opportunity Finder v2 (Telegram + Web Portals)
   VIO / SLAM / Navigation / GNSS / Robotics
====================================================
"""
    )

    seen = load_seen()
    print(f"📚 Seen DB loaded: {len(seen)} entries")

    if not test_gemini_api():
        telegram_send_message("❌ PhD Opportunity Finder Error: Gemini API test failed. Check GEMINI_API_KEY or model quota.")
        raise SystemExit(1)

    # ۱. استخراج از کانال‌های تلگرام (با صفحه‌بندی)
    telegram_candidates = collect_telegram_candidates(seen)
    print(f"\n📊 Total new Telegram candidates: {len(telegram_candidates)}")

    # ۲. استخراج از پورتال‌های وب (FindAPhD + EURAXESS + ScholarshipDB)
    web_candidates = collect_web_candidates(seen)
    print(f"\n📊 Total new Web portal candidates: {len(web_candidates)}")

    # ۳. تجمیع و حذف موارد تکراری بر اساس UID
    all_candidates = telegram_candidates + web_candidates
    unique = {}
    for it in all_candidates:
        uid = it["uid"]
        if uid not in unique and uid not in seen:
            unique[uid] = it

    candidates = list(unique.values())
    print(f"\n🚀 Total unique new candidates to evaluate: {len(candidates)}")

    if not candidates:
        print("⚠️ No new candidates found in this run.")
        if SEND_EMPTY_RUN_SUMMARY:
            telegram_send_message(
                "ℹ️ PhD Opportunity Finder Status:\n"
                "Scanned Telegram channels + FindAPhD + EURAXESS + ScholarshipDB.\n"
                "No new unseen candidates matched local filters today."
            )
        save_seen(seen)
        return

    # ۴. ارزیابی توسط مدل Gemini
    evaluated = evaluate_candidates(candidates)

    # فقط مواردی که با موفقیت توسط جمنای بررسی شدند به لیست seen اضافه می‌شوند
    for it in evaluated:
        seen.add(it["uid"])

    relevant = [x for x in evaluated if x.get("relevant")]
    relevant.sort(key=lambda x: x.get("score", 0), reverse=True)

    print(f"\n✅ Evaluated: {len(evaluated)} | 🎯 Relevant: {len(relevant)}")

    if relevant:
        print("\n================ TOP RESULTS ================\n")
        for item in relevant[:10]:
            print(f"⭐ {item.get('score')}/100 | {item.get('title')}")
            print(f"   {item.get('url')}")
            print(f"   {item.get('reason')}\n")

    send_notifications(evaluated, len(candidates))
    save_seen(seen)
    print("\n✅ Run finished successfully.")


if __name__ == "__main__":
    main()
