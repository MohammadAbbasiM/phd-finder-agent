import os
import re
import json
import time
import html
import hashlib
import requests

from datetime import datetime, timezone, timedelta
from urllib.parse import quote, urlparse

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

# کلیدواژه‌های جستجوی تخصصی در FindAPhD
FINDAPHD_QUERIES = [
    "Visual Inertial Odometry",
    "Visual SLAM",
    "Robot Localization",
    "Autonomous Navigation",
    "Sensor Fusion",
    "State Estimation Robotics",
]

SEEN_FILE = "seen_professors.json"

# ----------------------------------------------------------------------
# Gemini Configuration (صرفاً برای ارزیابی و امتیازدهی)
# ----------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_EVAL_MODEL = os.getenv(
    "GEMINI_EVAL_MODEL",
    "gemini-3.5-flash-lite"
)

# ----------------------------------------------------------------------
# Limits & Windows
# ----------------------------------------------------------------------

GEMINI_BATCH_SIZE = 10
MAX_TELEGRAM_CANDIDATES_PER_RUN = 30
MAX_FINDAPHD_CANDIDATES_PER_RUN = 30

# بررسی پست‌های تا ۶۰ روز گذشته
TELEGRAM_DAYS_BACK = 60

GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 4

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
}


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

    # Localization / Navigation
    r"\brobot localization\b",
    r"\brobot localisation\b",
    r"\bstate estimation\b",
    r"\bpose estimation\b",
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
    r"\bmulti[- ]sensor fusion\b",
    r"\blidar\b",
    r"\bcamera[- ]imu\b",

    # Robotics
    r"\brobotics\b",
    r"\bmobile robot\b",
    r"\bautonomous robot\b",
    r"\bautonomous navigation\b",
    r"\bautonomous systems\b",

    # فارسی
    r"رباتیک",
    r"بینایی ماشین",
    r"ناوبری",
    r"تخمین وضعیت",
    r"تخمین موقعیت",
    r"سیستم‌های خودران",
    r"خودران",
    r"سنسور فیوژن",
]

ADJACENT_RESEARCH_PATTERNS = [
    r"\bcomputer vision\b",
    r"\b3d vision\b",
    r"\b3d reconstruction\b",
    r"\brobot perception\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bimage processing\b",
    r"پردازش تصویر",
    r"یادگیری عمیق",
    r"هوش مصنوعی",
]

HARD_EXCLUDE_PATTERNS = [
    r"\bpostdoc\b",
    r"\bpostdoctoral\b",
    r"\bundergraduate\b",
    r"\binternship\b",
    r"\bbiology\b",
    r"\bchemistry\b",
    r"\bchemical engineering\b",
    r"\bmaterials science\b",
    r"\bcfd\b",
    r"پست داک",
    r"پسادکتری",
    r"کارآموزی",
]


def regex_any(text, patterns):
    text = text.lower()
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def count_matches(text, patterns):
    text = text.lower()
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def local_candidate_filter(title, description):
    text = f"{title}\n{description}".lower()

    if regex_any(text, HARD_EXCLUDE_PATTERNS):
        return False

    if not regex_any(text, PHD_PATTERNS):
        return False

    strong_count = count_matches(text, STRONG_RESEARCH_PATTERNS)
    adjacent_count = count_matches(text, ADJACENT_RESEARCH_PATTERNS)

    if strong_count >= 1 or adjacent_count >= 2:
        return True

    return False


# ==============================================================================
# UTILS
# ==============================================================================

def normalize_text(text):
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_uid(title, url, source):
    raw = f"{source}|{title}|{url}"
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
# TELEGRAM SCRAPER (اصلاح‌شده برای جلوگیری از باگ تگ‌های تو در تو)
# ==============================================================================

def scrape_telegram_channel(channel):
    url = f"https://t.me/s/{channel}"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        if response.status_code != 200:
            print(f"   ⚠️ Telegram HTTP {response.status_code}")
            return []
        page = response.text
    except Exception as e:
        print(f"   ⚠️ Telegram request failed: {e}")
        return []

    results = []

    # شکستن کل صفحه بر اساس مرز هر پست تلگرام برای جلوگیری از باگ رگکس قبلی
    raw_blocks = page.split('class="tgme_widget_message_wrap')
    if len(raw_blocks) <= 1:
        raw_blocks = page.split('class="tgme_widget_message ')

    for block in raw_blocks[1:]:
        # استخراج لینک پست
        url_match = re.search(r'href="(https://t\.me/[^"]+)"', block)
        if not url_match:
            continue
        post_url = html.unescape(url_match.group(1))

        # استخراج متن کامل پست
        text_match = re.search(
            r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>\s*<div class="tgme_widget_message_footer',
            block,
            flags=re.DOTALL,
        )
        if not text_match:
            text_match = re.search(
                r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>',
                block,
                flags=re.DOTALL,
            )

        text = ""
        if text_match:
            text = re.sub(r"<br\s*/?>", "\n", text_match.group(1))
            text = re.sub(r"<[^>]+>", " ", text)
            text = normalize_text(text)

        # استخراج تاریخ
        date_match = re.search(r'<time[^>]+datetime="([^"]+)"', block)
        post_date = date_match.group(1) if date_match else None

        if not is_recent(post_date):
            continue

        if not text:
            continue

        lines = [x.strip() for x in text.split("\n") if x.strip()]
        title = lines[0][:250] if lines else text[:250]

        if not local_candidate_filter(title, text):
            continue

        uid = make_uid(title, post_url, f"telegram:{channel}")
        results.append({
            "uid": uid,
            "title": title,
            "description": text[:5000],
            "url": post_url,
            "source": f"Telegram @{channel}",
            "type": "telegram",
        })

    return results


def collect_telegram_candidates(seen):
    candidates = []
    print("\n================ TELEGRAM ================\n")

    for channel in TELEGRAM_CHANNELS:
        print(f"📡 Scanning @{channel}...")
        channel_results = scrape_telegram_channel(channel)
        new_results = [item for item in channel_results if item["uid"] not in seen]
        print(f"   ✓ {len(new_results)} relevant candidates")

        candidates.extend(new_results)
        if len(candidates) >= MAX_TELEGRAM_CANDIDATES_PER_RUN:
            break

    unique = {item["uid"]: item for item in candidates}
    return list(unique.values())[:MAX_TELEGRAM_CANDIDATES_PER_RUN]


# ==============================================================================
# FINDAPHD DIRECT SCRAPER (اصلاح‌شده برای رفع خطای 404)
# ==============================================================================

def scrape_findaphd_keyword(keyword):
    encoded = quote(keyword)
    # آدرس رسمی و فعال موتور جستجوی FindAPhD
    url = f"https://www.findaphd.com/phds/?Keywords={encoded}"

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
        if response.status_code != 200:
            print(f"   ⚠️ FindAPhD HTTP {response.status_code}")
            return []
        page = response.text
    except Exception as e:
        print(f"   ⚠️ FindAPhD request failed: {e}")
        return []

    results = []

    # استخراج عنوان و آدرس کامل پروژه‌ها
    matches = re.findall(
        r'<a\s+[^>]*href="(/phds/project/[^"]+)"[^>]*>(.*?)</a>',
        page,
        flags=re.DOTALL,
    )

    for relative_url, raw_title in matches:
        title = normalize_text(re.sub(r"<[^>]+>", " ", raw_title))

        if len(title) < 15 or "read more" in title.lower() or "apply" in title.lower():
            continue

        full_url = f"https://www.findaphd.com{relative_url}"
        desc = f"PhD Opportunity on FindAPhD in {keyword}: {title}"

        if not local_candidate_filter(title, desc):
            continue

        uid = make_uid(title, full_url, "findaphd")
        results.append({
            "uid": uid,
            "title": title,
            "description": desc,
            "url": full_url,
            "source": "FindAPhD",
            "type": "web",
        })

    return results


def collect_findaphd_candidates(seen):
    candidates = []
    print("\n================ FINDAPHD SCRAPER ================\n")

    for kw in FINDAPHD_QUERIES:
        print(f"🌐 Scraping FindAPhD for: '{kw}'...")
        items = scrape_findaphd_keyword(kw)
        new_items = [it for it in items if it["uid"] not in seen]
        print(f"   ✓ {len(new_items)} candidates found")
        candidates.extend(new_items)
        time.sleep(1)

    unique = {item["uid"]: item for item in candidates}
    return list(unique.values())[:MAX_FINDAPHD_CANDIDATES_PER_RUN]


# ==============================================================================
# GEMINI EVALUATION
# ==============================================================================

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is not configured.")
    raise SystemExit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_QUOTA_EXHAUSTED = False


def gemini_generate(model, contents, config=None, retries=GEMINI_MAX_RETRIES):
    global GEMINI_QUOTA_EXHAUSTED

    if GEMINI_QUOTA_EXHAUSTED:
        return None

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response
        except Exception as e:
            error_text = str(e)
            lower = error_text.lower()
            print(f"   ⚠️ Gemini error (attempt {attempt + 1}/{retries + 1}): {error_text[:250]}")

            if "429" in lower or "resource_exhausted" in lower or "quota" in lower:
                GEMINI_QUOTA_EXHAUSTED = True
                print("\n🛑 Gemini quota exhausted during evaluation.")
                return None

            if "503" in lower or "unavailable" in lower:
                if attempt < retries:
                    wait = GEMINI_RETRY_DELAY * (attempt + 1)
                    time.sleep(wait)
                    continue

            if attempt < retries:
                time.sleep(2)

    return None


def test_gemini_api():
    print("\n================ GEMINI API TEST ================\n")
    response = gemini_generate(
        model=GEMINI_EVAL_MODEL,
        contents="Reply with: GEMINI_OK",
        retries=0,
    )
    if response is None:
        print("❌ Basic Gemini API test FAILED.")
        return False

    print(f"✅ Basic Gemini API test PASSED. ({getattr(response, 'text', '').strip()})")
    return True


EVALUATION_SYSTEM = """
You are evaluating PhD opportunities for a student with target research:
- Visual-Inertial Odometry (VIO)
- Visual SLAM / SLAM
- Visual Navigation / Robot Localization
- State Estimation / Sensor Fusion (Camera, IMU, GNSS, LiDAR)
- Mobile Robotics / 3D Vision

Target: Funded Direct PhD.
Reject: Non-PhD jobs, Master's-only, Unrelated engineering fields.

Output JSON schema:
{
  "results": [
    {
      "index": 0,
      "relevant": true,
      "score": 0-100,
      "reason": "concise English reason"
    }
  ]
}

Scoring:
90-100: Direct VIO/SLAM/Inertial Navigation/Robot Localization
75-89: Sensor Fusion / Perception / Robotics Vision
60-74: Adjacent autonomous systems
0-59: Weak or irrelevant
"""


def evaluate_batch(batch):
    if GEMINI_QUOTA_EXHAUSTED:
        return []

    items_text = []
    for i, item in enumerate(batch):
        items_text.append(
            f"INDEX: {i}\nTITLE: {item.get('title', '')}\nSOURCE: {item.get('source', '')}\n"
            f"DESCRIPTION:\n{item.get('description', '')[:3000]}"
        )

    prompt = f"{EVALUATION_SYSTEM}\n\nCANDIDATES:\n" + "\n---\n".join(items_text)

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    response = gemini_generate(
        model=GEMINI_EVAL_MODEL,
        contents=prompt,
        config=config,
    )

    if not response or not getattr(response, "text", ""):
        return []

    try:
        data = json.loads(response.text)
        return data.get("results", [])
    except Exception as e:
        print(f"   ⚠️ JSON Parse Error: {e}")
        return []


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
        for res in results:
            try:
                idx = int(res.get("index"))
                if 0 <= idx < len(batch):
                    cand = batch[idx].copy()
                    cand["relevant"] = bool(res.get("relevant", False))
                    cand["score"] = int(res.get("score", 0))
                    cand["reason"] = str(res.get("reason", ""))
                    evaluated.append(cand)
            except Exception:
                continue

        time.sleep(1)

    return evaluated


# ==============================================================================
# TELEGRAM NOTIFICATIONS
# ==============================================================================

def telegram_send_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        res = requests.post(
            url,
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            timeout=20,
        )
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


def send_notifications(results):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("⚠️ Telegram bot credentials not configured. Skipping notifications.")
        return

    relevant = [x for x in results if x.get("relevant")]
    relevant.sort(key=lambda x: x.get("score", 0), reverse=True)

    for item in relevant:
        telegram_send_message(bot_token, chat_id, format_candidate(item))
        time.sleep(0.5)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print(
        """
====================================================
      PhD Opportunity Finder (Telegram + FindAPhD)
      VIO / SLAM / Navigation / GNSS / Robotics
====================================================
"""
    )

    seen = load_seen()
    print(f"📚 Seen DB: {len(seen)} entries")

    if not test_gemini_api():
        print("\n❌ Gemini API test failed. Check API key.")
        return

    # ۱. استخراج از تلگرام
    telegram_candidates = collect_telegram_candidates(seen)
    print(f"\n📊 Total Telegram candidates: {len(telegram_candidates)}")

    # ۲. استخراج مستقیم از FindAPhD
    findaphd_candidates = collect_findaphd_candidates(seen)
    print(f"\n📊 Total FindAPhD candidates: {len(findaphd_candidates)}")

    # ۳. تجمیع و حذف موارد تکراری
    all_candidates = telegram_candidates + findaphd_candidates
    unique = {}
    for it in all_candidates:
        norm_url = it.get("url", "").split("#")[0].rstrip("/")
        if norm_url and norm_url not in unique:
            unique[norm_url] = it

    candidates = list(unique.values())
    print(f"\n🚀 Total new candidates to evaluate: {len(candidates)}")

    if not candidates:
        print("⚠️ No new candidates found.")
        save_seen(seen)
        return

    # ۴. ارزیابی توسط مدل جمنای
    evaluated = evaluate_candidates(candidates)

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

    send_notifications(relevant)
    save_seen(seen)
    print("\n✅ Run finished successfully.")


if __name__ == "__main__":
    main()
