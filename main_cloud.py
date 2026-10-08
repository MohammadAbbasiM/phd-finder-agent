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

SEEN_FILE = "seen_professors.json"

# ----------------------------------------------------------------------
# Gemini
# ----------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Current Gemini 3.8 Flash model IDs
GEMINI_SEARCH_MODEL = os.getenv(
    "GEMINI_SEARCH_MODEL",
    "gemini-3.8-flash"
)

GEMINI_EVAL_MODEL = os.getenv(
    "GEMINI_EVAL_MODEL",
    "gemini-3.8-flash"
)

# ----------------------------------------------------------------------
# Search / evaluation limits
# ----------------------------------------------------------------------

MAX_WEB_SEARCH_TASKS = 5
MAX_RESULTS_PER_WEB_TASK = 6

GEMINI_BATCH_SIZE = 15

MAX_TELEGRAM_CANDIDATES_PER_RUN = 40
MAX_WEB_CANDIDATES_PER_RUN = 30

# Telegram recent window
TELEGRAM_DAYS_BACK = 21

# Retry behavior
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 4


# ==============================================================================
# TELEGRAM
# ==============================================================================

TELEGRAM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0 Safari/537.36"
    )
}


# ==============================================================================
# LOCAL FILTERS
# ==============================================================================

PHD_PATTERNS = [
    r"\bph\.?\s*d\.?\b",
    r"\bphd\b",
    r"\bdoctoral\b",
    r"\bdoctorate\b",
    r"\bdoctoral researcher\b",
    r"\bdoctoral student\b",
    r"\bdoctoral candidate\b",
    r"\bphd position\b",
    r"\bphd studentship\b",
    r"\bphd studentship\b",
    r"\bfunded phd\b",
    r"\bfully funded phd\b",
    r"\bphd vacancy\b",
    r"\bphd opening\b",
    r"\bphd opportunity\b",
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
    r"\bsemantic slam\b",
    r"\bvisual slam\b",
    r"\bvins\b",

    # Localization / navigation
    r"\brobot localization\b",
    r"\brobot localisation\b",
    r"\bstate estimation\b",
    r"\bpose estimation\b",
    r"\bpositioning\b",
    r"\blocalization\b",
    r"\blocalisation\b",
    r"\bnavigation\b",

    # GNSS / INS
    r"\bgnss\b",
    r"\bgps\b",
    r"\bins\b",
    r"\bgnss/ins\b",
    r"\bgnss ins\b",
    r"\bsensor fusion\b",
    r"\bmulti[- ]sensor fusion\b",

    # Robotics
    r"\brobotics\b",
    r"\bmobile robot\b",
    r"\bautonomous robot\b",
    r"\bautonomous navigation\b",
    r"\bautonomous systems\b",

    # Perception
    r"\b3d perception\b",
    r"\b3d vision\b",
    r"\bcomputer vision\b",
    r"\brobot perception\b",
    r"\blidar\b",
    r"\bli[- ]?dar\b",
    r"\bcamera[- ]imu\b",
    r"\bcamera imu\b",

    # Related ML / geometry
    r"\bgeometric vision\b",
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\bmultimodal perception\b",
    r"\bscene understanding\b",
    r"\b3d reconstruction\b",
    r"\bstructure from motion\b",
    r"\bsensor calibration\b",
]

ADJACENT_RESEARCH_PATTERNS = [
    r"\bcomputer vision\b",
    r"\b3d vision\b",
    r"\b3d reconstruction\b",
    r"\brobot perception\b",
    r"\brobotics\b",
    r"\bautonomous systems\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bimage processing\b",
    r"\bmultimodal\b",
    r"\bperception\b",
]

HARD_EXCLUDE_PATTERNS = [
    # Jobs / academic levels
    r"\bpostdoc\b",
    r"\bpostdoctoral\b",
    r"\bpost-doctoral\b",
    r"\bundergraduate\b",
    r"\binternship\b",
    r"\bintern\b",

    # Biology / medicine
    r"\bbiology\b",
    r"\bmolecular\b",
    r"\bgenomics\b",
    r"\bgenetic\b",
    r"\bbiomedical\b",
    r"\bneuroscience\b",
    r"\bclinical\b",

    # Chemistry / materials
    r"\bchemistry\b",
    r"\bchemical engineering\b",
    r"\bmaterials science\b",
    r"\bmaterial science\b",
    r"\bpolymer\b",
    r"\bnanomaterial\b",

    # Mechanical
    r"\bmechanical design\b",
    r"\bmanufacturing\b",
    r"\bcfd\b",
    r"\bfluid dynamics\b",
    r"\bthermodynamics\b",
    r"\bstructural engineering\b",
    r"\bfinite element\b",

    # Other irrelevant
    r"\bprosthetic\b",
    r"\bbiomechanics\b",
    r"\bpanel discussion\b",
    r"\bapplication fee\b",
]


def regex_any(text, patterns):
    text = text.lower()

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


def count_matches(text, patterns):
    text = text.lower()

    count = 0

    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            count += 1

    return count


def local_candidate_filter(title, description):
    """
    Fast local filter.

    Goal:
    - must look like a PhD/doctoral opportunity
    - must contain strong VIO/SLAM/navigation/GNSS/robotics terms
      OR multiple adjacent research terms
    - reject obvious irrelevant areas
    """

    text = f"{title}\n{description}".lower()

    # Hard exclusions
    if regex_any(text, HARD_EXCLUDE_PATTERNS):
        return False

    # Must be PhD/doctoral
    has_phd = regex_any(text, PHD_PATTERNS)

    if not has_phd:
        return False

    strong_count = count_matches(text, STRONG_RESEARCH_PATTERNS)
    adjacent_count = count_matches(text, ADJACENT_RESEARCH_PATTERNS)

    # Very strong direct match
    if strong_count >= 1:
        return True

    # Broader adjacent opportunity
    if adjacent_count >= 2:
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

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


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
            json.dump(
                sorted(seen),
                f,
                ensure_ascii=False,
                indent=2,
            )

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
# TELEGRAM SCRAPER
# ==============================================================================

def scrape_telegram_channel(channel):
    url = f"https://t.me/s/{channel}"

    try:
        response = requests.get(
            url,
            headers=TELEGRAM_HEADERS,
            timeout=20,
        )

        if response.status_code != 200:
            print(
                f"   ⚠️ Telegram HTTP {response.status_code}"
            )
            return []

        page = response.text

    except Exception as e:
        print(f"   ⚠️ Telegram request failed: {e}")
        return []

    results = []

    # Telegram post blocks
    blocks = re.findall(
        r'<div class="tgme_widget_message_wrap".*?</div>\s*</div>',
        page,
        flags=re.DOTALL,
    )

    # Fallback if HTML structure differs
    if not blocks:
        blocks = re.findall(
            r'<div class="tgme_widget_message".*?</div>\s*</div>',
            page,
            flags=re.DOTALL,
        )

    for block in blocks:

        # Post URL
        url_match = re.search(
            r'href="(https://t\.me/[^"]+)"',
            block,
        )

        if not url_match:
            continue

        post_url = html.unescape(url_match.group(1))

        # Text
        text_match = re.search(
            r'<div class="tgme_widget_message_text[^>]*>(.*?)</div>',
            block,
            flags=re.DOTALL,
        )

        text = ""

        if text_match:
            text = re.sub(
                r"<br\s*/?>",
                "\n",
                text_match.group(1),
            )

            text = re.sub(
                r"<[^>]+>",
                " ",
                text,
            )

            text = normalize_text(text)

        # Date
        date_match = re.search(
            r'<time[^>]+datetime="([^"]+)"',
            block,
        )

        post_date = None

        if date_match:
            post_date = date_match.group(1)

        if not is_recent(post_date):
            continue

        if not text:
            continue

        # Use first line as rough title
        lines = [
            x.strip()
            for x in text.split("\n")
            if x.strip()
        ]

        title = lines[0][:250] if lines else text[:250]

        if not local_candidate_filter(title, text):
            continue

        uid = make_uid(
            title,
            post_url,
            f"telegram:{channel}",
        )

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

        new_results = []

        for item in channel_results:
            if item["uid"] not in seen:
                new_results.append(item)

        print(
            f"   ✓ {len(new_results)} relevant local candidates"
        )

        candidates.extend(new_results)

        if len(candidates) >= MAX_TELEGRAM_CANDIDATES_PER_RUN:
            break

    # Deduplicate
    unique = {}

    for item in candidates:
        unique[item["uid"]] = item

    candidates = list(unique.values())

    return candidates[:MAX_TELEGRAM_CANDIDATES_PER_RUN]


# ==============================================================================
# GEMINI
# ==============================================================================

if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY is not configured.")
    raise SystemExit(1)


client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_QUOTA_EXHAUSTED = False


def gemini_generate(
    model,
    contents,
    config=None,
    retries=GEMINI_MAX_RETRIES,
):
    """
    Central Gemini API wrapper.

    Important:
    - 503 -> retry
    - 429 / RESOURCE_EXHAUSTED / quota -> stop
    """

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

            print(
                f"   ⚠️ Gemini error "
                f"(attempt {attempt + 1}/{retries + 1}): "
                f"{error_text[:700]}"
            )

            lower = error_text.lower()

            # Quota / rate limit
            if (
                "429" in lower
                or "resource_exhausted" in lower
                or "quota" in lower
            ):
                GEMINI_QUOTA_EXHAUSTED = True

                print(
                    "\n🛑 Gemini quota/rate limit exhausted."
                )

                return None

            # Temporary server overload
            if (
                "503" in lower
                or "unavailable" in lower
                or "high demand" in lower
            ):
                if attempt < retries:
                    wait = GEMINI_RETRY_DELAY * (attempt + 1)

                    print(
                        f"   ⏳ Retrying in {wait}s..."
                    )

                    time.sleep(wait)

                    continue

            # Other errors
            if attempt < retries:
                time.sleep(2)

    return None


# ==============================================================================
# GEMINI BASIC API TEST
# ==============================================================================

def test_gemini_api():
    """
    Tiny ungrounded request.

    This separates:
      1. API key/project quota problem
      2. Web Search grounding problem
    """

    print("\n================ GEMINI API TEST ================\n")

    response = gemini_generate(
        model=GEMINI_EVAL_MODEL,
        contents="Reply with exactly: GEMINI_OK",
        config=types.GenerateContentConfig(),
        retries=0,
    )

    if response is None:
        print(
            "❌ Basic Gemini API test FAILED."
        )

        print(
            "   The problem is likely API quota/billing/project configuration."
        )

        return False

    text = getattr(response, "text", "")

    print("✅ Basic Gemini API test PASSED.")
    print(f"   Response: {text}")

    return True


# ==============================================================================
# GEMINI WEB SEARCH
# ==============================================================================

WEB_SEARCH_TASKS = [
    {
        "name": "Core VIO / SLAM",
        "query": """
Find current PhD / doctoral positions for Fall 2027 or upcoming intake
in Visual-Inertial Odometry, VIO, Visual SLAM, SLAM, Visual Odometry,
Localization, Pose Estimation, or State Estimation.

Focus on funded PhD positions in universities and research labs.
Prioritize robotics, autonomous systems, computer vision and navigation.
Exclude postdocs, internships, master's-only positions and unrelated fields.
""",
    },

    {
        "name": "Navigation / Localization",
        "query": """
Find current funded PhD / doctoral opportunities for Fall 2027 or upcoming intake
in robot localization, autonomous navigation, visual navigation,
robot navigation, state estimation, sensor fusion, or mobile robotics.

Prioritize positions involving cameras, IMU, LiDAR, GNSS, robotics,
or autonomous systems.
Exclude postdoctoral, internship, master's-only and unrelated positions.
""",
    },

    {
        "name": "Robotics / Perception",
        "query": """
Find funded PhD / doctoral positions for Fall 2027 or upcoming intake
in robotics perception, 3D vision, computer vision for robotics,
robot perception, 3D reconstruction, LiDAR-camera-IMU systems,
visual localization, or autonomous systems.

Prioritize research relevant to SLAM, VIO, localization, navigation
and sensor fusion.
Exclude unrelated biology, chemistry, materials and mechanical engineering.
""",
    },

    {
        "name": "GNSS / Sensor Fusion",
        "query": """
Find funded PhD / doctoral positions for Fall 2027 or upcoming intake
in GNSS, GPS, GNSS/INS, inertial navigation, sensor fusion,
multi-sensor positioning, localization, navigation or state estimation.

Prioritize research combining GNSS/INS with cameras, LiDAR,
robotics, autonomous systems or machine learning.
Exclude master's-only, internships and unrelated engineering fields.
""",
    },

    {
        "name": "Visual Navigation / Autonomous Systems",
        "query": """
Find funded PhD / doctoral positions for Fall 2027 or upcoming intake
in visual navigation, autonomous navigation, robot localization,
visual perception, computer vision for autonomous systems,
SLAM, VIO, sensor fusion or mobile robotics.

Search university PhD vacancy pages, laboratory pages,
professor/project pages and official doctoral advertisements.
Prioritize direct research matches to VIO/SLAM/navigation.
Exclude postdocs, internships and master's-only positions.
""",
    },
]


def gemini_web_search(task):
    if GEMINI_QUOTA_EXHAUSTED:
        return []

    print(
        f"\n🌐 Gemini Web Search: {task['name']}"
    )

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    # IMPORTANT:
    # Gemini 3.8 Flash does NOT use temperature here.
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
    )

    response = gemini_generate(
        model=GEMINI_SEARCH_MODEL,
        contents=task["query"],
        config=config,
    )

    if response is None:
        return []

    text = getattr(response, "text", "")

    if not text:
        return []

    return parse_web_search_response(text)


def parse_web_search_response(text):
    """
    Gemini may return ordinary prose rather than strict JSON.
    We extract URLs and nearby context.
    """

    results = []

    # Markdown links
    markdown_links = re.findall(
        r"\[([^\]]+)\]\((https?://[^\s\)]+)\)",
        text,
    )

    for title, url in markdown_links:
        results.append({
            "title": normalize_text(title),
            "url": url.strip(),
            "description": text[:3000],
        })

    # Bare URLs
    bare_urls = re.findall(
        r'https?://[^\s\)\]>"\'`]+',
        text,
    )

    existing_urls = {
        x["url"]
        for x in results
    }

    for url in bare_urls:
        url = url.rstrip(".,;")

        if url in existing_urls:
            continue

        results.append({
            "title": urlparse(url).netloc,
            "url": url,
            "description": text[:3000],
        })

    # Deduplicate
    unique = {}

    for item in results:
        unique[item["url"]] = item

    return list(unique.values())[:MAX_RESULTS_PER_WEB_TASK]


def collect_web_candidates(seen):
    candidates = []

    print(
        "\n================ GEMINI WEB SEARCH ================\n"
    )

    for task in WEB_SEARCH_TASKS[:MAX_WEB_SEARCH_TASKS]:

        if GEMINI_QUOTA_EXHAUSTED:
            print(
                "🛑 Gemini quota exhausted. "
                "Skipping remaining web searches."
            )
            break

        raw_results = gemini_web_search(task)

        for result in raw_results:

            title = result.get("title", "")
            description = result.get("description", "")
            url = result.get("url", "")

            # Local filtering
            if not local_candidate_filter(
                title,
                description,
            ):
                continue

            uid = make_uid(
                title,
                url,
                "gemini-web",
            )

            if uid in seen:
                continue

            result["uid"] = uid
            result["source"] = "Gemini Web Search"
            result["type"] = "web"

            candidates.append(result)

        print(
            f"   ✓ {len(raw_results)} search results received"
        )

    # Dedup
    unique = {}

    for item in candidates:
        unique[item["uid"]] = item

    candidates = list(unique.values())

    print(
        f"\n📊 Web candidates after filtering: "
        f"{len(candidates)}"
    )

    return candidates[:MAX_WEB_CANDIDATES_PER_RUN]


# ==============================================================================
# BATCH EVALUATION
# ==============================================================================

EVALUATION_SYSTEM = """
You are evaluating PhD opportunities for a candidate whose target research is:

- Visual-Inertial Odometry (VIO)
- Visual SLAM
- SLAM
- Visual navigation
- Robot localization
- State estimation
- Sensor fusion
- GNSS / GNSS-INS
- Camera-IMU-LiDAR fusion
- Robotics perception
- Autonomous navigation
- 3D vision for robotics

Target degree:
DIRECT FUNDED PHD.

Reject:
- Master's-only positions
- Postdocs
- internships
- undergraduate positions
- jobs without PhD enrollment
- unrelated biology
- medicine
- chemistry
- materials
- mechanical design
- manufacturing
- CFD
- unrelated social sciences

Return JSON only.

For each candidate return:

{
  "results": [
    {
      "index": 0,
      "relevant": true,
      "score": 0-100,
      "reason": "short reason"
    }
  ]
}

Scoring:
90-100 = direct VIO/SLAM/navigation/localization match
75-89  = strong robotics/perception/sensor-fusion match
60-74  = adjacent but potentially useful
0-59   = weak or irrelevant

Only mark relevant=true if it is realistically useful for this PhD target.
"""


def evaluate_batch(batch):
    if GEMINI_QUOTA_EXHAUSTED:
        return []

    items_text = []

    for i, item in enumerate(batch):

        items_text.append(
            f"""
INDEX: {i}
TITLE: {item.get('title', '')}
SOURCE: {item.get('source', '')}
URL: {item.get('url', '')}
DESCRIPTION:
{item.get('description', '')[:5000]}
"""
        )

    prompt = (
        EVALUATION_SYSTEM
        + "\n\nCANDIDATES:\n"
        + "\n---\n".join(items_text)
    )

    # IMPORTANT:
    # No temperature for Gemini 3.8 Flash.
    config = types.GenerateContentConfig()

    response = gemini_generate(
        model=GEMINI_EVAL_MODEL,
        contents=prompt,
        config=config,
    )

    if response is None:
        return []

    text = getattr(response, "text", "")

    if not text:
        return []

    # Extract JSON
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        print(
            "   ⚠️ Could not parse Gemini evaluation JSON."
        )
        return []

    try:
        data = json.loads(match.group(0))

    except Exception as e:
        print(
            f"   ⚠️ JSON parse failed: {e}"
        )
        return []

    results = data.get("results", [])

    if not isinstance(results, list):
        return []

    return results


def evaluate_candidates(candidates):
    if not candidates:
        return []

    print(
        "\n================ GEMINI EVALUATION ================\n"
    )

    evaluated = []

    for start in range(
        0,
        len(candidates),
        GEMINI_BATCH_SIZE,
    ):

        if GEMINI_QUOTA_EXHAUSTED:
            print(
                "🛑 Gemini quota exhausted. "
                "Stopping evaluation."
            )
            break

        batch = candidates[
            start:start + GEMINI_BATCH_SIZE
        ]

        print(
            f"🧠 Evaluating batch "
            f"{start + 1}-{start + len(batch)} "
            f"of {len(candidates)}"
        )

        results = evaluate_batch(batch)

        if not results:
            continue

        for result in results:

            try:
                index = int(result.get("index"))

            except Exception:
                continue

            if index < 0 or index >= len(batch):
                continue

            candidate = batch[index].copy()

            candidate["relevant"] = bool(
                result.get("relevant", False)
            )

            candidate["score"] = int(
                result.get("score", 0)
            )

            candidate["reason"] = (
                result.get("reason", "")
            )

            evaluated.append(candidate)

        time.sleep(1)

    return evaluated


# ==============================================================================
# TELEGRAM NOTIFICATION
# ==============================================================================

def telegram_send_message(bot_token, chat_id, text):
    url = (
        f"https://api.telegram.org/bot"
        f"{bot_token}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            },
            timeout=20,
        )

        return response.ok

    except Exception as e:
        print(
            f"⚠️ Telegram notification failed: {e}"
        )

        return False


def format_candidate(item):
    title = item.get("title", "Untitled")

    score = item.get("score", 0)

    reason = item.get(
        "reason",
        "Relevant PhD opportunity."
    )

    source = item.get(
        "source",
        "Unknown source"
    )

    url = item.get(
        "url",
        ""
    )

    return (
        f"🎓 {title}\n\n"
        f"⭐ Score: {score}/100\n"
        f"📌 Source: {source}\n"
        f"🧠 {reason}\n\n"
        f"🔗 {url}"
    )


def send_notifications(results):
    bot_token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if not bot_token or not chat_id:
        print(
            "⚠️ Telegram bot credentials not configured."
        )
        return

    relevant = [
        x for x in results
        if x.get("relevant")
    ]

    relevant.sort(
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    for item in relevant:

        text = format_candidate(item)

        telegram_send_message(
            bot_token,
            chat_id,
            text,
        )

        time.sleep(0.5)


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print(
        """
====================================================

      PhD Opportunity Finder

      VIO / SLAM / Navigation / GNSS / Robotics

====================================================
"""
    )

    print(
        f"Search model: {GEMINI_SEARCH_MODEL}"
    )

    print(
        f"Evaluation model: {GEMINI_EVAL_MODEL}"
    )

    print(
        f"Web search tasks: {MAX_WEB_SEARCH_TASKS}"
    )

    print(
        f"Gemini batch size: {GEMINI_BATCH_SIZE}"
    )

    # --------------------------------------------------
    # Load seen
    # --------------------------------------------------

    seen = load_seen()

    print(
        f"📚 Seen DB: {len(seen)} entries"
    )

    # --------------------------------------------------
    # IMPORTANT:
    # Basic API test BEFORE scraping/searching
    # --------------------------------------------------

    if not test_gemini_api():

        print(
            "\n❌ Stopping before the main workflow."
        )

        print(
            "   Fix Gemini API quota/billing first."
        )

        return

    # --------------------------------------------------
    # Telegram
    # --------------------------------------------------

    telegram_candidates = collect_telegram_candidates(
        seen
    )

    print(
        f"\n📊 Telegram candidates after filtering: "
        f"{len(telegram_candidates)}"
    )

    # --------------------------------------------------
    # Web
    # --------------------------------------------------

    web_candidates = collect_web_candidates(
        seen
    )

    # --------------------------------------------------
    # Merge
    # --------------------------------------------------

    all_candidates = (
        telegram_candidates
        + web_candidates
    )

    unique = {}

    for item in all_candidates:

        url = item.get("url", "")

        if not url:
            continue

        # URL-level deduplication
        normalized_url = url.split("#")[0].rstrip("/")

        if normalized_url not in unique:
            unique[normalized_url] = item

    all_candidates = list(
        unique.values()
    )

    print(
        "\n===================================================="
    )

    print(
        f"Telegram candidates: "
        f"{len(telegram_candidates)}"
    )

    print(
        f"Web candidates: "
        f"{len(web_candidates)}"
    )

    print(
        f"Unique new candidates: "
        f"{len(all_candidates)}"
    )

    print(
        "===================================================="
    )

    if not all_candidates:

        print(
            "\n⚠️ No candidates evaluated."
        )

        save_seen(seen)

        if GEMINI_QUOTA_EXHAUSTED:
            print(
                "\n🛑 Run ended because Gemini quota "
                "was exhausted."
            )

        return

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    evaluated = evaluate_candidates(
        all_candidates
    )

    # --------------------------------------------------
    # Only mark successfully evaluated candidates
    # as seen.
    # --------------------------------------------------

    for item in evaluated:
        seen.add(
            item["uid"]
        )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    relevant = [
        x for x in evaluated
        if x.get("relevant")
    ]

    relevant.sort(
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    print(
        f"\n✅ Evaluated: {len(evaluated)}"
    )

    print(
        f"🎯 Relevant: {len(relevant)}"
    )

    if relevant:

        print(
            "\n================ TOP RESULTS ================\n"
        )

        for item in relevant[:20]:

            print(
                f"⭐ {item.get('score', 0)}/100 | "
                f"{item.get('title', '')}"
            )

            print(
                f"   {item.get('url', '')}"
            )

            print(
                f"   {item.get('reason', '')}"
            )

            print()

    # --------------------------------------------------
    # Notifications
    # --------------------------------------------------

    send_notifications(
        relevant
    )

    # --------------------------------------------------
    # Save DB
    # --------------------------------------------------

    save_seen(seen)

    if GEMINI_QUOTA_EXHAUSTED:

        print(
            "\n🛑 Run ended because Gemini quota "
            "was exhausted."
        )

    else:

        print(
            "\n✅ Finished successfully."
        )


if __name__ == "__main__":
    main()
