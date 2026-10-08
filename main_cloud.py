import os
import re
import json
import time
import html
import hashlib
import warnings
from datetime import datetime, timezone, timedelta

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG
# ==============================================================================

load_dotenv()

BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_SEARCH_MODEL = os.getenv(
    "GEMINI_SEARCH_MODEL",
    "gemini-3.8-flash"
)

GEMINI_EVAL_MODEL = os.getenv(
    "GEMINI_EVAL_MODEL",
    "gemini-3.8-flash"
)

DATA_DIR = "data"
SEEN_FILE = os.path.join(DATA_DIR, "seen_posts.json")

MAX_POST_AGE_DAYS = 60

# Keep 5 web search tasks as requested.
MAX_WEB_SEARCH_TASKS = 5
MAX_RESULTS_PER_WEB_TASK = 6

# Gemini batch settings.
GEMINI_BATCH_SIZE = 15

# Safety limits.
MAX_TELEGRAM_CANDIDATES_PER_RUN = 40
MAX_WEB_CANDIDATES_PER_RUN = 30

REQUEST_DELAY = 1.0

TELEGRAM_MIN_CONFIDENCE = 0.72
SEND_TIER_2 = True

TELEGRAM_PAGES = 5

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


# ==============================================================================
# GEMINI STATE
# ==============================================================================

GEMINI_QUOTA_EXHAUSTED = False

try:
    from google import genai
    from google.genai import types

    if GEMINI_API_KEY:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    else:
        ai_client = None

except Exception as e:
    print(f"⚠️ Could not import google-genai: {e}")
    ai_client = None


# ==============================================================================
# USER PROFILE / EVALUATION PROMPT
# ==============================================================================

SYSTEM_PROMPT = """
You are evaluating PhD opportunities for a candidate applying for Fall 2027.

Candidate research profile:

- Electrical Engineering / robotics / localization
- VIO: Visual-Inertial Odometry
- Visual SLAM
- Visual Navigation
- Visual Localization
- Sensor Fusion
- GNSS / GNSS positioning
- GNSS/INS
- Inertial Navigation
- Robot Localization
- State Estimation
- Autonomous Navigation
- Factor Graphs
- LiDAR-Camera-IMU fusion
- Robotics perception
- Computer Vision
- 3D vision
- Deep Learning / Transformers
- Uncertainty-aware learning
- Multi-task learning
- Neural networks for localization/navigation

Current research includes:
- GNSS pseudorange correction
- GNSSFormer / Transformer-based models
- learned weighting
- uncertainty-aware / multi-task learning
- positioning and localization

Target:
Direct funded PhD positions for Fall 2027 or positions starting during 2027.

IMPORTANT:
Do not reject a position simply because it is not explicitly called VIO.
Strong adjacent areas such as SLAM, visual localization, sensor fusion,
robot localization, state estimation, GNSS/INS, autonomous navigation,
robot perception, LiDAR-camera fusion, and 3D vision are relevant.

TIER 1:
Direct match:
VIO, visual-inertial navigation, visual SLAM, visual odometry,
visual localization, GNSS/INS, GNSS positioning, sensor fusion,
robot localization, autonomous navigation, state estimation,
factor graphs, LiDAR-camera-IMU fusion, robotics localization.

TIER 2:
Strong adjacent match:
computer vision, 3D vision, robotics perception, LiDAR,
robot perception, pose estimation, trajectory estimation,
autonomous vehicles, field robotics, probabilistic robotics,
deep learning for robotics, self-supervised perception,
uncertainty estimation, neural localization.

TIER 3:
Technically interesting but substantially farther away.

TIER 4:
Weak or irrelevant.

REJECT:
biology, molecular biology, chemistry, materials science,
biomedical wet lab, prosthetics, biomechanics, mechanical design,
manufacturing, CFD/fluid dynamics, social science, economics,
policy, management, generic business/data science,
generic AI without a robotics/navigation/computer vision connection,
Master-only opportunities, undergraduate opportunities,
postdoctoral positions, internships.

For every candidate determine:
- is_relevant
- tier
- confidence_score from 0 to 1
- key_topics
- research_fit
- technical_gaps
- reason
"""


# ==============================================================================
# LOCAL FILTERS
# ==============================================================================

PHD_PATTERNS = [
    r"\bph\.?d\b",
    r"\bphd\b",
    r"\bdoctoral\b",
    r"\bdoctorate\b",
    r"دکتری",
    r"دکترا",
]

MASTER_ONLY_PATTERNS = [
    r"\bmaster'?s?\b",
    r"\bmsc\b",
    r"\bma\b",
    r"master degree",
    r"کارشناسی ارشد",
    r"ارشد",
]

STRONG_RESEARCH_PATTERNS = [
    r"\bvios?\b",
    r"\bvisual[- ]inertial\b",
    r"\bvisual odometry\b",
    r"\bvisual slam\b",
    r"\bslam\b",
    r"\bvisual localization\b",
    r"\blocalization\b",
    r"\bnavigation\b",
    r"\bsensor fusion\b",
    r"\bgnss\b",
    r"\bgps\b",
    r"\bins\b",
    r"\binertial navigation\b",
    r"\bstate estimation\b",
    r"\brobot localization\b",
    r"\brobot navigation\b",
    r"\brobotics\b",
    r"\bcomputer vision\b",
    r"\b3d vision\b",
    r"\blidar\b",
    r"\bperception\b",
    r"\bfactor graph\b",
    r"\bautonomous vehicle\b",
    r"\bautonomous navigation\b",
    r"\bpose estimation\b",
    r"\btrajectory estimation\b",
    r"\bcamera[- ]imu\b",
]

ADJACENT_PATTERNS = [
    r"\bdeep learning\b",
    r"\bmachine learning\b",
    r"\btransformer\b",
    r"\bneural network\b",
    r"\bself[- ]supervised\b",
    r"\b3d reconstruction\b",
    r"\bpoint cloud\b",
    r"\brobot perception\b",
    r"\bautonomous systems\b",
    r"\brobot perception\b",
]

HARD_EXCLUDE_PATTERNS = [
    r"\bpostdoc\b",
    r"\bpostdoctoral\b",
    r"\bundergraduate\b",
    r"\bbachelor'?s?\b",
    r"\bbiology\b",
    r"\bmolecular\b",
    r"\bgenomics\b",
    r"\bcell culture\b",
    r"\bchemistry\b",
    r"\bmaterials science\b",
    r"\bprosthetic\b",
    r"\borthopedic\b",
    r"\bbiomechanical\b",
    r"\bbiomechanics\b",
    r"\bmechanical design\b",
    r"\bmanufacturing\b",
    r"\bcfd\b",
    r"\bfluid dynamics\b",
    r"\bpanel discussion\b",
    r"\bapplication fee\b",
]


def normalize_whitespace(text):
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def text_matches_research(text):
    """
    Fast local pre-filter.

    This should be intentionally strict because anything passing this
    function may later consume Gemini API quota.
    """

    text = normalize_whitespace(text).lower()

    if not text:
        return False

    # --------------------------------------------------------------
    # Hard exclusions
    # --------------------------------------------------------------

    for pattern in HARD_EXCLUDE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    # --------------------------------------------------------------
    # Must explicitly look like a PhD / doctoral opportunity.
    # --------------------------------------------------------------

    has_phd = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in PHD_PATTERNS
    )

    if not has_phd:
        return False

    # --------------------------------------------------------------
    # Research relevance.
    # --------------------------------------------------------------

    strong_matches = sum(
        1
        for pattern in STRONG_RESEARCH_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    )

    adjacent_matches = sum(
        1
        for pattern in ADJACENT_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    )

    # At least one strong target term.
    #
    # Generic AI / ML alone should NOT pass.
    if strong_matches >= 1:
        return True

    # Two adjacent terms can also pass.
    if adjacent_matches >= 2:
        return True

    return False


# ==============================================================================
# URL / UID
# ==============================================================================

def canonicalize_url(url):
    if not url:
        return ""

    url = html.unescape(str(url)).strip()

    # Remove common tracking parameters.
    url = re.sub(
        r"([?&])(utm_[^=&]+|fbclid|gclid)=[^&]*",
        "",
        url,
        flags=re.IGNORECASE,
    )

    url = re.sub(r"[?&]+$", "", url)

    return url


def make_uid(position):
    url = canonicalize_url(position.get("url", ""))

    title = normalize_whitespace(
        position.get("title", "")
    ).lower()

    source = normalize_whitespace(
        position.get("source", "")
    ).lower()

    raw = f"{url}|{title}|{source}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# ==============================================================================
# DATE
# ==============================================================================

def is_recent(dt_value, max_days=MAX_POST_AGE_DAYS):
    if not dt_value:
        return True

    try:
        if isinstance(dt_value, datetime):
            dt = dt_value
        else:
            dt = datetime.fromisoformat(
                str(dt_value).replace("Z", "+00:00")
            )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        return (now - dt) <= timedelta(days=max_days)

    except Exception:
        return True


# ==============================================================================
# SEEN DATABASE
# ==============================================================================

def load_seen():
    os.makedirs(DATA_DIR, exist_ok=True)

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
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(
                sorted(list(seen)),
                f,
                ensure_ascii=False,
                indent=2,
            )

    except Exception as e:
        print(f"❌ Could not save seen DB: {e}")


def mark_seen(position, seen):
    seen.add(make_uid(position))


# ==============================================================================
# JSON EXTRACTION
# ==============================================================================

def extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # First attempt: direct JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting object.
    obj_match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    # Try extracting array.
    arr_match = re.search(
        r"\[.*\]",
        text,
        flags=re.DOTALL,
    )

    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass

    return None


# ==============================================================================
# GEMINI REQUEST HELPER
# ==============================================================================

def gemini_generate(
    model,
    contents,
    config=None,
    retries=2,
):
    global GEMINI_QUOTA_EXHAUSTED

    if GEMINI_QUOTA_EXHAUSTED:
        return None

    if ai_client is None:
        print("❌ Gemini client is not configured.")
        return None

    for attempt in range(retries + 1):

        try:
            response = ai_client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            return response

        except Exception as e:

            error_text = str(e).lower()

            # ----------------------------------------------------------
            # Quota: DO NOT RETRY
            # ----------------------------------------------------------

            if (
                "429" in error_text
                or "resource_exhausted" in error_text
                or "quota" in error_text
            ):
                GEMINI_QUOTA_EXHAUSTED = True

                print(
                    "\n"
                    "🛑 Gemini quota exhausted.\n"
                    "Stopping all further Gemini calls for this run.\n"
                )

                return None

            # ----------------------------------------------------------
            # Temporary overload: retry.
            # ----------------------------------------------------------

            if (
                "503" in error_text
                or "unavailable" in error_text
                or "high demand" in error_text
            ):
                if attempt < retries:
                    wait_time = 2 ** (attempt + 1)

                    print(
                        f"⚠️ Gemini temporarily unavailable "
                        f"(attempt {attempt + 1}/{retries + 1}). "
                        f"Retrying in {wait_time}s..."
                    )

                    time.sleep(wait_time)
                    continue

            print(f"❌ Gemini error: {e}")
            return None

    return None


# ==============================================================================
# GEMINI WEB SEARCH
# ==============================================================================

def gemini_web_search(task_name, search_query):
    global GEMINI_QUOTA_EXHAUSTED

    if GEMINI_QUOTA_EXHAUSTED:
        return []

    if ai_client is None:
        return []

    print(
        f"\n🌐 Gemini Web Search: {task_name}"
    )

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1,
    )

    prompt = f"""
Find current funded PhD opportunities relevant to this candidate.

SEARCH TASK:
{task_name}

SEARCH QUERY:
{search_query}

Candidate target:
- Direct funded PhD
- Fall 2027 or starting during 2027
- VIO
- visual SLAM
- visual odometry
- visual navigation
- visual localization
- sensor fusion
- GNSS / GNSS-INS
- inertial navigation
- robot localization
- state estimation
- autonomous navigation
- robotics perception
- LiDAR / camera / IMU fusion
- 3D computer vision

Search the live web.

Prioritize:
1. Official university PhD/project pages
2. FindAPhD
3. EURAXESS
4. Official research group / supervisor pages

Avoid:
- Master programs
- Bachelor's programs
- Postdocs
- internships
- jobs without PhD enrollment
- clearly unrelated fields

Return ONLY valid JSON:

{{
  "results": [
    {{
      "title": "...",
      "university": "...",
      "country": "...",
      "supervisor": "...",
      "deadline": "...",
      "funding": "...",
      "source": "...",
      "url": "...",
      "description": "...",
      "relevance": "..."
    }}
  ]
}}

Maximum {MAX_RESULTS_PER_WEB_TASK} results.
Use the actual URL of each opportunity.
"""

    response = gemini_generate(
        model=GEMINI_SEARCH_MODEL,
        contents=prompt,
        config=config,
        retries=2,
    )

    if response is None:
        return []

    try:
        raw_text = response.text
    except Exception:
        return []

    data = extract_json(raw_text)

    if not isinstance(data, dict):
        print("⚠️ Gemini search returned invalid JSON.")
        return []

    results = data.get("results", [])

    if not isinstance(results, list):
        return []

    cleaned = []

    for item in results[:MAX_RESULTS_PER_WEB_TASK]:

        if not isinstance(item, dict):
            continue

        title = normalize_whitespace(
            item.get("title", "")
        )

        url = canonicalize_url(
            item.get("url", "")
        )

        if not title:
            continue

        cleaned.append({
            "title": title,
            "university": item.get("university", ""),
            "country": item.get("country", ""),
            "supervisor": item.get("supervisor", ""),
            "deadline": item.get("deadline", ""),
            "funding": item.get("funding", ""),
            "source": item.get("source", "Gemini Web Search"),
            "url": url,
            "description": normalize_whitespace(
                item.get("description", "")
            ),
            "research_text": normalize_whitespace(
                item.get("relevance", "")
            ),
        })

    print(
        f"   ✓ Found {len(cleaned)} candidates"
    )

    return cleaned


# ==============================================================================
# BATCH EVALUATION
# ==============================================================================

def build_batch_prompt(candidates):
    blocks = []

    for idx, candidate in enumerate(candidates, start=1):

        text = f"""
CANDIDATE {idx}

Title:
{candidate.get("title", "")}

University:
{candidate.get("university", "")}

Country:
{candidate.get("country", "")}

Supervisor:
{candidate.get("supervisor", "")}

Deadline:
{candidate.get("deadline", "")}

Funding:
{candidate.get("funding", "")}

Source:
{candidate.get("source", "")}

URL:
{candidate.get("url", "")}

Description:
{candidate.get("description", "")}

Research text:
{candidate.get("research_text", "")}

Original Telegram text:
{candidate.get("text", "")}
"""

        blocks.append(text)

    joined = "\n\n".join(blocks)

    return f"""
{SYSTEM_PROMPT}

Evaluate ALL candidates below.

{joined}

Return ONLY valid JSON in this exact structure:

{{
  "results": [
    {{
      "id": 1,
      "is_relevant": true,
      "tier": 1,
      "confidence_score": 0.95,
      "key_topics": ["VIO", "sensor fusion"],
      "research_fit": "...",
      "technical_gaps": ["..."],
      "reason": "..."
    }}
  ]
}}

Rules:
- id must correspond exactly to the candidate number.
- confidence_score must be between 0 and 1.
- tier must be 1, 2, 3, or 4.
- Do not invent information.
- Be conservative.
- Return one result for every candidate.
"""


def evaluate_batch_with_gemini(candidates):
    global GEMINI_QUOTA_EXHAUSTED

    if not candidates:
        return []

    if GEMINI_QUOTA_EXHAUSTED:
        return []

    all_results = []

    for start in range(
        0,
        len(candidates),
        GEMINI_BATCH_SIZE,
    ):

        batch = candidates[
            start:start + GEMINI_BATCH_SIZE
        ]

        print(
            f"\n🧠 Gemini batch evaluation: "
            f"{start + 1}-{start + len(batch)} "
            f"of {len(candidates)}"
        )

        prompt = build_batch_prompt(batch)

        config = types.GenerateContentConfig(
            temperature=0.1,
        )

        response = gemini_generate(
            model=GEMINI_EVAL_MODEL,
            contents=prompt,
            config=config,
            retries=2,
        )

        if response is None:
            print(
                "⚠️ Batch evaluation failed. "
                "Candidates in this batch will NOT be marked seen."
            )
            continue

        try:
            raw_text = response.text
        except Exception:
            print("⚠️ Could not read Gemini response.")
            continue

        data = extract_json(raw_text)

        if not isinstance(data, dict):
            print("⚠️ Invalid JSON from batch evaluator.")
            continue

        results = data.get("results", [])

        if not isinstance(results, list):
            continue

        for result in results:

            if not isinstance(result, dict):
                continue

            try:
                idx = int(result.get("id"))
            except Exception:
                continue

            if idx < 1 or idx > len(batch):
                continue

            candidate = batch[idx - 1]

            result["_candidate"] = candidate
            result["_api_ok"] = True

            all_results.append(result)

        if not GEMINI_QUOTA_EXHAUSTED:
            time.sleep(REQUEST_DELAY)

    return all_results


# ==============================================================================
# TELEGRAM
# ==============================================================================

def extract_telegram_messages(html_text, channel):
    soup = BeautifulSoup(
        html_text,
        "html.parser",
    )

    messages = []

    for message in soup.select(".tgme_widget_message"):

        try:
            text_node = message.select_one(
                ".tgme_widget_message_text"
            )

            text = (
                text_node.get_text(
                    " ",
                    strip=True,
                )
                if text_node
                else ""
            )

            if not text:
                continue

            date_node = message.select_one(
                "time"
            )

            date_value = None

            if date_node:
                date_value = date_node.get(
                    "datetime"
                )

            link_node = message.select_one(
                ".tgme_widget_message_date"
            )

            url = ""

            if link_node:
                url = link_node.get("href", "")

            messages.append({
                "title": text[:180],
                "text": text,
                "url": url,
                "source": f"Telegram @{channel}",
                "date": date_value,
                "university": "",
                "country": "",
                "supervisor": "",
                "deadline": "",
                "funding": "",
                "description": text,
                "research_text": text,
            })

        except Exception:
            continue

    return messages


def scrape_telegram_channel(channel):
    all_messages = []

    for page in range(1, TELEGRAM_PAGES + 1):

        url = (
            f"https://t.me/s/{channel}"
            f"?before={page * 20}"
        )

        try:
            response = requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "Chrome/120 Safari/537.36"
                    )
                },
            )

            if response.status_code != 200:
                print(
                    f"⚠️ Telegram @{channel} "
                    f"page {page}: HTTP "
                    f"{response.status_code}"
                )
                continue

            messages = extract_telegram_messages(
                response.text,
                channel,
            )

            all_messages.extend(messages)

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(
                f"⚠️ Telegram @{channel}: {e}"
            )

    return all_messages


def scrape_telegram():
    candidates = []

    print("\n================ TELEGRAM ================\n")

    for channel in TELEGRAM_CHANNELS:

        print(
            f"📡 Scanning @{channel}..."
        )

        messages = scrape_telegram_channel(
            channel
        )

        accepted = 0

        for message in messages:

            # Date filter.
            if not is_recent(
                message.get("date")
            ):
                continue

            # Strong local filter.
            if not text_matches_research(
                message.get("text", "")
            ):
                continue

            candidates.append(message)
            accepted += 1

        print(
            f"   ✓ {accepted} relevant local candidates"
        )

    # Deduplicate.
    unique = {}
    for candidate in candidates:
        unique[make_uid(candidate)] = candidate

    candidates = list(unique.values())

    if len(candidates) > MAX_TELEGRAM_CANDIDATES_PER_RUN:
        candidates = candidates[
            :MAX_TELEGRAM_CANDIDATES_PER_RUN
        ]

    print(
        f"\n📊 Telegram candidates after filtering: "
        f"{len(candidates)}"
    )

    return candidates


# ==============================================================================
# WEB SEARCH
# ==============================================================================

WEB_SEARCH_TASKS = [
    (
        "Core VIO / SLAM",
        (
            "funded PhD positions 2026 2027 "
            "visual inertial odometry OR visual SLAM "
            "OR visual odometry OR visual localization "
            "OR camera IMU OR VIO"
        ),
    ),

    (
        "Navigation / Localization",
        (
            "funded PhD positions 2026 2027 "
            "robot localization OR robot navigation "
            "OR sensor fusion OR state estimation "
            "OR GNSS INS OR inertial navigation "
            "OR autonomous navigation"
        ),
    ),

    (
        "Robotics / Perception",
        (
            "funded PhD positions 2026 2027 "
            "robotics computer vision 3D perception "
            "LiDAR camera fusion autonomous vehicles "
            "field robotics pose estimation"
        ),
    ),

    (
        "GNSS / Sensor Fusion",
        (
            "funded PhD 2027 GNSS positioning "
            "GNSS INS sensor fusion localization "
            "inertial navigation robotics"
        ),
    ),

    (
        "Visual Navigation / Autonomous Systems",
        (
            "funded PhD 2027 visual navigation "
            "visual localization autonomous robots "
            "SLAM perception computer vision "
            "robot navigation state estimation"
        ),
    ),
]


def scrape_with_gemini():
    candidates = []

    print(
        "\n================ GEMINI WEB SEARCH ================\n"
    )

    for task_name, query in WEB_SEARCH_TASKS[
        :MAX_WEB_SEARCH_TASKS
    ]:

        if GEMINI_QUOTA_EXHAUSTED:
            print(
                "🛑 Gemini quota exhausted. "
                "Skipping remaining web searches."
            )
            break

        results = gemini_web_search(
            task_name,
            query,
        )

        for result in results:

            combined_text = " ".join([
                result.get("title", ""),
                result.get("description", ""),
                result.get("research_text", ""),
                result.get("university", ""),
            ])

            # Local filtering.
            if not text_matches_research(
                combined_text
            ):
                continue

            candidates.append(result)

        time.sleep(REQUEST_DELAY)

    # Deduplicate.
    unique = {}

    for candidate in candidates:

        uid = make_uid(candidate)

        if uid not in unique:
            unique[uid] = candidate

    candidates = list(unique.values())

    if len(candidates) > MAX_WEB_CANDIDATES_PER_RUN:
        candidates = candidates[
            :MAX_WEB_CANDIDATES_PER_RUN
        ]

    print(
        f"\n📊 Web candidates after filtering: "
        f"{len(candidates)}"
    )

    return candidates


# ==============================================================================
# TELEGRAM NOTIFICATION
# ==============================================================================

def send_telegram_message(message):
    if not BOT_TOKEN or not CHAT_ID:
        print(
            "⚠️ Telegram bot credentials missing."
        )
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=20,
        )

        if response.status_code != 200:
            print(
                f"❌ Telegram send failed: "
                f"{response.status_code} "
                f"{response.text[:300]}"
            )
            return False

        return True

    except Exception as e:
        print(
            f"❌ Telegram send exception: {e}"
        )
        return False


def format_position_alert(
    candidate,
    evaluation,
):
    title = html.escape(
        str(candidate.get("title", "Untitled"))
    )

    university = html.escape(
        str(candidate.get("university", "") or "")
    )

    country = html.escape(
        str(candidate.get("country", "") or "")
    )

    supervisor = html.escape(
        str(candidate.get("supervisor", "") or "")
    )

    deadline = html.escape(
        str(candidate.get("deadline", "") or "")
    )

    funding = html.escape(
        str(candidate.get("funding", "") or "")
    )

    source = html.escape(
        str(candidate.get("source", "") or "")
    )

    url = candidate.get("url", "")

    tier = evaluation.get(
        "tier",
        4,
    )

    confidence = evaluation.get(
        "confidence_score",
        0,
    )

    topics = evaluation.get(
        "key_topics",
        [],
    )

    reason = evaluation.get(
        "reason",
        "",
    )

    research_fit = evaluation.get(
        "research_fit",
        "",
    )

    gaps = evaluation.get(
        "technical_gaps",
        [],
    )

    topic_text = ", ".join(
        str(x)
        for x in topics
    )

    gap_text = ", ".join(
        str(x)
        for x in gaps
    )

    parts = [
        f"🎓 <b>PhD Opportunity — Tier {tier}</b>",
        "",
        f"<b>{title}</b>",
    ]

    if university:
        parts.append(
            f"🏫 {university}"
        )

    if country:
        parts.append(
            f"🌍 {country}"
        )

    if supervisor:
        parts.append(
            f"👤 {supervisor}"
        )

    if deadline:
        parts.append(
            f"⏰ Deadline: {deadline}"
        )

    if funding:
        parts.append(
            f"💰 Funding: {funding}"
        )

    parts.extend([
        "",
        f"🎯 Confidence: {confidence:.2f}",
        f"🔬 Topics: {html.escape(topic_text)}",
        "",
        f"<b>Why it fits:</b> "
        f"{html.escape(str(reason))}",
    ])

    if research_fit:
        parts.append(
            f"\n<b>Research fit:</b> "
            f"{html.escape(str(research_fit))}"
        )

    if gap_text:
        parts.append(
            f"\n<b>Potential gaps:</b> "
            f"{html.escape(gap_text)}"
        )

    if source:
        parts.append(
            f"\n📡 Source: {source}"
        )

    if url:
        parts.append(
            f'\n🔗 <a href="{html.escape(url)}">'
            f"Open opportunity</a>"
        )

    return "\n".join(parts)


# ==============================================================================
# PROCESS EVALUATIONS
# ==============================================================================

def process_evaluation_results(
    evaluations,
    seen,
):
    sent = 0
    rejected = 0
    marked = 0

    for evaluation in evaluations:

        candidate = evaluation.get(
            "_candidate"
        )

        if not candidate:
            continue

        # Successful API response.
        if not evaluation.get(
            "_api_ok",
            False,
        ):
            continue

        uid = make_uid(candidate)

        if uid in seen:
            continue

        try:
            tier = int(
                evaluation.get(
                    "tier",
                    4,
                )
            )
        except Exception:
            tier = 4

        try:
            confidence = float(
                evaluation.get(
                    "confidence_score",
                    0,
                )
            )
        except Exception:
            confidence = 0

        is_relevant = bool(
            evaluation.get(
                "is_relevant",
                False,
            )
        )

        # Mark as seen ONLY after successful evaluation.
        seen.add(uid)
        marked += 1

        if not is_relevant:
            rejected += 1
            continue

        if tier > 2:
            rejected += 1
            continue

        if confidence < TELEGRAM_MIN_CONFIDENCE:
            rejected += 1
            continue

        if tier == 2 and not SEND_TIER_2:
            rejected += 1
            continue

        message = format_position_alert(
            candidate,
            evaluation,
        )

        if send_telegram_message(message):
            sent += 1

            print(
                f"   🚨 SENT Tier {tier}: "
                f"{candidate.get('title', '')[:100]}"
            )

            time.sleep(
                REQUEST_DELAY
            )

    return sent, rejected, marked


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print(
        "\n"
        "====================================================\n"
        "      PhD Opportunity Finder\n"
        "      VIO / SLAM / Navigation / GNSS / Robotics\n"
        "====================================================\n"
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

    if not GEMINI_API_KEY:
        print(
            "❌ GEMINI_API_KEY is missing."
        )
        return

    if ai_client is None:
        print(
            "❌ Gemini client is unavailable."
        )
        return

    seen = load_seen()

    print(
        f"📚 Seen DB: {len(seen)} entries"
    )

    # ==================================================================
    # TELEGRAM
    # ==================================================================

    telegram_candidates = scrape_telegram()

    # ==================================================================
    # WEB SEARCH
    # ==================================================================

    web_candidates = []

    if not GEMINI_QUOTA_EXHAUSTED:
        web_candidates = scrape_with_gemini()

    # ==================================================================
    # MERGE / DEDUP
    # ==================================================================

    all_candidates = []

    all_candidates.extend(
        telegram_candidates
    )

    all_candidates.extend(
        web_candidates
    )

    unique = {}

    for candidate in all_candidates:

        uid = make_uid(candidate)

        if uid in seen:
            continue

        if uid not in unique:
            unique[uid] = candidate

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
        "====================================================\n"
    )

    # ==================================================================
    # BATCH GEMINI EVALUATION
    # ==================================================================

    if (
        all_candidates
        and not GEMINI_QUOTA_EXHAUSTED
    ):

        evaluations = evaluate_batch_with_gemini(
            all_candidates
        )

        sent, rejected, marked = (
            process_evaluation_results(
                evaluations,
                seen,
            )
        )

        print(
            "\n================ SUMMARY ================\n"
        )

        print(
            f"Evaluated: {len(evaluations)}"
        )

        print(
            f"Sent: {sent}"
        )

        print(
            f"Rejected: {rejected}"
        )

        print(
            f"Marked seen: {marked}"
        )

    else:

        print(
            "⚠️ No candidates evaluated."
        )

    # ==================================================================
    # SAVE
    # ==================================================================

    save_seen(seen)

    print(
        f"\n💾 Seen DB saved: {len(seen)} entries"
    )

    if GEMINI_QUOTA_EXHAUSTED:
        print(
            "\n🛑 Run ended because Gemini quota "
            "was exhausted."
        )

    print(
        "\n✅ Finished.\n"
    )


if __name__ == "__main__":
    main()
