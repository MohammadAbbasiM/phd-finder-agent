# ==============================================================================
# PhD FINDER AGENT
# Gemini Web Search + Telegram + Relevance Evaluation
#
# Main strategy:
#   1. Scan Telegram academic channels
#   2. Ask Gemini to SEARCH THE WEB for real PhD vacancies
#   3. Prioritize FindAPhD / EURAXESS / university pages
#   4. Extract real vacancy URLs
#   5. Evaluate research fit against the user's profile
#   6. Deduplicate with seen_posts.json
#   7. Send strong matches to Telegram
#
# IMPORTANT:
#   This version does NOT directly crawl FindAPhD.
#   Gemini uses Google Search grounding instead.
#
# Required environment variables:
#   GEMINI_API_KEY
#   TG_BOT_TOKEN
#   TG_CHAT_ID
#
# Optional:
#   GEMINI_SEARCH_MODEL
#   GEMINI_EVAL_MODEL
# ==============================================================================

import os
import re
import json
import time
import html
import hashlib
import warnings
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ==============================================================================
# CONFIG
# ==============================================================================

load_dotenv()

BOT_TOKEN = (
    os.getenv("TG_BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

CHAT_ID = (
    os.getenv("TG_CHAT_ID")
    or os.getenv("TELEGRAM_CHAT_ID")
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_SEARCH_MODEL = os.getenv(
    "GEMINI_SEARCH_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_EVAL_MODEL = os.getenv(
    "GEMINI_EVAL_MODEL",
    "gemini-3.5-flash-lite"
)

DATA_DIR = "data"
SEEN_FILE = os.path.join(DATA_DIR, "seen_posts.json")

MAX_POST_AGE_DAYS = 60

TELEGRAM_PAGES_PER_CHANNEL = 5

# Gemini web-search discovery.
# Each task can trigger multiple Google searches internally.
MAX_WEB_SEARCH_TASKS = 18
MAX_RESULTS_PER_WEB_TASK = 8

# Small delay to avoid hammering Telegram / APIs.
REQUEST_DELAY = 0.8

# Only send positions above this relevance threshold.
TELEGRAM_MIN_CONFIDENCE = 0.72

# Whether to send Tier 2 positions.
SEND_TIER_2 = True


# ==============================================================================
# HTTP
# ==============================================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        )
    }
)


# ==============================================================================
# GEMINI
# ==============================================================================

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


if GEMINI_API_KEY and genai is not None:
    try:
        ai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:
        print(f"[!] Gemini client initialization failed: {exc}")
        ai_client = None
else:
    ai_client = None


# ==============================================================================
# TELEGRAM CHANNELS
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


# ==============================================================================
# WEB SEARCH TASKS
# ==============================================================================

WEB_SEARCH_TASKS = [
    {
        "name": "Visual-Inertial Odometry",
        "query": (
            "funded PhD position visual inertial odometry "
            "VIO visual-inertial navigation"
        ),
    },
    {
        "name": "Visual SLAM",
        "query": (
            "funded PhD position visual SLAM "
            "simultaneous localization mapping robotics"
        ),
    },
    {
        "name": "Visual Navigation",
        "query": (
            "funded PhD position visual navigation "
            "robot navigation camera localization"
        ),
    },
    {
        "name": "Sensor Fusion",
        "query": (
            "funded PhD position sensor fusion "
            "robot localization state estimation"
        ),
    },
    {
        "name": "GNSS Positioning",
        "query": (
            "funded PhD position GNSS positioning "
            "GNSS localization navigation"
        ),
    },
    {
        "name": "GNSS INS",
        "query": (
            "funded PhD position GNSS INS "
            "GNSS inertial navigation sensor fusion"
        ),
    },
    {
        "name": "Robot Localization",
        "query": (
            "funded PhD position robot localization "
            "mobile robot state estimation"
        ),
    },
    {
        "name": "Robot Navigation",
        "query": (
            "funded PhD position autonomous robot navigation "
            "robot perception localization"
        ),
    },
    {
        "name": "Computer Vision Robotics",
        "query": (
            "funded PhD position computer vision robotics "
            "3D vision robot perception"
        ),
    },
    {
        "name": "LiDAR Camera Fusion",
        "query": (
            "funded PhD position LiDAR camera fusion "
            "multi sensor perception robotics"
        ),
    },
    {
        "name": "Camera IMU Fusion",
        "query": (
            "funded PhD position camera IMU fusion "
            "visual inertial sensor fusion"
        ),
    },
    {
        "name": "State Estimation",
        "query": (
            "funded PhD position state estimation robotics "
            "factor graph optimization localization"
        ),
    },
    {
        "name": "Autonomous Navigation",
        "query": (
            "funded PhD position autonomous navigation "
            "autonomous vehicles robotics"
        ),
    },
    {
        "name": "Field Robotics",
        "query": (
            "funded PhD position field robotics "
            "outdoor robot localization navigation"
        ),
    },
    {
        "name": "Visual Localization",
        "query": (
            "funded PhD position visual localization "
            "place recognition camera localization"
        ),
    },
    {
        "name": "Pose Estimation",
        "query": (
            "funded PhD position pose estimation "
            "trajectory estimation robotics computer vision"
        ),
    },
    {
        "name": "Robotics ML",
        "query": (
            "funded PhD position machine learning robotics "
            "deep learning computer vision autonomous robots"
        ),
    },
    {
        "name": "Uncertainty Robotics",
        "query": (
            "funded PhD position uncertainty estimation robotics "
            "probabilistic localization sensor fusion"
        ),
    },
]


# ==============================================================================
# RESEARCH PROFILE
# ==============================================================================

SYSTEM_PROMPT = r"""
You are an expert PhD-position relevance evaluator.

Candidate profile:

The candidate is an Electrical/Electronics Engineering researcher
interested in:

PRIMARY TARGETS
- Visual-Inertial Odometry (VIO)
- Visual SLAM
- Visual Navigation
- Visual Localization
- Sensor Fusion
- GNSS / INS
- GNSS positioning
- Robot Localization
- Robot Navigation
- Autonomous Navigation
- State Estimation
- Factor Graph Optimization
- Computer Vision for Robotics
- LiDAR-camera-IMU fusion
- Multi-sensor localization
- Autonomous vehicles
- Field robotics

SECONDARY TARGETS
- 3D computer vision
- Robot perception
- Pose estimation
- Trajectory estimation
- Object tracking
- Place recognition
- Probabilistic robotics
- Machine learning for robotics
- Deep learning for computer vision
- Self-supervised learning for robotics
- Uncertainty estimation
- Domain adaptation
- Graph neural networks
- Embedded AI for robotics
- CUDA / GPU computer vision
- ROS / robotic systems

RELEVANT TECHNICAL BACKGROUND
- GNSS positioning
- GNSS pseudorange correction
- GNSS sensor fusion
- Deep learning
- Transformer architectures
- Multi-task learning
- Uncertainty-aware learning
- Computer vision
- Localization
- Navigation
- Python
- PyTorch
- Research-oriented engineering

TARGET
Direct funded PhD / fully funded PhD opportunities for Fall 2027
or positions that can reasonably lead to a Fall 2027 PhD start.

SCORING

Tier 1:
Directly related to:
VIO, visual SLAM, visual navigation, visual localization,
GNSS/INS, GNSS positioning, sensor fusion, robot localization,
autonomous navigation, state estimation, factor graphs,
LiDAR-camera-IMU fusion, robotics perception.

Tier 2:
Strongly adjacent:
3D vision, robotics perception, pose estimation,
trajectory estimation, autonomous vehicles,
field robotics, probabilistic robotics,
ML for robotics, uncertainty estimation,
self-supervised visual learning.

Tier 3:
Technically interesting but substantially farther away.

Tier 4:
Weak relevance.

IMPORTANT:
Do NOT reject a position merely because it does not explicitly say
"VIO" or "SLAM".

A robotics position involving localization, perception,
multi-sensor fusion, state estimation, navigation, or 3D vision
can be highly relevant.

REJECT:
- wet-lab biology
- chemistry
- materials science
- biomedical research unrelated to robotics
- pure mechanical design
- prosthetics
- social science
- policy
- economics
- pure theoretical mathematics
- positions without meaningful technical overlap
- generic master's programs
- generic job advertisements
"""


# ==============================================================================
# FILTERS
# ==============================================================================

BROAD_PATTERNS = [
    # VIO / SLAM
    r"\bvio\b",
    r"visual[- ]inertial",
    r"inertial[- ]visual",
    r"visual odometry",
    r"visual slam",
    r"\bslam\b",
    r"simultaneous localization",
    r"visual localization",
    r"visual navigation",

    # Localization / navigation
    r"robot localization",
    r"robot navigation",
    r"autonomous navigation",
    r"state estimation",
    r"pose estimation",
    r"trajectory estimation",
    r"position estimation",
    r"localization",
    r"navigation",

    # Sensor fusion
    r"sensor fusion",
    r"multi[- ]sensor",
    r"multisensor",
    r"camera[- ]imu",
    r"imu[- ]camera",
    r"lidar[- ]camera",
    r"camera[- ]lidar",
    r"gnss",
    r"gps",
    r"gnss/ins",
    r"inertial navigation",

    # Robotics / CV
    r"robotics",
    r"robot perception",
    r"computer vision",
    r"3d vision",
    r"3d computer vision",
    r"autonomous vehicle",
    r"mobile robot",
    r"field robotics",
    r"perception",

    # Estimation / optimization
    r"factor graph",
    r"graph optimization",
    r"probabilistic robotics",
    r"kalman filter",
    r"bayesian",
    r"uncertainty",

    # ML
    r"deep learning",
    r"machine learning",
    r"self[- ]supervised",
    r"transformer",
    r"graph neural",
    r"neural network",
]


EXCLUDE_PATTERNS = [
    # Biology / wet lab
    r"\bbiology\b",
    r"molecular",
    r"cell culture",
    r"genomics",
    r"proteomics",
    r"wet[- ]lab",
    r"biochemistry",

    # Chemistry / materials
    r"\bchemistry\b",
    r"chemical synthesis",
    r"polymer",
    r"materials science",

    # Mechanical / biomedical
    r"prosthetic",
    r"orthopedic",
    r"biomechanical",
    r"mechanical design",
    r"manufacturing",

    # Nontechnical
    r"social science",
    r"policy",
    r"economics",
    r"management",
]


# ==============================================================================
# UTILS
# ==============================================================================

def normalize_whitespace(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def canonicalize_url(url):
    if not url:
        return ""

    try:
        parsed = urlparse(url)

        # Remove tracking parameters.
        query_items = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            key_lower = key.lower()

            if key_lower.startswith("utm_"):
                continue

            if key_lower in {
                "fbclid",
                "gclid",
                "ref",
                "source",
                "campaign",
            }:
                continue

            query_items.append((key, value))

        parsed = parsed._replace(
            query=urlencode(query_items),
            fragment=""
        )

        return urlunparse(parsed)

    except Exception:
        return url


def make_uid(url, title=""):
    raw = (
        canonicalize_url(url).lower()
        + "|"
        + normalize_whitespace(title).lower()
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_recent(date_value, max_days=MAX_POST_AGE_DAYS):
    if not date_value:
        return True

    try:
        if isinstance(date_value, datetime):
            dt = date_value

        else:
            date_str = str(date_value)

            dt = datetime.fromisoformat(
                date_str.replace("Z", "+00:00")
            )

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - dt

        return age <= timedelta(days=max_days)

    except Exception:
        return True


def text_matches_research(text):
    text = normalize_whitespace(text).lower()

    positive = any(
        re.search(pattern, text, flags=re.I)
        for pattern in BROAD_PATTERNS
    )

    if not positive:
        return False

    excluded = any(
        re.search(pattern, text, flags=re.I)
        for pattern in EXCLUDE_PATTERNS
    )

    return not excluded


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

    except Exception as exc:
        print(f"[!] Could not load seen database: {exc}")

    return set()


def save_seen(seen):
    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(
                sorted(seen),
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as exc:
        print(f"[!] Could not save seen database: {exc}")


def mark_as_seen(seen, uid):
    seen.add(uid)


# ==============================================================================
# GEMINI JSON EXTRACTION
# ==============================================================================

def extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Direct JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # ```json ... ```
    match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.I | re.S
    )

    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # First object
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


# ==============================================================================
# GEMINI WEB RESEARCHER
# ==============================================================================

def gemini_web_search(task_name, search_query):
    """
    Ask Gemini to search the live web using Google's Search grounding.

    The model itself performs the searches.
    We request structured vacancy records.
    """

    if ai_client is None:
        print("  [!] Gemini client unavailable.")
        return []

    prompt = f"""
You are a PhD vacancy research agent.

Today's date is {datetime.now().date()}.

Find CURRENT and REAL PhD vacancy/project opportunities related to:

{task_name}

Search query:
{search_query}

SEARCH STRATEGY
---------------
Search the live web.

Prioritize:
1. FindAPhD
2. EURAXESS
3. Official university PhD vacancy pages
4. Official research group / laboratory pages
5. Reputable academic job boards

IMPORTANT:
- Search specifically for actual PhD vacancies.
- Prefer funded / fully funded positions.
- Prefer positions accepting international applicants.
- Prefer positions relevant to Fall 2027 or currently open positions
  that could plausibly start around 2027.
- Do NOT return generic university pages.
- Do NOT return generic search result pages.
- Do NOT return articles about PhD opportunities.
- Do NOT invent vacancies.
- Only include a position when you can identify a real source URL.
- If the deadline is unknown, use null.
- If funding is unknown, use null.
- If supervisor is unknown, use null.

For each candidate, extract:
- exact project title
- university
- country
- supervisor if available
- deadline if available
- funding information
- source website
- actual vacancy/project URL
- concise project description
- why it appears relevant to the research topic

Return ONLY valid JSON:

{{
  "positions": [
    {{
      "title": "...",
      "university": "...",
      "country": "...",
      "supervisor": null,
      "deadline": null,
      "funding": null,
      "source": "...",
      "url": "...",
      "description": "...",
      "relevance": "..."
    }}
  ]
}}

Return at most {MAX_RESULTS_PER_WEB_TASK} positions.
"""

    try:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            temperature=0.1,
        )

        response = ai_client.models.generate_content(
            model=GEMINI_SEARCH_MODEL,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None)

        if not text:
            print("  [!] Gemini returned empty response.")
            return []

        data = extract_json(text)

        if not isinstance(data, dict):
            print("  [!] Gemini returned invalid JSON.")
            return []

        positions = data.get("positions", [])

        if not isinstance(positions, list):
            return []

        return positions

    except Exception as exc:
        print(f"  [!] Gemini web search failed: {exc}")
        return []


# ==============================================================================
# GEMINI RELEVANCE EVALUATOR
# ==============================================================================

def evaluate_with_gemini(position):
    if ai_client is None:
        return {
            "is_relevant": False,
            "tier": 4,
            "confidence_score": 0.0,
            "reason": "Gemini unavailable.",
            "_api_ok": False,
        }

    title = normalize_whitespace(position.get("title"))
    description = normalize_whitespace(position.get("description"))
    relevance = normalize_whitespace(position.get("relevance"))
    university = normalize_whitespace(position.get("university"))
    source = normalize_whitespace(position.get("source"))
    url = normalize_whitespace(position.get("url"))

    prompt = f"""
{SYSTEM_PROMPT}

Evaluate this PhD opportunity.

TITLE:
{title}

UNIVERSITY:
{university}

SOURCE:
{source}

URL:
{url}

DESCRIPTION:
{description}

DISCOVERY AGENT'S RELEVANCE NOTE:
{relevance}

Return ONLY JSON:

{{
  "is_relevant": true,
  "tier": 1,
  "confidence_score": 0.95,
  "title": "...",
  "key_topics": ["...", "..."],
  "research_fit": "...",
  "technical_gaps": ["...", "..."],
  "reason": "..."
}}

Rules:
- confidence_score must be between 0 and 1.
- tier must be 1, 2, 3, or 4.
- Tier 1 is the strongest match.
- Tier 2 is a good adjacent match.
- Do not inflate relevance merely because the position says "AI".
"""

    try:
        response = ai_client.models.generate_content(
            model=GEMINI_EVAL_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )

        data = extract_json(
            getattr(response, "text", "")
        )

        if not isinstance(data, dict):
            return {
                "is_relevant": False,
                "tier": 4,
                "confidence_score": 0.0,
                "reason": "Invalid Gemini evaluation.",
                "_api_ok": False,
            }

        data["_api_ok"] = True

        return data

    except Exception as exc:
        print(f"  [!] Gemini evaluation failed: {exc}")

        return {
            "is_relevant": False,
            "tier": 4,
            "confidence_score": 0.0,
            "reason": str(exc),
            "_api_ok": False,
        }


# ==============================================================================
# TELEGRAM
# ==============================================================================

def send_telegram_message(text):
    if not BOT_TOKEN or not CHAT_ID:
        print("  [!] Telegram credentials not configured.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }

    try:
        response = SESSION.post(
            url,
            json=payload,
            timeout=20,
        )

        if response.status_code != 200:
            print(
                f"  [!] Telegram HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
            return False

        return True

    except Exception as exc:
        print(f"  [!] Telegram error: {exc}")
        return False


def format_position_alert(position, evaluation):
    title = html.escape(
        normalize_whitespace(
            position.get("title") or evaluation.get("title")
        )
    )

    university = html.escape(
        normalize_whitespace(position.get("university"))
        or "Unknown"
    )

    country = html.escape(
        normalize_whitespace(position.get("country"))
        or "Unknown"
    )

    supervisor = html.escape(
        normalize_whitespace(position.get("supervisor"))
        or "Not specified"
    )

    deadline = html.escape(
        normalize_whitespace(position.get("deadline"))
        or "Not specified"
    )

    funding = html.escape(
        normalize_whitespace(position.get("funding"))
        or "Not specified"
    )

    source = html.escape(
        normalize_whitespace(position.get("source"))
        or "Web"
    )

    url = canonicalize_url(position.get("url", ""))

    tier = evaluation.get("tier", 4)

    confidence = evaluation.get(
        "confidence_score",
        0.0
    )

    reason = html.escape(
        normalize_whitespace(
            evaluation.get("reason")
        )
    )

    topics = evaluation.get(
        "key_topics",
        []
    )

    if not isinstance(topics, list):
        topics = []

    topics_text = ", ".join(
        html.escape(str(x))
        for x in topics[:8]
    )

    description = html.escape(
        normalize_whitespace(
            position.get("description")
        )
    )

    if len(description) > 700:
        description = description[:700] + "..."

    return (
        "🚀 <b>PhD POSITION FOUND</b>\n"
        "\n"
        f"<b>{title}</b>\n"
        f"🏫 {university}\n"
        f"🌍 {country}\n"
        f"🎓 Tier: <b>{tier}</b>\n"
        f"📊 Confidence: <b>{confidence:.2f}</b>\n"
        "\n"
        f"👤 Supervisor: {supervisor}\n"
        f"💰 Funding: {funding}\n"
        f"⏰ Deadline: {deadline}\n"
        f"🔎 Source: {source}\n"
        "\n"
        f"🧠 <b>Topics:</b> {topics_text}\n"
        "\n"
        f"📝 {description}\n"
        "\n"
        f"🎯 <b>Why relevant:</b> {reason}\n"
        "\n"
        f"🔗 <a href=\"{html.escape(url, quote=True)}\">"
        f"Open position</a>"
    )


# ==============================================================================
# PROCESS A POSITION
# ==============================================================================

def process_position(position, seen):
    url = canonicalize_url(
        position.get("url", "")
    )

    title = normalize_whitespace(
        position.get("title")
    )

    if not url:
        print("    [-] No URL.")
        return False

    if not title:
        print("    [-] No title.")
        return False

    uid = make_uid(url, title)

    if uid in seen:
        print("    [=] Already seen.")
        return False

    # Basic sanity filter before expensive Gemini evaluation.
    combined_text = " ".join(
        [
            title,
            normalize_whitespace(position.get("description")),
            normalize_whitespace(position.get("relevance")),
        ]
    )

    if not text_matches_research(combined_text):
        print("    [-] Weak keyword match.")
        return False

    print(
        f"    🧠 Evaluating: "
        f"{title[:100]}"
    )

    evaluation = evaluate_with_gemini(position)

    if not evaluation.get("_api_ok"):
        # IMPORTANT:
        # Do not mark as seen when Gemini failed.
        return False

    is_relevant = bool(
        evaluation.get("is_relevant", False)
    )

    tier = int(
        evaluation.get("tier", 4)
    )

    try:
        confidence = float(
            evaluation.get(
                "confidence_score",
                0
            )
        )
    except Exception:
        confidence = 0.0

    if not is_relevant:
        print(
            f"    [-] Rejected | "
            f"Tier {tier} | "
            f"{confidence:.2f}"
        )

        mark_as_seen(seen, uid)
        return False

    if tier > 2:
        print(
            f"    [-] Tier {tier} "
            f"(below threshold)"
        )

        mark_as_seen(seen, uid)
        return False

    if confidence < TELEGRAM_MIN_CONFIDENCE:
        print(
            f"    [-] Confidence "
            f"{confidence:.2f} < "
            f"{TELEGRAM_MIN_CONFIDENCE}"
        )

        mark_as_seen(seen, uid)
        return False

    if tier == 2 and not SEND_TIER_2:
        mark_as_seen(seen, uid)
        return False

    print(
        f"    [✓] MATCH | "
        f"Tier {tier} | "
        f"confidence={confidence:.2f}"
    )

    alert = format_position_alert(
        position,
        evaluation
    )

    sent = send_telegram_message(alert)

    # Mark as seen after successful processing.
    # Even if Telegram fails, the position has been evaluated.
    mark_as_seen(seen, uid)

    if sent:
        print("    [✓] Telegram alert sent.")

    return True


# ==============================================================================
# GEMINI WEB DISCOVERY
# ==============================================================================

def scrape_with_gemini(seen):
    print("=" * 70)
    print("🌐 GEMINI WEB RESEARCHER")
    print("=" * 70)

    if ai_client is None:
        print()
        print("[!] Gemini is not configured.")
        print("    Add GEMINI_API_KEY to GitHub Secrets.")
        print()
        return

    total_discovered = 0
    total_matches = 0

    tasks = WEB_SEARCH_TASKS[:MAX_WEB_SEARCH_TASKS]

    for index, task in enumerate(tasks, start=1):

        print()
        print(
            f"🔎 [{index}/{len(tasks)}] "
            f"{task['name']}"
        )

        print(
            f"    Query: {task['query']}"
        )

        positions = gemini_web_search(
            task["name"],
            task["query"]
        )

        print(
            f"    [>] Gemini returned "
            f"{len(positions)} candidates."
        )

        total_discovered += len(positions)

        for position in positions:

            # Force source/task metadata.
            if not position.get("source"):
                position["source"] = (
                    "Gemini Web Search"
                )

            if not position.get("discovered_topic"):
                position["discovered_topic"] = (
                    task["name"]
                )

            url = canonicalize_url(
                position.get("url", "")
            )

            if not url:
                print("    [-] Candidate without URL.")
                continue

            # Ensure it really looks like a web page.
            parsed = urlparse(url)

            if parsed.scheme not in {
                "http",
                "https",
            }:
                print("    [-] Invalid URL.")
                continue

            if process_position(
                position,
                seen
            ):
                total_matches += 1

            time.sleep(REQUEST_DELAY)

        time.sleep(REQUEST_DELAY)

    print()
    print(
        f"[✓] Gemini discovered: "
        f"{total_discovered}"
    )

    print(
        f"[✓] Relevant matches: "
        f"{total_matches}"
    )


# ==============================================================================
# TELEGRAM CHANNEL SCRAPER
# ==============================================================================

def extract_telegram_messages(html_text):
    soup = BeautifulSoup(
        html_text,
        "html.parser"
    )

    messages = []

    for wrapper in soup.select(
        ".tgme_widget_message_wrap"
    ):
        text_node = wrapper.select_one(
            ".tgme_widget_message_text"
        )

        if not text_node:
            continue

        text = normalize_whitespace(
            text_node.get_text(" ", strip=True)
        )

        if not text:
            continue

        time_node = wrapper.select_one(
            "time"
        )

        date_value = None

        if time_node:
            date_value = (
                time_node.get("datetime")
                or time_node.get("title")
            )

        link_node = wrapper.select_one(
            ".tgme_widget_message_date"
        )

        url = ""

        if link_node:
            url = link_node.get("href", "")

        messages.append(
            {
                "text": text,
                "date": date_value,
                "url": url,
            }
        )

    return messages


def scrape_telegram_channel(
    channel,
    seen
):
    print(
        f"📡 Scanning @{channel}..."
    )

    base_url = (
        f"https://t.me/s/{channel}"
    )

    all_messages = []

    before = None

    for _ in range(
        TELEGRAM_PAGES_PER_CHANNEL
    ):

        url = base_url

        if before:
            url += f"?before={before}"

        try:
            response = SESSION.get(
                url,
                timeout=20
            )

            if response.status_code != 200:
                print(
                    f"    [-] HTTP "
                    f"{response.status_code}"
                )
                break

            messages = extract_telegram_messages(
                response.text
            )

            if not messages:
                break

            all_messages.extend(messages)

            # Telegram message IDs are encoded
            # in the href.
            ids = []

            for message in messages:
                match = re.search(
                    r"/(\d+)$",
                    message.get("url", "")
                )

                if match:
                    ids.append(
                        int(match.group(1))
                    )

            if not ids:
                break

            before = min(ids)

        except Exception as exc:
            print(
                f"    [-] Telegram error: {exc}"
            )
            break

        time.sleep(REQUEST_DELAY)

    for message in all_messages:

        text = message["text"]

        if not text_matches_research(text):
            continue

        url = canonicalize_url(
            message.get("url", "")
        )

        if not url:
            url = (
                f"https://t.me/s/{channel}"
            )

        position = {
            "title": (
                text[:150]
                + ("..." if len(text) > 150 else "")
            ),
            "university": "",
            "country": "",
            "supervisor": None,
            "deadline": None,
            "funding": None,
            "source": f"Telegram @{channel}",
            "url": url,
            "description": text,
            "relevance": (
                "Telegram academic vacancy "
                "or PhD announcement."
            ),
        }

        process_position(
            position,
            seen
        )


def scrape_telegram(seen):
    print("=" * 70)
    print("1️⃣ TELEGRAM CHANNELS")
    print("=" * 70)

    for channel in TELEGRAM_CHANNELS:
        scrape_telegram_channel(
            channel,
            seen
        )


# ==============================================================================
# GEMINI SEARCH TEST
# ==============================================================================

def test_gemini_web_search():
    print("=" * 70)
    print("🧪 GEMINI WEB SEARCH TEST")
    print("=" * 70)

    if ai_client is None:
        print("[!] Gemini unavailable.")
        return False

    test_query = (
        "Find one currently advertised funded PhD "
        "position related to visual SLAM, VIO, "
        "robot localization, or sensor fusion."
    )

    results = gemini_web_search(
        "Test",
        test_query
    )

    print(
        f"[✓] Test returned "
        f"{len(results)} positions."
    )

    for result in results[:3]:
        print()
        print(
            "Title:",
            result.get("title")
        )
        print(
            "University:",
            result.get("university")
        )
        print(
            "Source:",
            result.get("source")
        )
        print(
            "URL:",
            result.get("url")
        )

    return bool(results)


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print()
    print("=" * 70)
    print("🚀 Running PhD Finder Agent")
    print("=" * 70)
    print()

    seen = load_seen()

    print(
        f"📂 Cached database has "
        f"{len(seen)} items."
    )

    print()
    print("🔎 Discovery configuration:")
    print(
        f"   Gemini: "
        f"{'✓ configured' if ai_client else '✗ NOT configured'}"
    )

    print(
        f"   Search model: "
        f"{GEMINI_SEARCH_MODEL}"
    )

    print(
        f"   Evaluation model: "
        f"{GEMINI_EVAL_MODEL}"
    )

    print(
        "   FindAPhD: "
        "Gemini Google Search grounding"
    )

    print(
        "   EURAXESS: "
        "Gemini Google Search grounding"
    )

    print(
        "   University pages: "
        "Gemini Google Search grounding"
    )

    # --------------------------------------------------------------------------
    # 1. Telegram
    # --------------------------------------------------------------------------

    scrape_telegram(seen)

    save_seen(seen)

    # --------------------------------------------------------------------------
    # 2. Gemini Web Research
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("2️⃣ GEMINI WEB DISCOVERY")
    print("=" * 70)

    scrape_with_gemini(seen)

    save_seen(seen)

    # --------------------------------------------------------------------------
    # Finish
    # --------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("✅ RUN COMPLETE")
    print("=" * 70)

    print(
        f"📂 Seen database now contains "
        f"{len(seen)} items."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
