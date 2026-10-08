import os
import json
import time
import re
import warnings
import urllib.parse

from datetime import datetime, timezone, timedelta
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")


# ==============================================================================
# OPTIONAL CLOUDSCRAPER
# ==============================================================================

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False


# ==============================================================================
# ENVIRONMENT
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

# --------------------------------------------------------------------------
# Google Programmable Search / Custom Search JSON API
#
# Add these to GitHub Secrets:
#
# GOOGLE_CSE_API_KEY
# GOOGLE_CSE_ID
#
# They are optional. If missing, the script will still try direct scraping.
# --------------------------------------------------------------------------

GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

GOOGLE_CSE_ENDPOINT = (
    "https://www.googleapis.com/customsearch/v1"
)


# ==============================================================================
# DATA
# ==============================================================================

DATA_DIR = "data"
SEEN_FILE = os.path.join(DATA_DIR, "seen_posts.json")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(SEEN_FILE):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


MAX_POST_AGE_DAYS = 60

# Telegram
PAGES_PER_CHANNEL = 5

# FindAPhD direct scraping
FINDAPHD_MAX_PAGES_PER_QUERY = 4
FINDAPHD_MAX_RESULTS_PER_QUERY = 20
FINDAPHD_REQUEST_DELAY = 1.2

# Google discovery
GOOGLE_SEARCH_RESULTS_PER_QUERY = 10
GOOGLE_SEARCH_DELAY = 1.0

# Maximum number of FindAPhD opportunities evaluated in one run.
# This protects Gemini quota.
FINDAPHD_MAX_CANDIDATES_PER_RUN = 60

# Whether to attempt direct FindAPhD pages before switching to search.
FINDAPHD_TRY_DIRECT = True

# Once FindAPhD returns 403, stop direct scraping for the rest of this run.
FINDAPHD_STOP_DIRECT_ON_403 = True


# ==============================================================================
# HTTP SESSION
# ==============================================================================

if HAS_CLOUDSCRAPER:
    session = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "linux",
            "mobile": False
        }
    )
else:
    session = requests.Session()


session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
})


# ==============================================================================
# GEMINI
# ==============================================================================

from google import genai
from google.genai import types


if not GEMINI_API_KEY:
    print("[!] GEMINI_API_KEY is not configured.")

ai_client = genai.Client(api_key=GEMINI_API_KEY)


# ==============================================================================
# TELEGRAM CHANNELS
# ==============================================================================

CHANNELS_TO_SCRAPE = [
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
# FINDAPHD CONFIGURATION
# ==============================================================================

FINDAPHD_DOMAIN = "www.findaphd.com"

FINDAPHD_SEARCH_URL = (
    "https://www.findaphd.com/phds/"
)

FINDAPHD_CATEGORY_URLS = [
    "https://www.findaphd.com/phds/engineering/?10M7o0",
    "https://www.findaphd.com/phds/computer-science/?10M7g0",
    "https://www.findaphd.com/phds/engineering/",
    "https://www.findaphd.com/phds/computer-science/",
]


# ==============================================================================
# TARGETED ACADEMIC QUERIES
# ==============================================================================

ACADEMIC_SEARCH_QUERIES = [

    # Tier 1
    "visual inertial odometry",
    "visual SLAM",
    "visual navigation",
    "sensor fusion localization",
    "GNSS positioning",
    "GNSS sensor fusion",
    "robot localization",
    "robot navigation",

    # Computer vision + robotics
    "computer vision robotics",
    "3D computer vision robotics",
    "visual perception autonomous robots",
    "visual localization",
    "camera IMU fusion",
    "LiDAR camera fusion",
    "visual odometry",

    # Estimation / optimization
    "factor graph optimization robotics",
    "state estimation robotics",
    "probabilistic robotics",
    "pose estimation",
    "trajectory estimation",

    # Autonomous systems
    "autonomous navigation",
    "field robotics",
    "mobile robot localization",
    "outdoor robotics",
    "autonomous vehicles",

    # ML
    "machine learning robotics",
    "deep learning computer vision robotics",
    "self supervised visual navigation",
    "uncertainty estimation robotics",

    # Embedded / Edge
    "embedded AI robotics",
    "edge AI computer vision",
    "GPU accelerated computer vision",
    "CUDA computer vision",
]


# Additional queries specifically for search-engine discovery.
#
# These are deliberately different from the FindAPhD internal search terms.
# Google will search the FindAPhD index rather than requesting FindAPhD directly.
FINDAPHD_DISCOVERY_QUERIES = [

    'site:findaphd.com/phds/ "visual inertial odometry"',
    'site:findaphd.com/phds/ "visual SLAM"',
    'site:findaphd.com/phds/ "visual navigation"',
    'site:findaphd.com/phds/ "sensor fusion" localization',
    'site:findaphd.com/phds/ "GNSS" positioning',
    'site:findaphd.com/phds/ "GNSS" navigation',
    'site:findaphd.com/phds/ "robot localization"',
    'site:findaphd.com/phds/ "robot navigation"',
    'site:findaphd.com/phds/ "computer vision" robotics',
    'site:findaphd.com/phds/ "3D computer vision"',
    'site:findaphd.com/phds/ "visual perception"',
    'site:findaphd.com/phds/ "visual localization"',
    'site:findaphd.com/phds/ "camera IMU"',
    'site:findaphd.com/phds/ "LiDAR" vision',
    'site:findaphd.com/phds/ "visual odometry"',
    'site:findaphd.com/phds/ "factor graph" robotics',
    'site:findaphd.com/phds/ "state estimation" robotics',
    'site:findaphd.com/phds/ "pose estimation" robotics',
    'site:findaphd.com/phds/ "trajectory estimation"',
    'site:findaphd.com/phds/ "autonomous navigation"',
    'site:findaphd.com/phds/ "field robotics"',
    'site:findaphd.com/phds/ "mobile robot" localization',
    'site:findaphd.com/phds/ "autonomous vehicle"',
    'site:findaphd.com/phds/ "self-driving"',
    'site:findaphd.com/phds/ "machine learning" robotics',
    'site:findaphd.com/phds/ "deep learning" "computer vision"',
    'site:findaphd.com/phds/ "uncertainty estimation" robotics',
    'site:findaphd.com/phds/ "embedded AI" robotics',
    'site:findaphd.com/phds/ "edge AI" robotics',

    # Broader queries useful for vacancies whose title is generic.
    'site:findaphd.com/phds/ "autonomous systems" robotics',
    'site:findaphd.com/phds/ "robot perception"',
    'site:findaphd.com/phds/ "localization" "computer vision"',
    'site:findaphd.com/phds/ "navigation" "computer vision"',
]


# Latest PhD discovery.
FINDAPHD_LATEST_DISCOVERY_QUERIES = [
    'site:findaphd.com/phds/latest/ "robotics"',
    'site:findaphd.com/phds/latest/ "computer vision"',
    'site:findaphd.com/phds/latest/ "autonomous"',
    'site:findaphd.com/phds/latest/ "navigation"',
    'site:findaphd.com/phds/latest/ "sensor fusion"',
]


# ==============================================================================
# KEYWORD FILTERING
# ==============================================================================

BROAD_PATTERNS = [

    # VIO / SLAM / Navigation
    r"\bv[io]\b",
    r"visual[- ]inertial",
    r"visual[- ]inertial odometry",
    r"\bvio\b",
    r"visual odometry",
    r"visual[- ]inertial navigation",
    r"visual navigation",
    r"\bslam\b",
    r"visual slam",
    r"visual[- ]inertial slam",
    r"simultaneous localization and mapping",
    r"localization and mapping",

    # Sensor fusion
    r"sensor fusion",
    r"multi[- ]sensor fusion",
    r"multimodal sensor fusion",
    r"camera[- ]imu",
    r"vision[- ]imu",
    r"gnss[- ]imu",
    r"gnss/imu",
    r"gnss fusion",

    # GNSS / positioning
    r"\bgnss\b",
    r"gps positioning",
    r"gnss positioning",
    r"satellite positioning",
    r"robust positioning",
    r"precise positioning",
    r"navigation system",
    r"localization",
    r"positioning",

    # Computer vision
    r"computer vision",
    r"machine vision",
    r"3d vision",
    r"3d computer vision",
    r"image processing",
    r"video processing",
    r"object detection",
    r"object tracking",
    r"multi[- ]object tracking",
    r"visual perception",
    r"scene understanding",
    r"depth estimation",
    r"stereo vision",
    r"visual localization",
    r"place recognition",
    r"camera[- ]based navigation",

    # Robotics
    r"robot localization",
    r"robot navigation",
    r"autonomous navigation",
    r"autonomous systems",
    r"mobile robot",
    r"mobile robotics",
    r"field robotics",
    r"outdoor robotics",
    r"autonomous robot",
    r"robot perception",
    r"autonomous vehicle",
    r"self[- ]driving",

    # ML
    r"machine learning",
    r"deep learning",
    r"neural network",
    r"transformer",
    r"self[- ]supervised learning",
    r"representation learning",
    r"domain adaptation",
    r"domain generalization",
    r"uncertainty estimation",
    r"probabilistic learning",
    r"graph neural network",
    r"\bgnn\b",

    # Estimation
    r"factor graph",
    r"factor graph optimization",
    r"graph[- ]based optimization",
    r"nonlinear optimization",
    r"state estimation",
    r"pose estimation",
    r"trajectory estimation",
    r"kalman filter",
    r"extended kalman",
    r"particle filter",
    r"probabilistic robotics",

    # Sensors
    r"\bimu\b",
    r"inertial measurement",
    r"inertial navigation",
    r"inertial sensing",
    r"lidar",
    r"lidar[- ]based",
    r"camera[- ]lidar",
    r"lidar[- ]camera",
    r"gnss",
    r"rtk",
    r"pseudorange",

    # Embedded / Edge
    r"embedded ai",
    r"edge ai",
    r"edge computing",
    r"embedded systems",
    r"embedded software",
    r"embedded machine learning",
    r"tinyml",
    r"ai accelerator",
    r"hardware acceleration",
    r"gpu acceleration",
    r"cuda",
    r"nvidia jetson",
    r"edge robotics",
    r"\bros\b",
    r"ros2",
    r"robot operating system",

    # Hardware
    r"\bfpga\b",
    r"\bsdr\b",
    r"software defined radio",
    r"vhdl",
    r"verilog",
    r"digital hardware",
    r"digital systems",
    r"embedded hardware",

    # PhD signals
    r"\bphd\b",
    r"ph\.d",
    r"doctoral",
    r"phd position",
    r"phd candidate",
    r"doctoral position",
    r"research assistant",
    r"funded phd",
    r"fully funded",
    r"funded position",

    # Persian
    r"ناوبری",
    r"موقعیت[- ]?یابی",
    r"مکان[- ]?یابی",
    r"بینایی ماشین",
    r"بینایی کامپیوتر",
    r"پردازش تصویر",
    r"پردازش ویدئو",
    r"یادگیری ماشین",
    r"یادگیری عمیق",
    r"ادغام سنسورها",
    r"همجوشی سنسورها",
    r"اسلم",
    r"ناوبری ربات",
    r"ربات خودران",
    r"سیستم نهفته",
    r"هوش مصنوعی لبه",
]


EXCLUDE_PATTERNS = [

    # Immigration / language
    r"ویزای همسر",
    r"ویزای کاری",
    r"تعیین وقت سفارت",
    r"کلاس زبان",
    r"آموزش آیلتس",
    r"ielts class",
    r"immigration lawyer",

    # Wet lab / biology
    r"wet[- ]lab",
    r"molecular biology",
    r"cell culture",
    r"gene expression",
    r"genomics",
    r"proteomics",
    r"immunotherapy",
    r"oncology",
    r"tissue staining",
    r"cardiovascular",

    # Chemistry / materials
    r"organic chemistry",
    r"inorganic chemistry",
    r"polymer chemistry",

    # Mechanical / manipulation
    r"robotic arm manipulation",
    r"robot[- ]assisted surgery",
    r"prosthetics",
    r"exoskeleton",
    r"pure mechanical design",
    r"fluid mechanics",
    r"thermodynamics",

    # Policy / social science
    r"public policy",
    r"social sciences",
    r"political science",
]


# ==============================================================================
# GEMINI SYSTEM PROMPT
# ==============================================================================

SYSTEM_PROMPT = """You are an expert academic evaluator. Assess whether a PhD vacancy
matches the candidate's research profile and technical background.

Candidate Profile:

Education:
- M.Sc. Electrical Engineering (Digital Electronics), Iran University of Science and Technology
- B.Sc. Electrical Engineering, Ferdowsi University of Mashhad

Core Research Interests:
- Visual-Inertial Odometry (VIO)
- Visual SLAM
- Computer Vision
- Sensor Fusion
- GNSS Navigation and Positioning
- Autonomous Navigation
- Robotics
- Embedded AI

Research / Technical Background:
- GNSS positioning using Deep Learning and Factor Graph Optimization
- Sensor fusion and robust localization
- Computer vision and image processing
- Object detection and multi-object tracking
- Level Set medical image segmentation
- Embedded systems and ARM microcontrollers
- NVIDIA Jetson-oriented robotics project
- ROS2
- Python, C/C++, PyTorch, OpenCV
- MATLAB
- Git, Linux
- VHDL / FPGA-oriented digital hardware
- CUDA / GPU acceleration is an area of interest and ongoing development

Candidate's Target Research Direction:
The strongest target is research involving visual navigation, VIO, SLAM,
sensor fusion, GNSS/INS integration, robot localization, autonomous navigation,
computer vision for robotics, and perception for autonomous systems.

Tier 1 — Direct Match:
- Visual-Inertial Odometry
- Visual SLAM
- Visual navigation
- Sensor fusion for localization/navigation
- GNSS/INS integration
- GNSS positioning and robust localization
- Robot localization
- Autonomous navigation
- Computer vision for robotics
- LiDAR-camera-IMU fusion
- State estimation for mobile robots
- Factor graph optimization for robotics/navigation
- Perception and localization for autonomous systems

Tier 2 — Strongly Adjacent:
- Computer vision
- 3D computer vision
- Visual perception
- Object detection/tracking for autonomous systems
- LiDAR perception
- Multi-modal perception
- Probabilistic robotics
- Pose/trajectory estimation
- Machine learning / deep learning for robotics
- Self-supervised learning for visual navigation
- Domain adaptation/generalization for perception
- Graph neural networks for robotics
- Uncertainty estimation
- Autonomous vehicles
- Field robotics
- Mobile robotics
- Embedded AI / Edge AI
- CUDA/GPU acceleration for robotics or computer vision
- ROS/ROS2

Tier 3 — Relevant Hardware / Electronics:
- Embedded systems
- Embedded software
- ARM / microcontrollers
- FPGA
- VHDL / Verilog
- Digital hardware acceleration
- AI accelerators
- SDR
- Sensor interfaces
- Real-time systems

Tier 4 — Weak/Conditional Match:
- General machine learning
- General deep learning
- General image processing
- General signal processing
- General wireless sensing
- General IoT
- General autonomous systems

Important:
Research topic and expected work are more important than the list of tools.
Do not consider a vacancy highly relevant merely because it contains generic
terms such as AI, Machine Learning, Python, or Robotics.

A robotics position is a strong match if it involves localization, perception,
navigation, SLAM, sensor fusion, or autonomous systems.

A computer vision position is a strong match if it involves robotics,
navigation, localization, 3D vision, or autonomous systems.

A GNSS position is a strong match even if it does not mention deep learning.

Prefer funded PhD / doctoral research positions.

Reject:
- Pure mechanical engineering
- Pure robotic manipulation without perception/localization
- Prosthetics / exoskeletons unless strongly sensing/navigation focused
- Medical wet-lab biology
- Pure chemistry
- Pure materials science
- Pure theoretical mathematics
- Pure telecommunications without localization/sensing relevance
- Pure power electronics
- Pure control theory with no robotics/navigation application
- Pure software engineering with no research connection
- Visa advertisements
- Language courses

Return ONLY valid JSON:

{
  "is_relevant": true,
  "tier": 1,
  "confidence_score": 9,
  "title": "Short position title",
  "key_topics": ["topic1", "topic2", "topic3"],
  "research_fit": "1-line explanation",
  "technical_gaps": ["missing skill 1"],
  "reason": "1-line final explanation"
}

Scoring:
9-10 = Excellent fit
7-8 = Strong fit
5-6 = Potential fit
3-4 = Weak fit
0-2 = Not relevant

Tier:
1 = Direct match
2 = Strongly adjacent
3 = Hardware / embedded
4 = Weak / conditional
"""


# ==============================================================================
# SEEN DATABASE
# ==============================================================================

def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass

    return set()


def mark_as_seen(uid: str, seen: set):
    if not uid:
        return

    seen.add(uid)

    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(
                sorted(list(seen)),
                f,
                ensure_ascii=False,
                indent=2
            )
    except Exception as e:
        print(f"[!] Could not save seen database: {e}")


# ==============================================================================
# GENERAL HELPERS
# ==============================================================================

def normalize_whitespace(text: str) -> str:
    if not text:
        return ""

    return re.sub(r"\s+", " ", str(text)).strip()


def canonicalize_url(url: str) -> str:
    if not url:
        return ""

    url = url.strip()

    parsed = urlparse(url)

    parsed = parsed._replace(fragment="")

    query = parse_qs(
        parsed.query,
        keep_blank_values=True
    )

    tracking_prefixes = (
        "utm_",
        "fbclid",
        "gclid",
        "ref",
        "source"
    )

    clean_query = {}

    for key, values in query.items():

        if key.lower().startswith(tracking_prefixes):
            continue

        clean_query[key] = values

    new_query = urlencode(
        clean_query,
        doseq=True
    )

    parsed = parsed._replace(
        query=new_query
    )

    result = urlunparse(parsed)

    if (
        result.endswith("/")
        and parsed.path not in ("", "/")
    ):
        result = result[:-1]

    return result


def make_findaphd_uid(
    url: str,
    title: str = ""
) -> str:

    canonical = canonicalize_url(url)

    if canonical:
        return canonical

    return (
        "findaphd:"
        + normalize_whitespace(title).lower()
    )


# ==============================================================================
# FILTERING
# ==============================================================================

def contains_excluded_pattern(text: str) -> bool:

    lower_t = text.lower()

    return any(
        re.search(pattern, lower_t)
        for pattern in EXCLUDE_PATTERNS
    )


def keyword_score(text: str) -> int:

    lower_t = text.lower()

    return sum(
        1
        for pattern in BROAD_PATTERNS
        if re.search(pattern, lower_t)
    )


def fast_prefilter(text: str) -> bool:

    if not text:
        return False

    if contains_excluded_pattern(text):
        return False

    return keyword_score(text) > 0


def findaphd_soft_prefilter(
    title: str,
    description: str
) -> bool:

    text = f"{title}\n{description}"

    if contains_excluded_pattern(text):
        return False

    strong_patterns = [
        r"visual",
        r"robot",
        r"autonomous",
        r"navigation",
        r"localization",
        r"localisation",
        r"slam",
        r"odometry",
        r"sensor fusion",
        r"computer vision",
        r"perception",
        r"gnss",
        r"gps",
        r"lidar",
        r"inertial",
        r"imu",
        r"positioning",
        r"mobile robotics",
        r"machine learning",
        r"deep learning",
        r"3d vision",
        r"autonomous vehicle",
        r"self[- ]driving",
        r"embedded ai",
    ]

    lower_text = text.lower()

    return any(
        re.search(pattern, lower_text)
        for pattern in strong_patterns
    )


# ==============================================================================
# DATE HELPERS
# ==============================================================================

def is_recent_date(dt_obj) -> bool:

    if not dt_obj:
        return True

    try:

        now = datetime.now(timezone.utc)

        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(
                tzinfo=timezone.utc
            )

        return (
            now - dt_obj
        ) <= timedelta(
            days=MAX_POST_AGE_DAYS
        )

    except Exception:
        return True


def parse_iso_time(time_str: str):

    try:

        return datetime.fromisoformat(
            time_str.replace(
                "Z",
                "+00:00"
            )
        )

    except Exception:
        return None


# ==============================================================================
# GEMINI JSON
# ==============================================================================

def extract_clean_json(
    raw_text: str
) -> dict:

    if not raw_text:
        return {
            "is_relevant": False,
            "reason": "Empty Gemini response"
        }

    try:

        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")

        if (
            start_idx != -1
            and end_idx != -1
            and end_idx >= start_idx
        ):

            return json.loads(
                raw_text[
                    start_idx:end_idx + 1
                ]
            )

    except Exception:
        pass

    return {
        "is_relevant": False,
        "reason": "Failed to parse JSON output"
    }


def evaluate_with_gemini(
    text: str
) -> dict:

    if not GEMINI_API_KEY:

        return {
            "is_relevant": False,
            "reason": "GEMINI_API_KEY missing"
        }

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite"
    ]

    for attempt in range(3):

        for model_name in models_to_try:

            try:

                response = (
                    ai_client.models.generate_content(
                        model=model_name,
                        contents=(
                            "Evaluate this PhD post:\n\n"
                            f"{text[:7000]}"
                        ),
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.1,
                        )
                    )
                )

                time.sleep(4.2)

                if (
                    not response.candidates
                    or not response.candidates[0].content.parts
                ):

                    return {
                        "is_relevant": False,
                        "reason": (
                            "Blocked by Gemini "
                            "Safety Filters"
                        )
                    }

                return extract_clean_json(
                    response.text
                )

            except Exception as e:

                err_str = str(e)

                if (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED"
                    in err_str
                ):

                    print(
                        "[!] Gemini rate limit. "
                        "Waiting 25 seconds..."
                    )

                    time.sleep(25)

                    break

                print(
                    f"[!] Gemini error "
                    f"({model_name}): "
                    f"{err_str[:180]}"
                )

                continue

    return {
        "is_relevant": False,
        "reason": "API error after retries"
    }


# ==============================================================================
# TELEGRAM ALERT
# ==============================================================================

def send_alert(
    url: str,
    analysis: dict,
    snippet: str
):

    if not isinstance(analysis, dict):
        return False

    tier_badges = {
        1: "🔥 Tier 1",
        2: "⚡ Tier 2",
        3: "🛠 Tier 3"
    }

    tier_val = analysis.get(
        "tier",
        2
    )

    label = tier_badges.get(
        tier_val,
        "Relevant Opportunity"
    )

    score = analysis.get(
        "confidence_score",
        "N/A"
    )

    title = analysis.get(
        "title",
        "PhD Opportunity"
    )

    topics_list = analysis.get(
        "key_topics",
        []
    )

    topics = (
        ", ".join(topics_list)
        if isinstance(topics_list, list)
        else str(topics_list)
    )

    reason = analysis.get(
        "reason",
        "Matches research profile."
    )

    msg = (
        f"🎯 <b>New PhD Match!</b>\n"
        f"<b>Category:</b> {label}\n"
        f"<b>Score:</b> {score}/10\n"
        f"<b>Title:</b> {title}\n"
        f"<b>Topics:</b> <code>{topics}</code>\n\n"
        f"💡 <b>Reason:</b> {reason}\n\n"
        f'🔗 <a href="{url}">Open Vacancy / Post</a>\n'
        f"────────────────────\n"
        f"📝 <b>Preview:</b>\n"
        f"{snippet[:700]}..."
    )

    endpoint = (
        "https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": False
    }

    try:

        response = session.post(
            endpoint,
            json=payload,
            timeout=20
        )

        print(
            f"[Telegram] HTTP "
            f"{response.status_code}"
        )

        if response.status_code != 200:

            print(
                "[-] Telegram failed!"
            )

            return False

        result = response.json()

        if not result.get("ok"):

            print(
                f"[-] Telegram API error: "
                f"{result}"
            )

            return False

        print(
            "[+] Telegram alert sent successfully."
        )

        return True

    except Exception as e:

        print(
            f"[-] Alert sending error: {e}"
        )

        return False


# ==============================================================================
# TELEGRAM SCRAPER
# ==============================================================================

def scrape_telegram_channel_deep(
    channel: str,
    seen: set
):

    print(
        f"\n📡 Scanning @{channel}..."
    )

    current_url = (
        f"https://t.me/s/{channel}"
    )

    for _ in range(
        PAGES_PER_CHANNEL
    ):

        try:

            r = session.get(
                current_url,
                timeout=25
            )

            if r.status_code != 200:

                print(
                    f"[-] Telegram HTTP "
                    f"{r.status_code}"
                )

                break

            soup = BeautifulSoup(
                r.text,
                "html.parser"
            )

            messages = soup.find_all(
                "div",
                class_="tgme_widget_message"
            )

            if not messages:
                break

            oldest_id_on_page = None

            for msg in reversed(messages):

                msg_id = msg.get(
                    "data-post"
                )

                if not msg_id:
                    continue

                raw_num = (
                    msg_id.split("/")[-1]
                )

                if raw_num.isdigit():

                    oldest_id_on_page = (
                        int(raw_num)
                        if oldest_id_on_page is None
                        else min(
                            oldest_id_on_page,
                            int(raw_num)
                        )
                    )

                if msg_id in seen:
                    continue

                time_tag = msg.find("time")

                if (
                    time_tag
                    and time_tag.get("datetime")
                ):

                    dt = parse_iso_time(
                        time_tag.get(
                            "datetime"
                        )
                    )

                    if (
                        dt
                        and not is_recent_date(dt)
                    ):

                        mark_as_seen(
                            msg_id,
                            seen
                        )

                        continue

                text_elem = msg.find(
                    "div",
                    class_="tgme_widget_message_text"
                )

                if not text_elem:

                    mark_as_seen(
                        msg_id,
                        seen
                    )

                    continue

                text = text_elem.get_text(
                    separator="\n"
                ).strip()

                if (
                    len(text) < 40
                    or not fast_prefilter(text)
                ):

                    mark_as_seen(
                        msg_id,
                        seen
                    )

                    continue

                print(
                    f"[+] Evaluating Telegram post: "
                    f"{msg_id}"
                )

                analysis = evaluate_with_gemini(
                    text
                )

                if analysis.get(
                    "is_relevant"
                ):

                    send_alert(
                        f"https://t.me/{msg_id}",
                        analysis,
                        text
                    )

                    print(
                        "    [✓] MATCH CONFIRMED "
                        f"(Tier {analysis.get('tier')} - "
                        f"Score "
                        f"{analysis.get('confidence_score')}/10)"
                    )

                else:

                    print(
                        "    [-] Skipped: "
                        f"{analysis.get('reason', 'LLM filter')}"
                    )

                mark_as_seen(
                    msg_id,
                    seen
                )

            if oldest_id_on_page:

                current_url = (
                    f"https://t.me/s/{channel}"
                    f"?before={oldest_id_on_page}"
                )

            else:
                break

        except Exception as e:

            print(
                f"[-] @{channel} issue: {e}"
            )

            break


# ==============================================================================
# FINDAPHD URL HELPERS
# ==============================================================================

def build_findaphd_search_url(
    keyword: str,
    page: int = 1
) -> str:

    params = {
        "Keywords": keyword
    }

    if page > 1:
        params["Page"] = page

    return (
        FINDAPHD_SEARCH_URL
        + "?"
        + urllib.parse.urlencode(params)
    )


def build_findaphd_category_url(
    base_url: str,
    page: int
) -> str:

    if page <= 1:
        return base_url

    parsed = urlparse(base_url)

    query = parse_qs(
        parsed.query,
        keep_blank_values=True
    )

    query["Page"] = [str(page)]

    new_query = urlencode(
        query,
        doseq=True
    )

    return urlunparse(
        parsed._replace(
            query=new_query
        )
    )


def is_findaphd_vacancy_url(
    url: str
) -> bool:

    if not url:
        return False

    parsed = urlparse(url)

    if (
        "findaphd.com"
        not in parsed.netloc.lower()
    ):
        return False

    path = parsed.path.lower()

    if path in (
        "",
        "/",
        "/phds",
        "/phds/"
    ):
        return False

    if not path.startswith(
        "/phds/"
    ):
        return False

    remaining = path[
        len("/phds/"):
    ]

    # Search/category pages usually don't have
    # another path segment.
    if "/" not in remaining:
        return False

    bad_prefixes = [
        "/phds/engineering",
        "/phds/computer-science",
        "/phds/discipline",
        "/phds/latest",
        "/phds/subject",
        "/phds/universities",
    ]

    if any(
        path.startswith(p)
        for p in bad_prefixes
    ):
        return False

    return True


# ==============================================================================
# FINDAPHD LISTING PARSER
# ==============================================================================

def extract_findaphd_listing_links(
    soup: BeautifulSoup,
    base_url: str
):

    results = {}

    for a in soup.find_all(
        "a",
        href=True
    ):

        href = a.get(
            "href",
            ""
        ).strip()

        if not href:
            continue

        absolute_url = canonicalize_url(
            urljoin(
                base_url,
                href
            )
        )

        if not is_findaphd_vacancy_url(
            absolute_url
        ):
            continue

        title = normalize_whitespace(
            a.get_text(
                " ",
                strip=True
            )
        )

        if len(title) < 10:
            continue

        navigation_words = {
            "view",
            "apply",
            "more",
            "details",
            "read more",
            "view project",
            "view phd",
            "see all",
        }

        if title.lower() in navigation_words:
            continue

        results.setdefault(
            absolute_url,
            {
                "url": absolute_url,
                "anchor_title": title
            }
        )

    return list(
        results.values()
    )


def extract_findaphd_card_for_link(
    soup: BeautifulSoup,
    link: str
):

    canonical_link = canonicalize_url(
        link
    )

    anchor = None

    for a in soup.find_all(
        "a",
        href=True
    ):

        candidate = canonicalize_url(
            urljoin(
                "https://www.findaphd.com",
                a.get("href", "")
            )
        )

        if candidate == canonical_link:

            anchor = a
            break

    if not anchor:
        return ""

    parent = anchor

    for _ in range(6):

        parent = parent.parent

        if not parent:
            break

        text = normalize_whitespace(
            parent.get_text(
                " ",
                strip=True
            )
        )

        if 80 <= len(text) <= 5000:

            if (
                len(parent.find_all("a")) >= 2
                or parent.find(
                    ["h1", "h2", "h3", "h4"]
                )
            ):
                return text

    return normalize_whitespace(
        anchor.parent.get_text(
            " ",
            strip=True
        )
    )


# ==============================================================================
# FINDAPHD DIRECT DETAIL PAGE
# ==============================================================================

def parse_findaphd_detail_page(
    url: str
) -> dict:

    result = {
        "title": "",
        "description": "",
        "university": "",
        "location": "",
        "funding": "",
        "deadline": "",
    }

    try:

        response = session.get(
            url,
            timeout=25
        )

        if response.status_code != 200:

            print(
                f"    [-] Detail HTTP "
                f"{response.status_code}"
            )

            return result

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        title_elem = (
            soup.find("h1")
            or soup.find(
                "meta",
                property="og:title"
            )
        )

        if title_elem:

            if title_elem.name == "meta":

                result["title"] = (
                    title_elem.get(
                        "content",
                        ""
                    ).strip()
                )

            else:

                result["title"] = (
                    normalize_whitespace(
                        title_elem.get_text(
                            " ",
                            strip=True
                        )
                    )
                )

        meta_desc = soup.find(
            "meta",
            attrs={
                "name": "description"
            }
        )

        if meta_desc:

            result["description"] = (
                normalize_whitespace(
                    meta_desc.get(
                        "content",
                        ""
                    )
                )
            )

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.body
        )

        if main:

            main_text = normalize_whitespace(
                main.get_text(
                    " ",
                    strip=True
                )
            )

            if len(main_text) > len(
                result["description"]
            ):

                result["description"] = (
                    main_text[:12000]
                )

        # JSON-LD
        for script in soup.find_all(
            "script",
            type="application/ld+json"
        ):

            try:

                raw = script.string

                if not raw:
                    continue

                data = json.loads(raw)

                if isinstance(
                    data,
                    list
                ):

                    candidates = data

                else:

                    candidates = [data]

                for item in candidates:

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue

                    if not result["title"]:

                        result["title"] = (
                            str(
                                item.get(
                                    "name",
                                    ""
                                )
                            ).strip()
                        )

                    desc = str(
                        item.get(
                            "description",
                            ""
                        )
                    ).strip()

                    if (
                        desc
                        and len(desc)
                        > len(
                            result["description"]
                        )
                    ):

                        result["description"] = desc

            except Exception:
                continue

        page_text = normalize_whitespace(
            soup.get_text(
                " ",
                strip=True
            )
        )

        # Label-based extraction.
        label_patterns = {

            "funding": [
                r"funding",
                r"funded",
                r"studentship"
            ],

            "deadline": [
                r"deadline",
                r"application deadline"
            ],

            "location": [
                r"location",
                r"study location"
            ],
        }

        for field, patterns in (
            label_patterns.items()
        ):

            for pattern in patterns:

                match = re.search(
                    rf"{pattern}\s*[:\-]?\s*(.{{5,200}})",
                    page_text,
                    re.I
                )

                if match:

                    result[field] = (
                        normalize_whitespace(
                            match.group(1)
                        )
                    )

                    break

        return result

    except Exception as e:

        print(
            f"    [-] FindAPhD detail error: "
            f"{e}"
        )

        return result


# ==============================================================================
# FINDAPHD DIRECT ACCESS STATE
# ==============================================================================

FINDAPHD_DIRECT_BLOCKED = False


def findaphd_direct_get(
    url: str
):
    """
    Centralized FindAPhD request.

    Once the server returns 403, we stop hammering the site.
    """

    global FINDAPHD_DIRECT_BLOCKED

    if FINDAPHD_DIRECT_BLOCKED:
        return None

    try:

        response = session.get(
            url,
            timeout=25
        )

        if response.status_code == 403:

            print(
                "    [!] FindAPhD returned HTTP 403."
            )

            if FINDAPHD_STOP_DIRECT_ON_403:

                FINDAPHD_DIRECT_BLOCKED = True

                print(
                    "    [!] Direct FindAPhD access "
                    "disabled for this run."
                )

            return None

        return response

    except Exception as e:

        print(
            f"    [-] FindAPhD request error: "
            f"{e}"
        )

        return None


# ==============================================================================
# FINDAPHD DIRECT LISTING
# ==============================================================================

def collect_findaphd_listing(
    listing,
    page_soup
):

    url = listing["url"]

    anchor_title = listing[
        "anchor_title"
    ]

    card_text = (
        extract_findaphd_card_for_link(
            page_soup,
            url
        )
    )

    detail = parse_findaphd_detail_page(
        url
    )

    title = (
        detail["title"]
        or anchor_title
    )

    description = (
        detail["description"]
        or card_text
    )

    full_text = (
        f"Title: {title}\n"
        f"University: "
        f"{detail['university']}\n"
        f"Location: "
        f"{detail['location']}\n"
        f"Funding: "
        f"{detail['funding']}\n"
        f"Deadline: "
        f"{detail['deadline']}\n"
        f"Description: "
        f"{description}"
    )

    return {
        "url": url,
        "title": title,
        "text": full_text,
        "university": detail[
            "university"
        ],
        "location": detail[
            "location"
        ],
        "funding": detail[
            "funding"
        ],
        "deadline": detail[
            "deadline"
        ],
    }


# ==============================================================================
# FINDAPHD CANDIDATE PROCESSOR
# ==============================================================================

def process_findaphd_candidate(
    vacancy: dict,
    seen: set,
    run_seen: set,
    source_label: str
) -> bool:

    url = vacancy.get(
        "url",
        ""
    )

    title = normalize_whitespace(
        vacancy.get(
            "title",
            ""
        )
    )

    text = normalize_whitespace(
        vacancy.get(
            "text",
            ""
        )
    )

    uid = make_findaphd_uid(
        url,
        title
    )

    if not uid:
        return False

    if uid in seen:
        return False

    if uid in run_seen:
        return False

    run_seen.add(uid)

    if not title:
        title = "FindAPhD Opportunity"

    if len(text) < 50:

        print(
            f"    [-] Too little content: "
            f"{title[:80]}"
        )

        # We do not permanently mark malformed
        # search results as seen.
        return False

    if not findaphd_soft_prefilter(
        title,
        text
    ):

        print(
            f"    [-] Prefilter rejected: "
            f"{title[:80]}"
        )

        mark_as_seen(
            uid,
            seen
        )

        return False

    print(
        f"    [+] Evaluating "
        f"[{source_label}]: "
        f"{title[:90]}"
    )

    analysis = evaluate_with_gemini(
        text
    )

    if analysis.get(
        "is_relevant"
    ):

        sent = send_alert(
            url,
            analysis,
            text
        )

        print(
            "        [✓] MATCH "
            f"Tier {analysis.get('tier')} "
            f"/ Score "
            f"{analysis.get('confidence_score')}/10"
        )

        if not sent:
            print(
                "        [!] Telegram alert "
                "could not be sent."
            )

    else:

        print(
            "        [-] Rejected: "
            f"{analysis.get('reason', 'LLM filter')}"
        )

    # IMPORTANT:
    # Only now permanently mark the opportunity as processed.
    mark_as_seen(
        uid,
        seen
    )

    return True


# ==============================================================================
# FINDAPHD DIRECT SEARCH
# ==============================================================================

def scrape_findaphd_search_direct(
    seen: set,
    run_seen: set
):

    if FINDAPHD_DIRECT_BLOCKED:
        return

    print(
        "\n  🔧 Attempting direct FindAPhD access..."
    )

    for kw in ACADEMIC_SEARCH_QUERIES:

        if FINDAPHD_DIRECT_BLOCKED:
            break

        print(
            f"\n  🔎 Direct query: {kw}"
        )

        for page in range(
            1,
            FINDAPHD_MAX_PAGES_PER_QUERY + 1
        ):

            if FINDAPHD_DIRECT_BLOCKED:
                break

            search_url = (
                build_findaphd_search_url(
                    kw,
                    page
                )
            )

            resp = findaphd_direct_get(
                search_url
            )

            if resp is None:
                break

            if resp.status_code != 200:

                print(
                    f"    [-] HTTP "
                    f"{resp.status_code}"
                )

                break

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            listings = (
                extract_findaphd_listing_links(
                    soup,
                    search_url
                )
            )

            if not listings:
                break

            print(
                f"    [>] Page {page}: "
                f"{len(listings)} links"
            )

            for listing in listings[
                :FINDAPHD_MAX_RESULTS_PER_QUERY
            ]:

                vacancy = (
                    collect_findaphd_listing(
                        listing,
                        soup
                    )
                )

                process_findaphd_candidate(
                    vacancy,
                    seen,
                    run_seen,
                    "DIRECT"
                )

                time.sleep(
                    FINDAPHD_REQUEST_DELAY
                )

            if len(listings) < 5:
                break


# ==============================================================================
# FINDAPHD DIRECT CATEGORY
# ==============================================================================

def scrape_findaphd_categories_direct(
    seen: set,
    run_seen: set
):

    if FINDAPHD_DIRECT_BLOCKED:
        return

    print(
        "\n  📚 Direct category discovery..."
    )

    for base_url in (
        FINDAPHD_CATEGORY_URLS
    ):

        if FINDAPHD_DIRECT_BLOCKED:
            break

        print(
            f"\n  📂 Category: {base_url}"
        )

        for page in range(
            1,
            FINDAPHD_MAX_PAGES_PER_QUERY + 1
        ):

            if FINDAPHD_DIRECT_BLOCKED:
                break

            page_url = (
                build_findaphd_category_url(
                    base_url,
                    page
                )
            )

            resp = findaphd_direct_get(
                page_url
            )

            if resp is None:
                break

            if resp.status_code != 200:
                break

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            listings = (
                extract_findaphd_listing_links(
                    soup,
                    page_url
                )
            )

            if not listings:
                break

            print(
                f"    [>] Page {page}: "
                f"{len(listings)} links"
            )

            for listing in listings:

                vacancy = (
                    collect_findaphd_listing(
                        listing,
                        soup
                    )
                )

                process_findaphd_candidate(
                    vacancy,
                    seen,
                    run_seen,
                    "CATEGORY"
                )

                time.sleep(
                    FINDAPHD_REQUEST_DELAY
                )


# ==============================================================================
# GOOGLE CUSTOM SEARCH
# ==============================================================================

def google_cse_available() -> bool:

    return bool(
        GOOGLE_CSE_API_KEY
        and GOOGLE_CSE_ID
    )


def google_custom_search(
    query: str,
    start: int = 1
):

    if not google_cse_available():

        return []

    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "start": start,
        "num": GOOGLE_SEARCH_RESULTS_PER_QUERY,
        "safe": "off",
    }

    try:

        response = session.get(
            GOOGLE_CSE_ENDPOINT,
            params=params,
            timeout=25
        )

        if response.status_code != 200:

            print(
                f"    [-] Google CSE HTTP "
                f"{response.status_code}"
            )

            try:
                print(
                    f"        {response.text[:300]}"
                )
            except Exception:
                pass

            return []

        data = response.json()

        if "error" in data:

            print(
                "    [-] Google CSE error: "
                f"{data['error']}"
            )

            return []

        return data.get(
            "items",
            []
        )

    except Exception as e:

        print(
            f"    [-] Google search error: "
            f"{e}"
        )

        return []


# ==============================================================================
# FINDAPHD SEARCH RESULT PARSER
# ==============================================================================

def clean_search_snippet(
    text: str
) -> str:

    if not text:
        return ""

    text = BeautifulSoup(
        text,
        "html.parser"
    ).get_text(
        " ",
        strip=True
    )

    return normalize_whitespace(
        text
    )


def vacancy_from_google_result(
    item: dict
) -> dict:

    url = canonicalize_url(
        item.get(
            "link",
            ""
        )
    )

    title = normalize_whitespace(
        item.get(
            "title",
            ""
        )
    )

    snippet = clean_search_snippet(
        item.get(
            "snippet",
            ""
        )
    )

    display_link = normalize_whitespace(
        item.get(
            "displayLink",
            ""
        )
    )

    html_snippet = clean_search_snippet(
        item.get(
            "htmlSnippet",
            ""
        )
    )

    combined_snippet = (
        snippet
        or html_snippet
    )

    text = (
        f"Title: {title}\n"
        f"Source: {display_link}\n"
        f"Search snippet: "
        f"{combined_snippet}\n"
        f"URL: {url}"
    )

    return {
        "url": url,
        "title": title,
        "text": text,
        "university": "",
        "location": "",
        "funding": "",
        "deadline": "",
    }


# ==============================================================================
# OPTIONAL DETAIL FETCH AFTER SEARCH DISCOVERY
# ==============================================================================

def enrich_findaphd_from_direct_page(
    vacancy: dict
) -> dict:

    global FINDAPHD_DIRECT_BLOCKED

    if FINDAPHD_DIRECT_BLOCKED:
        return vacancy

    url = vacancy.get(
        "url",
        ""
    )

    if not url:
        return vacancy

    try:

        resp = findaphd_direct_get(
            url
        )

        if resp is None:
            return vacancy

        if resp.status_code != 200:
            return vacancy

        soup = BeautifulSoup(
            resp.text,
            "html.parser"
        )

        title_elem = (
            soup.find("h1")
            or soup.find(
                "meta",
                property="og:title"
            )
        )

        title = ""

        if title_elem:

            if title_elem.name == "meta":

                title = normalize_whitespace(
                    title_elem.get(
                        "content",
                        ""
                    )
                )

            else:

                title = normalize_whitespace(
                    title_elem.get_text(
                        " ",
                        strip=True
                    )
                )

        meta_desc = soup.find(
            "meta",
            attrs={
                "name": "description"
            }
        )

        meta_description = ""

        if meta_desc:

            meta_description = (
                normalize_whitespace(
                    meta_desc.get(
                        "content",
                        ""
                    )
                )
            )

        main = (
            soup.find("main")
            or soup.find("article")
        )

        main_text = ""

        if main:

            main_text = normalize_whitespace(
                main.get_text(
                    " ",
                    strip=True
                )
            )

        if not title:
            title = vacancy.get(
                "title",
                ""
            )

        description = max(
            [
                vacancy.get(
                    "text",
                    ""
                ),
                meta_description,
                main_text
            ],
            key=len
        )

        vacancy["title"] = title

        vacancy["text"] = (
            f"Title: {title}\n"
            f"Description: "
            f"{description[:12000]}\n"
            f"URL: {url}"
        )

        return vacancy

    except Exception:
        return vacancy


# ==============================================================================
# FINDAPHD SEARCH-ENGINE DISCOVERY
# ==============================================================================

def scrape_findaphd_google_discovery(
    seen: set,
    run_seen: set
):

    if not google_cse_available():

        print(
            "\n  [!] Google CSE credentials "
            "not configured."
        )

        print(
            "  [!] FindAPhD search fallback "
            "will be unavailable."
        )

        print(
            "  [!] Add:"
        )

        print(
            "      GOOGLE_CSE_API_KEY"
        )

        print(
            "      GOOGLE_CSE_ID"
        )

        return

    print(
        "\n"
        + "-" * 65
    )

    print(
        "🔎 FINDAPHD INDEXED SEARCH DISCOVERY"
    )

    print(
        "-" * 65
    )

    discovered_urls = {}

    all_queries = (
        FINDAPHD_DISCOVERY_QUERIES
        + FINDAPHD_LATEST_DISCOVERY_QUERIES
    )

    for query in all_queries:

        print(
            f"\n  🔍 {query}"
        )

        items = google_custom_search(
            query,
            start=1
        )

        if not items:

            print(
                "    [-] No search results."
            )

            continue

        accepted = 0

        for item in items:

            url = canonicalize_url(
                item.get(
                    "link",
                    ""
                )
            )

            if not is_findaphd_vacancy_url(
                url
            ):
                continue

            title = normalize_whitespace(
                item.get(
                    "title",
                    ""
                )
            )

            snippet = clean_search_snippet(
                item.get(
                    "snippet",
                    ""
                )
            )

            vacancy = (
                vacancy_from_google_result(
                    item
                )
            )

            uid = make_findaphd_uid(
                url,
                title
            )

            if uid in run_seen:
                continue

            # Keep the strongest/longest result if
            # the same project appeared in multiple queries.
            old = discovered_urls.get(
                uid
            )

            if old:

                if len(
                    vacancy["text"]
                ) > len(
                    old["text"]
                ):

                    discovered_urls[
                        uid
                    ] = vacancy

                continue

            discovered_urls[
                uid
            ] = vacancy

            accepted += 1

        print(
            f"    [+] Accepted "
            f"{accepted} FindAPhD URLs"
        )

        time.sleep(
            GOOGLE_SEARCH_DELAY
        )

    print(
        "\n  📊 Unique FindAPhD URLs discovered: "
        f"{len(discovered_urls)}"
    )

    if not discovered_urls:
        return

    candidates_processed = 0

    for uid, vacancy in (
        discovered_urls.items()
    ):

        if (
            candidates_processed
            >= FINDAPHD_MAX_CANDIDATES_PER_RUN
        ):

            print(
                "  [!] FindAPhD candidate "
                "limit reached."
            )

            break

        if uid in seen:
            continue

        title = vacancy.get(
            "title",
            ""
        )

        snippet_text = vacancy.get(
            "text",
            ""
        )

        # Search-result-level prefilter.
        if not findaphd_soft_prefilter(
            title,
            snippet_text
        ):

            mark_as_seen(
                uid,
                seen
            )

            continue

        # Try direct detail only after discovery.
        #
        # If GitHub gets 403, enrich function will
        # simply keep the indexed snippet.
        if not FINDAPHD_DIRECT_BLOCKED:

            vacancy = (
                enrich_findaphd_from_direct_page(
                    vacancy
                )
            )

        process_findaphd_candidate(
            vacancy,
            seen,
            run_seen,
            "GOOGLE INDEX"
        )

        candidates_processed += 1

        time.sleep(
            FINDAPHD_REQUEST_DELAY
        )


# ==============================================================================
# FINDAPHD MAIN SCRAPER
# ==============================================================================

def scrape_findaphd_direct(
    seen: set
):

    global FINDAPHD_DIRECT_BLOCKED

    print(
        "\n"
        + "=" * 70
    )

    print(
        "🌍 FINDAPHD HYBRID SCRAPER"
    )

    print(
        "=" * 70
    )

    print(
        "Strategy:"
    )

    print(
        "  1. Direct FindAPhD access"
    )

    print(
        "  2. Detect 403"
    )

    print(
        "  3. Stop direct requests"
    )

    print(
        "  4. Search-engine indexed discovery"
    )

    print(
        "  5. Gemini evaluation"
    )

    run_seen = set()

    # --------------------------------------------------------------------------
    # DIRECT
    # --------------------------------------------------------------------------

    if FINDAPHD_TRY_DIRECT:

        scrape_findaphd_search_direct(
            seen,
            run_seen
        )

        if not FINDAPHD_DIRECT_BLOCKED:

            scrape_findaphd_categories_direct(
                seen,
                run_seen
            )

    # --------------------------------------------------------------------------
    # FALLBACK / PRIMARY DISCOVERY
    # --------------------------------------------------------------------------

    if FINDAPHD_DIRECT_BLOCKED:

        print(
            "\n"
            + "!" * 70
        )

        print(
            "⚠️ FindAPhD direct access is blocked."
        )

        print(
            "➡️ Switching to indexed search discovery."
        )

        print(
            "!" * 70
        )

    else:

        print(
            "\n  🔎 Running indexed discovery "
            "in addition to direct scraping..."
        )

    scrape_findaphd_google_discovery(
        seen,
        run_seen
    )

    print(
        "\n[✓] FindAPhD scan finished."
    )


# ==============================================================================
# EURAXESS
# ==============================================================================

EURAXESS_QUERIES = [
    "visual inertial odometry",
    "visual SLAM",
    "visual navigation",
    "sensor fusion",
    "GNSS positioning",
    "robot localization",
    "robot navigation",
    "computer vision robotics",
    "autonomous navigation",
    "field robotics",
    "embedded AI"
]


def scrape_euraxess_direct(
    seen: set
):

    print(
        "\n🇪🇺 Scanning EURAXESS..."
    )

    for kw in EURAXESS_QUERIES:

        try:

            search_url = (
                "https://euraxess.ec.europa.eu/jobs/search?"
                + urllib.parse.urlencode(
                    {"keywords": kw}
                )
            )

            resp = session.get(
                search_url,
                timeout=25
            )

            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            job_links = soup.find_all(
                "a",
                href=re.compile(
                    r"^/jobs/\d+"
                )
            )

            unique_jobs = {}

            for a in job_links:

                href = a.get(
                    "href"
                )

                title = normalize_whitespace(
                    a.get_text(
                        strip=True
                    )
                )

                if (
                    len(title) > 15
                    and href not in unique_jobs
                ):

                    unique_jobs[
                        href
                    ] = title

            if unique_jobs:

                print(
                    f"  [>] EURAXESS "
                    f"('{kw}'): "
                    f"{len(unique_jobs)} listings."
                )

            for href, title in list(
                unique_jobs.items()
            )[:6]:

                link = (
                    "https://euraxess.ec.europa.eu"
                    f"{href}"
                )

                if link in seen:
                    continue

                job_desc = ""

                try:

                    job_resp = session.get(
                        link,
                        timeout=15
                    )

                    if job_resp.status_code == 200:

                        jsoup = BeautifulSoup(
                            job_resp.text,
                            "html.parser"
                        )

                        desc_div = (
                            jsoup.find(
                                "div",
                                class_="node__content"
                            )
                            or jsoup.find(
                                "main"
                            )
                        )

                        if desc_div:

                            job_desc = (
                                desc_div.get_text(
                                    separator="\n",
                                    strip=True
                                )
                            )

                except Exception:
                    pass

                full_text = (
                    f"{title}\n"
                    f"{job_desc}"
                )[:6000]

                if not fast_prefilter(
                    full_text
                ):

                    mark_as_seen(
                        link,
                        seen
                    )

                    continue

                print(
                    f"[+] Evaluating EURAXESS: "
                    f"{title[:55]}..."
                )

                analysis = evaluate_with_gemini(
                    full_text
                )

                if analysis.get(
                    "is_relevant"
                ):

                    send_alert(
                        link,
                        analysis,
                        full_text
                    )

                else:

                    print(
                        f"    [-] Skipped: "
                        f"{analysis.get('reason', 'LLM filter')}"
                    )

                mark_as_seen(
                    link,
                    seen
                )

        except Exception as e:

            print(
                f"[-] EURAXESS error: {e}"
            )


# ==============================================================================
# RSS ACADEMIC PORTALS
# ==============================================================================

NEW_ACADEMIC_PORTALS = [

    "https://www.jobs.ac.uk/feeds/subject-areas/electrical-and-electronic-engineering",

    "https://www.jobs.ac.uk/feeds/subject-areas/computer-science",

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+robotics",

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+computer+vision",

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+navigation",

    "https://academicpositions.com/feed/rss?field=computer-science-electrical-engineering",

    "https://academicpositions.com/feed/rss?field=robotics",

    "https://academicpositions.com/feed/rss?field=artificial-intelligence",
]


def scrape_academic_rss_feeds(
    seen: set
):

    print(
        "\n🎓 Scanning academic RSS portals..."
    )

    for feed_url in NEW_ACADEMIC_PORTALS:

        try:

            resp = session.get(
                feed_url,
                timeout=25
            )

            if resp.status_code != 200:

                print(
                    f"  [-] RSS HTTP "
                    f"{resp.status_code}: "
                    f"{feed_url}"
                )

                continue

            feed = feedparser.parse(
                resp.content
            )

            feed_name = (
                feed_url.split("/")[-1]
                or feed_url
            )

            if feed.entries:

                print(
                    f"  [>] Feed '{feed_name}': "
                    f"{len(feed.entries)} entries."
                )

            for entry in feed.entries:

                link = entry.get(
                    "link",
                    ""
                )

                uid = entry.get(
                    "id",
                    link
                )

                if not uid:
                    continue

                if uid in seen:
                    continue

                if (
                    hasattr(
                        entry,
                        "published_parsed"
                    )
                    and entry.published_parsed
                ):

                    pub_dt = datetime(
                        *entry.published_parsed[:6],
                        tzinfo=timezone.utc
                    )

                    if not is_recent_date(
                        pub_dt
                    ):

                        mark_as_seen(
                            uid,
                            seen
                        )

                        continue

                content = (
                    f"{entry.get('title', '')}\n"
                    f"{entry.get('summary', '')}\n"
                    f"{entry.get('description', '')}"
                )

                if (
                    len(content) < 30
                    or not fast_prefilter(content)
                ):

                    mark_as_seen(
                        uid,
                        seen
                    )

                    continue

                print(
                    f"[+] Evaluating Academic: "
                    f"{entry.get('title', '')[:55]}..."
                )

                analysis = evaluate_with_gemini(
                    content
                )

                if analysis.get(
                    "is_relevant"
                ):

                    send_alert(
                        link,
                        analysis,
                        content
                    )

                    print(
                        "    [✓] MATCH CONFIRMED "
                        f"(Tier {analysis.get('tier')} - "
                        f"Score "
                        f"{analysis.get('confidence_score')}/10)"
                    )

                else:

                    print(
                        f"    [-] Skipped: "
                        f"{analysis.get('reason', 'LLM filter')}"
                    )

                mark_as_seen(
                    uid,
                    seen
                )

        except Exception as e:

            print(
                f"  [-] Academic feed error: "
                f"{feed_url[:60]}...: {e}"
            )


# ==============================================================================
# TEST TELEGRAM
# ==============================================================================

def test_telegram():

    endpoint = (
        "https://api.telegram.org/bot"
        f"{BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": "✅ PhD Finder Telegram test"
    }

    response = session.post(
        endpoint,
        json=payload,
        timeout=20
    )

    print(
        "STATUS:",
        response.status_code
    )

    print(
        "RESPONSE:",
        response.text
    )


# ==============================================================================
# MAIN
# ==============================================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "🚀 Running PhD Finder Agent"
    )

    print(
        "=" * 70
    )

    seen = load_seen()

    print(
        f"📂 Cached database has "
        f"{len(seen)} items."
    )

    # --------------------------------------------------------------------------
    # Google Search status
    # --------------------------------------------------------------------------

    print(
        "\n🔎 FindAPhD discovery configuration:"
    )

    if google_cse_available():

        print(
            "   [✓] Google CSE configured"
        )

    else:

        print(
            "   [!] Google CSE NOT configured"
        )

        print(
            "   → Direct FindAPhD only"
            " unless credentials are added."
        )

    # --------------------------------------------------------------------------
    # 1. Telegram
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "1️⃣ TELEGRAM CHANNELS"
    )

    print(
        "=" * 70
    )

    for ch in CHANNELS_TO_SCRAPE:

        scrape_telegram_channel_deep(
            ch,
            seen
        )

    # --------------------------------------------------------------------------
    # 2. FindAPhD
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "2️⃣ FINDAPHD"
    )

    print(
        "=" * 70
    )

    scrape_findaphd_direct(
        seen
    )

    # --------------------------------------------------------------------------
    # 3. EURAXESS
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "3️⃣ EURAXESS"
    )

    print(
        "=" * 70
    )

    scrape_euraxess_direct(
        seen
    )

    # --------------------------------------------------------------------------
    # 4. RSS
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "4️⃣ ACADEMIC RSS PORTALS"
    )

    print(
        "=" * 70
    )

    scrape_academic_rss_feeds(
        seen
    )

    # --------------------------------------------------------------------------
    # DONE
    # --------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "✅ RUN COMPLETE"
    )

    print(
        f"📂 Seen database now contains "
        f"{len(seen)} items."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    # Uncomment for Telegram testing.
    # test_telegram()

    main()
