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

# Support both naming conventions.
BOT_TOKEN = (
    os.getenv("TG_BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
)

CHAT_ID = (
    os.getenv("TG_CHAT_ID")
    or os.getenv("TELEGRAM_CHAT_ID")
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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

# FindAPhD
FINDAPHD_MAX_PAGES_PER_QUERY = 4
FINDAPHD_MAX_RESULTS_PER_QUERY = 20
FINDAPHD_REQUEST_DELAY = 1.2


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

# Direct category pages supplied by the user.
#
# The first two are the user's original URLs.
# Canonical category URLs are also included as fallback.
#
FINDAPHD_CATEGORY_URLS = [
    "https://www.findaphd.com/phds/engineering/?10M7o0",
    "https://www.findaphd.com/phds/computer-science/?10M7g0",

    # Canonical fallbacks
    "https://www.findaphd.com/phds/engineering/",
    "https://www.findaphd.com/phds/computer-science/",
]


# Search endpoint.
FINDAPHD_SEARCH_URL = "https://www.findaphd.com/phds/"


# Queries specifically selected for the candidate's target research.
ACADEMIC_SEARCH_QUERIES = [
    # --------------------------------------------------------------------------
    # Tier 1 — Direct research match
    # --------------------------------------------------------------------------
    "visual inertial odometry",
    "visual SLAM",
    "visual navigation",
    "sensor fusion localization",
    "GNSS positioning",
    "GNSS sensor fusion",
    "robot localization",
    "robot navigation",

    # --------------------------------------------------------------------------
    # Computer vision + robotics
    # --------------------------------------------------------------------------
    "computer vision robotics",
    "3D computer vision robotics",
    "visual perception autonomous robots",
    "visual localization",
    "camera IMU fusion",
    "LiDAR camera fusion",
    "visual odometry",

    # --------------------------------------------------------------------------
    # Estimation / optimization
    # --------------------------------------------------------------------------
    "factor graph optimization robotics",
    "state estimation robotics",
    "probabilistic robotics",
    "pose estimation",
    "trajectory estimation",

    # --------------------------------------------------------------------------
    # Autonomous systems
    # --------------------------------------------------------------------------
    "autonomous navigation",
    "field robotics",
    "mobile robot localization",
    "outdoor robotics",
    "autonomous vehicles",

    # --------------------------------------------------------------------------
    # ML relevant to robotics / perception
    # --------------------------------------------------------------------------
    "machine learning robotics",
    "deep learning computer vision robotics",
    "self supervised visual navigation",
    "uncertainty estimation robotics",

    # --------------------------------------------------------------------------
    # Embedded / Edge AI
    # --------------------------------------------------------------------------
    "embedded AI robotics",
    "edge AI computer vision",
    "GPU accelerated computer vision",
    "CUDA computer vision",
]


# ==============================================================================
# KEYWORD FILTERING
# ==============================================================================

BROAD_PATTERNS = [

    # ==========================================================================
    # Tier 1 — Direct Research Match
    # ==========================================================================

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

    r"sensor fusion",
    r"multi[- ]sensor fusion",
    r"multimodal sensor fusion",
    r"camera[- ]imu",
    r"vision[- ]imu",
    r"gnss[- ]imu",
    r"gnss/imu",
    r"gnss fusion",

    r"\bgnss\b",
    r"gps positioning",
    r"gnss positioning",
    r"satellite positioning",
    r"robust positioning",
    r"precise positioning",
    r"navigation system",
    r"localization",
    r"positioning",

    # ==========================================================================
    # Computer Vision
    # ==========================================================================

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

    # ==========================================================================
    # Robotics
    # ==========================================================================

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

    # ==========================================================================
    # ML / Deep Learning
    # ==========================================================================

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

    # ==========================================================================
    # Optimization / Estimation
    # ==========================================================================

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

    # ==========================================================================
    # Sensors
    # ==========================================================================

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

    # ==========================================================================
    # Embedded / Edge AI
    # ==========================================================================

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

    # ==========================================================================
    # Electronics / Hardware
    # ==========================================================================

    r"\bfpga\b",
    r"\bsdr\b",
    r"software defined radio",
    r"vhdl",
    r"verilog",
    r"digital hardware",
    r"digital systems",
    r"embedded hardware",

    # ==========================================================================
    # Position signals
    # ==========================================================================

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

    # Medical wet-lab
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

    # Policy / social sciences
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
- Visual-Inertial Odometry (VIO)
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

These should receive a lower score unless the vacancy connects them to
navigation, robotics, perception, localization, or sensor fusion.

Disqualifiers — REJECT:
- Pure mechanical engineering
- Pure robotic manipulation / robotic arms without perception or localization
- Prosthetics / exoskeletons unless strongly focused on sensing, perception or navigation
- Medical wet-lab biology
- Pure chemistry
- Pure materials science
- Pure theoretical mathematics
- Pure telecommunications without sensing/localization relevance
- Pure power electronics
- Pure control theory with no robotics/navigation application
- Pure software engineering with no research connection to the candidate's areas
- Visa advertisements
- Language courses

Important Evaluation Rules:
1. Do NOT consider a vacancy highly relevant merely because it contains
   generic terms such as "AI", "Machine Learning", "Python", or "Robotics".
2. Research topic and expected work are more important than the list of tools.
3. A robotics position is a strong match if it involves localization,
   perception, navigation, SLAM, sensor fusion, or autonomous systems.
4. A computer vision position is a strong match if the work involves
   robotics, navigation, localization, 3D vision, or autonomous systems.
5. A GNSS position is a strong match even if it does not mention deep learning.
6. An embedded/AI hardware position is relevant but secondary unless it
   involves robotics, computer vision, or edge AI.
7. Prefer funded PhD / doctoral research positions.
8. Penalize positions where the candidate would need a completely different
   research background.
9. Do not reject a position simply because one listed technology is missing
   from the candidate's current CV if the underlying research direction is
   strongly aligned.
10. Distinguish between "research fit" and "technical gap".
11. If the vacancy is only broadly about AI or computer science without a
    meaningful connection to robotics, vision, localization, navigation,
    sensing, or autonomous systems, score it low.
12. Prefer actual PhD research vacancies over generic postgraduate programs.

Return ONLY valid JSON:

{
  "is_relevant": true,
  "tier": 1,
  "confidence_score": 9,
  "title": "Short position title",
  "key_topics": ["topic1", "topic2", "topic3"],
  "research_fit": "1-line explanation of research alignment",
  "technical_gaps": ["missing skill 1"],
  "reason": "1-line final explanation"
}

Scoring:
- 9-10: Excellent fit; directly aligned with target research
- 7-8: Strong fit; closely related with manageable technical gaps
- 5-6: Potential fit; adjacent research area
- 3-4: Weak fit; significant mismatch
- 0-2: Not relevant

Tier:
- Tier 1 = Direct match
- Tier 2 = Strongly adjacent
- Tier 3 = Hardware / embedded / electronics
- Tier 4 = Weak / conditional

If the position is clearly irrelevant:
{
  "is_relevant": false,
  "tier": 4,
  "confidence_score": 1,
  "title": "Short position title",
  "key_topics": [],
  "research_fit": "No meaningful alignment with the candidate's research direction.",
  "technical_gaps": [],
  "reason": "Position is outside the candidate's target research areas."
}
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

    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str) -> str:
    """
    Normalize URLs for duplicate detection.
    Keeps meaningful query parameters but removes tracking parameters.
    """

    if not url:
        return ""

    url = url.strip()

    parsed = urlparse(url)

    # Remove fragments.
    parsed = parsed._replace(fragment="")

    # Remove common tracking parameters.
    query = parse_qs(parsed.query, keep_blank_values=True)

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

    new_query = urlencode(clean_query, doseq=True)

    parsed = parsed._replace(query=new_query)

    result = urlunparse(parsed)

    # Remove trailing slash except root.
    if result.endswith("/") and parsed.path not in ("", "/"):
        result = result[:-1]

    return result


def make_findaphd_uid(url: str, title: str = "") -> str:
    canonical = canonicalize_url(url)

    if canonical:
        return canonical

    return "findaphd:" + normalize_whitespace(title).lower()


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
    """
    Soft relevance score.

    This is intentionally NOT used as a hard gate for FindAPhD.
    """

    lower_t = text.lower()

    score = 0

    for pattern in BROAD_PATTERNS:
        if re.search(pattern, lower_t):
            score += 1

    return score


def fast_prefilter(text: str) -> bool:
    """
    General filter for Telegram/RSS.

    FindAPhD uses a softer version because its search snippets can be short.
    """

    if not text:
        return False

    if contains_excluded_pattern(text):
        return False

    return keyword_score(text) > 0


def findaphd_soft_prefilter(title: str, description: str) -> bool:
    """
    FindAPhD-specific soft filter.

    We don't require an exact VIO/SLAM keyword because some good vacancies
    have generic titles such as:
        Autonomous Systems
        Robot Perception
        Localization for Mobile Robots

    If the title/description has at least one relevant signal, pass it to Gemini.
    """

    text = f"{title}\n{description}"

    if contains_excluded_pattern(text):
        return False

    # Strong signals.
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
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)

        return (now - dt_obj) <= timedelta(days=MAX_POST_AGE_DAYS)

    except Exception:
        return True


def parse_iso_time(time_str: str):
    try:
        return datetime.fromisoformat(
            time_str.replace("Z", "+00:00")
        )
    except Exception:
        return None


# ==============================================================================
# GEMINI JSON
# ==============================================================================

def extract_clean_json(raw_text: str) -> dict:
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
                raw_text[start_idx:end_idx + 1]
            )

    except Exception:
        pass

    return {
        "is_relevant": False,
        "reason": "Failed to parse JSON output"
    }


def evaluate_with_gemini(text: str) -> dict:

    models_to_try = [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite"
    ]

    for attempt in range(3):

        for model_name in models_to_try:

            try:

                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=(
                        "Evaluate this PhD post:\n\n"
                        f"{text[:6000]}"
                    ),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE
                            ),
                        ]
                    )
                )

                time.sleep(4.2)

                if (
                    not response.candidates
                    or not response.candidates[0].content.parts
                ):
                    return {
                        "is_relevant": False,
                        "reason": "Blocked by Gemini Safety Filters"
                    }

                return extract_clean_json(response.text)

            except Exception as e:

                err_str = str(e)

                if (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                ):
                    print(
                        "[!] Gemini rate limit. Waiting 25 seconds..."
                    )

                    time.sleep(25)
                    break

                else:
                    print(
                        f"[!] Gemini error ({model_name}): "
                        f"{err_str[:150]}"
                    )

                    continue

    return {
        "is_relevant": False,
        "reason": "API error after retries"
    }


# ==============================================================================
# TELEGRAM ALERT
# ==============================================================================

def send_alert(url: str, analysis: dict, snippet: str):

    if not isinstance(analysis, dict):
        return False

    tier_badges = {
        1: "🔥 Tier 1",
        2: "⚡ Tier 2",
        3: "🛠 Tier 3"
    }

    tier_val = analysis.get("tier", 2)
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
        f"{snippet[:500]}..."
    )

    endpoint = (
        f"https://api.telegram.org/bot"
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
            f"[Telegram] HTTP {response.status_code}"
        )

        if response.status_code != 200:
            print(
                "[-] Telegram failed to send message!"
            )
            print(response.text)
            return False

        result = response.json()

        if not result.get("ok"):
            print(
                f"[-] Telegram API error: {result}"
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

    for _ in range(PAGES_PER_CHANNEL):

        try:

            r = session.get(
                current_url,
                timeout=25
            )

            if r.status_code != 200:
                print(
                    f"[-] Telegram HTTP {r.status_code}"
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

                raw_num = msg_id.split("/")[-1]

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
                        time_tag.get("datetime")
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

                if analysis.get("is_relevant"):

                    send_alert(
                        f"https://t.me/{msg_id}",
                        analysis,
                        text
                    )

                    print(
                        "    [✓] MATCH CONFIRMED "
                        f"(Tier {analysis.get('tier')} - "
                        f"Score {analysis.get('confidence_score')}/10)"
                    )

                else:

                    print(
                        "    [-] Skipped: "
                        f"{analysis.get('reason', 'Filtered by LLM')}"
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
# FINDAPHD HELPERS
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


def is_findaphd_vacancy_url(url: str) -> bool:

    if not url:
        return False

    parsed = urlparse(url)

    if "findaphd.com" not in parsed.netloc.lower():
        return False

    path = parsed.path.lower()

    # Actual vacancy pages usually contain /phds/
    # while category/search pages also contain /phds/.
    # We reject obvious navigation URLs.
    bad_fragments = [
        "/phds/engineering",
        "/phds/computer-science",
        "/phds/",
        "/search",
        "/subjects/",
        "/universities/",
    ]

    if path in (
        "",
        "/",
        "/phds",
        "/phds/"
    ):
        return False

    # A detail page normally has additional path depth.
    if path.startswith("/phds/"):

        remaining = path[len("/phds/"):]

        if "/" not in remaining:
            return False

        return True

    return False


def extract_findaphd_listing_links(
    soup: BeautifulSoup,
    base_url: str
):
    """
    Extract candidate vacancy links without depending on one CSS class.

    FindAPhD's frontend can change CSS class names, so we use URL structure
    and surrounding semantic content instead.
    """

    results = {}

    for a in soup.find_all("a", href=True):

        href = a.get("href", "").strip()

        if not href:
            continue

        absolute_url = urljoin(
            base_url,
            href
        )

        absolute_url = canonicalize_url(
            absolute_url
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

        # Ignore generic navigation.
        lower_title = title.lower()

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

        if lower_title in navigation_words:
            continue

        if absolute_url not in results:

            results[absolute_url] = {
                "url": absolute_url,
                "anchor_title": title
            }

    return list(results.values())


def extract_findaphd_card_for_link(
    soup: BeautifulSoup,
    link: str
):
    """
    Find the closest meaningful container around a vacancy link.

    We intentionally don't depend on a single class name.
    """

    anchor = soup.find(
        "a",
        href=lambda x: (
            x
            and canonicalize_url(
                urljoin(
                    "https://www.findaphd.com",
                    x
                )
            ) == canonicalize_url(link)
        )
    )

    if not anchor:
        return ""

    # Walk upward and select a reasonably sized container.
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

            # Prefer containers that have multiple links or heading-like text.
            if (
                len(parent.find_all("a")) >= 2
                or parent.find(["h1", "h2", "h3", "h4"])
            ):
                return text

    return normalize_whitespace(
        anchor.parent.get_text(
            " ",
            strip=True
        )
    )


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
                f"{response.status_code}: {url}"
            )

            return result

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # ----------------------------------------------------------------------
        # TITLE
        # ----------------------------------------------------------------------

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
                result["title"] = normalize_whitespace(
                    title_elem.get_text(
                        " ",
                        strip=True
                    )
                )

        # ----------------------------------------------------------------------
        # META DESCRIPTION
        # ----------------------------------------------------------------------

        meta_desc = soup.find(
            "meta",
            attrs={
                "name": "description"
            }
        )

        if meta_desc:

            result["description"] = normalize_whitespace(
                meta_desc.get(
                    "content",
                    ""
                )
            )

        # ----------------------------------------------------------------------
        # MAIN CONTENT
        # ----------------------------------------------------------------------

        main = (
            soup.find("main")
            or soup.find(
                "article"
            )
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

                # Don't allow absurdly huge pages into Gemini.
                result["description"] = main_text[:12000]

        # ----------------------------------------------------------------------
        # STRUCTURED DATA
        # ----------------------------------------------------------------------

        for script in soup.find_all(
            "script",
            type="application/ld+json"
        ):

            try:

                raw = script.string

                if not raw:
                    continue

                data = json.loads(raw)

                if isinstance(data, dict):

                    if not result["title"]:
                        result["title"] = str(
                            data.get(
                                "name",
                                ""
                            )
                        ).strip()

                    description = str(
                        data.get(
                            "description",
                            ""
                        )
                    ).strip()

                    if (
                        description
                        and len(description)
                        > len(result["description"])
                    ):
                        result["description"] = (
                            description
                        )

            except Exception:
                continue

        # ----------------------------------------------------------------------
        # UNIVERSITY / INSTITUTION
        # ----------------------------------------------------------------------

        university_patterns = [
            r"university",
            r"institution",
            r"organisation",
            r"organization",
            r"department",
        ]

        for pattern in university_patterns:

            elem = soup.find(
                string=re.compile(
                    pattern,
                    re.I
                )
            )

            if elem:

                parent_text = normalize_whitespace(
                    elem.parent.get_text(
                        " ",
                        strip=True
                    )
                )

                if 10 <= len(parent_text) <= 250:

                    result["university"] = (
                        parent_text
                    )

                    break

        # ----------------------------------------------------------------------
        # LABEL-BASED EXTRACTION
        # ----------------------------------------------------------------------

        page_text = normalize_whitespace(
            soup.get_text(
                " ",
                strip=True
            )
        )

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

        for field, patterns in label_patterns.items():

            for pattern in patterns:

                match = re.search(
                    rf"{pattern}\s*[:\-]?\s*(.{{5,200}})",
                    page_text,
                    re.I
                )

                if match:

                    value = normalize_whitespace(
                        match.group(1)
                    )

                    result[field] = value

                    break

        return result

    except Exception as e:

        print(
            f"    [-] FindAPhD detail error: "
            f"{e}"
        )

        return result


def collect_findaphd_listing(
    listing,
    page_soup: BeautifulSoup
):

    url = listing["url"]
    anchor_title = listing["anchor_title"]

    card_text = extract_findaphd_card_for_link(
        page_soup,
        url
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
        f"University: {detail['university']}\n"
        f"Location: {detail['location']}\n"
        f"Funding: {detail['funding']}\n"
        f"Deadline: {detail['deadline']}\n"
        f"Description: {description}"
    )

    return {
        "url": url,
        "title": title,
        "text": full_text,
        "university": detail["university"],
        "location": detail["location"],
        "funding": detail["funding"],
        "deadline": detail["deadline"],
    }


# ==============================================================================
# FINDAPHD SCRAPER
# ==============================================================================

def scrape_findaphd_search(
    seen: set
):

    print(
        "\n🌍 Scanning FindAPhD keyword searches..."
    )

    global_seen_this_run = set()

    for kw in ACADEMIC_SEARCH_QUERIES:

        print(
            f"\n  🔎 FindAPhD query: {kw}"
        )

        for page in range(
            1,
            FINDAPHD_MAX_PAGES_PER_QUERY + 1
        ):

            search_url = build_findaphd_search_url(
                kw,
                page
            )

            try:

                resp = session.get(
                    search_url,
                    timeout=25
                )

            except Exception as e:

                print(
                    f"    [-] Request error: {e}"
                )

                break

            if resp.status_code != 200:

                print(
                    f"    [-] HTTP {resp.status_code}"
                )

                break

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            listings = extract_findaphd_listing_links(
                soup,
                search_url
            )

            if not listings:

                if page == 1:
                    print(
                        "    [-] No vacancy links found."
                    )

                break

            print(
                f"    [>] Page {page}: "
                f"{len(listings)} candidate links"
            )

            processed_on_page = 0

            for listing in listings[
                :FINDAPHD_MAX_RESULTS_PER_QUERY
            ]:

                url = listing["url"]

                uid = make_findaphd_uid(
                    url,
                    listing["anchor_title"]
                )

                if (
                    uid in seen
                    or uid in global_seen_this_run
                ):
                    continue

                global_seen_this_run.add(uid)

                vacancy = collect_findaphd_listing(
                    listing,
                    soup
                )

                title = vacancy["title"]
                full_text = vacancy["text"]

                if len(full_text) < 50:

                    mark_as_seen(
                        uid,
                        seen
                    )

                    continue

                # Soft filter specifically for FindAPhD.
                if not findaphd_soft_prefilter(
                    title,
                    full_text
                ):

                    mark_as_seen(
                        uid,
                        seen
                    )

                    continue

                print(
                    f"    [+] Evaluating: "
                    f"{title[:80]}..."
                )

                analysis = evaluate_with_gemini(
                    full_text
                )

                if analysis.get(
                    "is_relevant"
                ):

                    send_alert(
                        url,
                        analysis,
                        full_text
                    )

                    print(
                        "        [✓] MATCH "
                        f"Tier {analysis.get('tier')} "
                        f"/ Score "
                        f"{analysis.get('confidence_score')}/10"
                    )

                else:

                    print(
                        "        [-] Rejected: "
                        f"{analysis.get('reason', 'LLM filter')}"
                    )

                mark_as_seen(
                    uid,
                    seen
                )

                processed_on_page += 1

                time.sleep(
                    FINDAPHD_REQUEST_DELAY
                )

            # If the page returned fewer listings than expected,
            # it is probably the final page.
            if len(listings) < 5:
                break

    print(
        "\n[✓] FindAPhD keyword scan complete."
    )


def scrape_findaphd_categories(
    seen: set
):

    print(
        "\n📚 Scanning FindAPhD category pages..."
    )

    global_seen_this_run = set()

    for base_url in FINDAPHD_CATEGORY_URLS:

        print(
            f"\n  📂 Category: {base_url}"
        )

        for page in range(
            1,
            FINDAPHD_MAX_PAGES_PER_QUERY + 1
        ):

            page_url = build_findaphd_category_url(
                base_url,
                page
            )

            try:

                resp = session.get(
                    page_url,
                    timeout=25
                )

            except Exception as e:

                print(
                    f"    [-] Category request error: {e}"
                )

                break

            if resp.status_code != 200:

                print(
                    f"    [-] HTTP {resp.status_code}"
                )

                # Don't keep requesting subsequent pages
                # if the category itself is unavailable.
                break

            soup = BeautifulSoup(
                resp.text,
                "html.parser"
            )

            listings = extract_findaphd_listing_links(
                soup,
                page_url
            )

            if not listings:
                break

            print(
                f"    [>] Page {page}: "
                f"{len(listings)} candidate links"
            )

            for listing in listings:

                url = listing["url"]

                uid = make_findaphd_uid(
                    url,
                    listing["anchor_title"]
                )

                if (
                    uid in seen
                    or uid in global_seen_this_run
                ):
                    continue

                global_seen_this_run.add(uid)

                vacancy = collect_findaphd_listing(
                    listing,
                    soup
                )

                title = vacancy["title"]
                full_text = vacancy["text"]

                if len(full_text) < 50:

                    mark_as_seen(
                        uid,
                        seen
                    )

                    continue

                if not findaphd_soft_prefilter(
                    title,
                    full_text
                ):

                    mark_as_seen(
                        uid,
                        seen
                    )

                    continue

                print(
                    f"    [+] Evaluating category vacancy: "
                    f"{title[:80]}..."
                )

                analysis = evaluate_with_gemini(
                    full_text
                )

                if analysis.get(
                    "is_relevant"
                ):

                    send_alert(
                        url,
                        analysis,
                        full_text
                    )

                    print(
                        "        [✓] CATEGORY MATCH "
                        f"(Tier {analysis.get('tier')}, "
                        f"Score {analysis.get('confidence_score')}/10)"
                    )

                else:

                    print(
                        "        [-] Rejected: "
                        f"{analysis.get('reason', 'LLM filter')}"
                    )

                mark_as_seen(
                    uid,
                    seen
                )

                time.sleep(
                    FINDAPHD_REQUEST_DELAY
                )

    print(
        "\n[✓] FindAPhD category scan complete."
    )


def scrape_findaphd_direct(
    seen: set
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "🌍 FINDAPHD ROBUST SCRAPER"
    )

    print(
        "=" * 70
    )

    # 1. Targeted keyword search.
    scrape_findaphd_search(
        seen
    )

    # 2. Category fallback / discovery.
    scrape_findaphd_categories(
        seen
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

                    unique_jobs[href] = title

            if unique_jobs:

                print(
                    f"  [>] EURAXESS ('{kw}'): "
                    f"{len(unique_jobs)} listings."
                )

            for href, title in list(
                unique_jobs.items()
            )[:6]:

                link = (
                    f"https://euraxess.ec.europa.eu"
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
                            or jsoup.find("main")
                        )

                        if desc_div:

                            job_desc = desc_div.get_text(
                                separator="\n",
                                strip=True
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
                    f"[+] Evaluating EURAXESS vacancy: "
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

    # --------------------------------------------------------------------------
    # UK
    # --------------------------------------------------------------------------

    "https://www.jobs.ac.uk/feeds/subject-areas/electrical-and-electronic-engineering",

    "https://www.jobs.ac.uk/feeds/subject-areas/computer-science",

    # --------------------------------------------------------------------------
    # Netherlands
    # --------------------------------------------------------------------------

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+robotics",

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+computer+vision",

    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+navigation",

    # --------------------------------------------------------------------------
    # European / International
    # --------------------------------------------------------------------------

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
                    f"[+] Evaluating Academic Vacancy: "
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
                        f"Score {analysis.get('confidence_score')}/10)"
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
                f"  [-] Academic feed error for "
                f"'{feed_url[:60]}...': {e}"
            )


# ==============================================================================
# TEST TELEGRAM
# ==============================================================================

def test_telegram():

    endpoint = (
        f"https://api.telegram.org/bot"
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
    # 1. Telegram Channels
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
    # 4. RSS Academic Portals
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
