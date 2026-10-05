import os
import json
import time
import re
import warnings
import urllib.parse
from datetime import datetime, timezone, timedelta
import requests
import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

load_dotenv()

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATA_DIR = "data"
SEEN_FILE = os.path.join(DATA_DIR, "seen_posts.json")
os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(SEEN_FILE):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

MAX_POST_AGE_DAYS = 60
PAGES_PER_CHANNEL = 5

if HAS_CLOUDSCRAPER:
    session = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'linux', 'mobile': False})
else:
    session = requests.Session()

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
})

from google import genai
from google.genai import types

ai_client = genai.Client(api_key=GEMINI_API_KEY)

CHANNELS_TO_SCRAPE = [
    "expertapply", "ApplyIR2UK", "applyforfree", "pargarwiki",
    "applyclub", "ApplyDaily", "computer_phd_apply", "EuropeanPhD",
    "PargarPositions"
]

CHANNELS_TO_SCRAPE = [
    "expertapply", "ApplyIR2UK", "applyforfree", "pargarwiki",
    "applyclub", "ApplyDaily", "computer_phd_apply", "EuropeanPhD",
    "PargarPositions"
]

# Targeted patterns for the candidate's research profile
BROAD_PATTERNS = [
    # ===== Tier 1: Direct Research Match =====
    r"\bv[io]\b", r"visual[- ]inertial", r"visual[- ]inertial odometry",
    r"\bvio\b", r"visual odometry",
    r"visual[- ]inertial navigation",
    r"visual navigation",
    
    r"\bslam\b", r"visual slam", r"visual[- ]inertial slam",
    r"simultaneous localization and mapping",
    r"localization and mapping",
    
    r"sensor fusion", r"multi[- ]sensor fusion",
    r"multimodal sensor fusion",
    r"camera[- ]imu", r"vision[- ]imu",
    r"gnss[- ]imu", r"gnss/imu", r"gnss fusion",
    
    r"\bgnss\b", r"gps positioning", r"gnss positioning",
    r"satellite positioning", r"robust positioning",
    r"precise positioning", r"navigation system",
    r"localization", r"positioning",
    
    # ===== Computer Vision / Robotics =====
    r"computer vision", r"machine vision",
    r"3d vision", r"3d computer vision",
    r"image processing", r"video processing",
    r"object detection", r"object tracking",
    r"multi[- ]object tracking",
    r"visual perception", r"scene understanding",
    r"depth estimation", r"stereo vision",
    r"visual localization", r"place recognition",
    r"camera[- ]based navigation",
    
    r"robot localization", r"robot navigation",
    r"autonomous navigation", r"autonomous systems",
    r"mobile robot", r"mobile robotics",
    r"field robotics", r"outdoor robotics",
    r"autonomous robot", r"robot perception",
    
    # ===== ML / Deep Learning relevant to above =====
    r"machine learning", r"deep learning",
    r"neural network", r"transformer",
    r"self[- ]supervised learning",
    r"representation learning",
    r"domain adaptation",
    r"domain generalization",
    r"uncertainty estimation",
    r"probabilistic learning",
    r"graph neural network", r"\bgnn\b",
    
    # ===== Optimization / Estimation =====
    r"factor graph", r"factor graph optimization",
    r"graph[- ]based optimization",
    r"nonlinear optimization",
    r"state estimation",
    r"pose estimation",
    r"trajectory estimation",
    r"kalman filter", r"extended kalman",
    r"particle filter",
    r"probabilistic robotics",
    
    # ===== Sensor / Navigation Technologies =====
    r"\bimu\b", r"inertial measurement",
    r"inertial navigation", r"inertial sensing",
    r"lidar", r"lidar[- ]based",
    r"camera[- ]lidar", r"lidar[- ]camera",
    r"gnss", r"rtk", r"pseudorange",
    
    # ===== Embedded / Edge AI =====
    r"embedded ai", r"edge ai", r"edge computing",
    r"embedded systems", r"embedded software",
    r"embedded machine learning",
    r"tinyml",
    r"ai accelerator", r"hardware acceleration",
    r"gpu acceleration", r"cuda",
    r"nvidia jetson", r"edge robotics",
    
    r"ros\b", r"ros2", r"robot operating system",
    
    # ===== Electronics / Hardware (lower priority) =====
    r"\bfpga\b", r"\bsdr\b", r"software defined radio",
    r"vhdl", r"verilog",
    r"digital hardware", r"digital systems",
    r"embedded hardware",
    
    # ===== Persian =====
    r"ناوبری", r"ناوبری هوشمند",
    r"موقعیت[- ]?یابی", r"مکان[- ]?یابی",
    r"بینایی ماشین", r"بینایی کامپیوتر",
    r"پردازش تصویر", r"پردازش ویدئو",
    r"یادگیری ماشین", r"یادگیری عمیق",
    r"ادغام سنسورها", r"همجوشی سنسورها",
    r"فیوژن سنسورها",
    r"اسلم", r"ناوبری ربات",
    r"ربات خودران", r"سیستم نهفته",
    r"هوش مصنوعی لبه",
    
    # ===== Application / Position signals =====
    r"\bphd\b", r"ph\.d", r"doctoral",
    r"phd position", r"phd candidate",
    r"doctoral position", r"research assistant",
    r"funded phd", r"fully funded",
    r"funded position",
    r"بورسیه", r"فاند", r"پوزیشن", r"دکتری"
]

EXCLUDE_PATTERNS = [
    # Generic irrelevant domains
    r"ویزای همسر", r"ویزای کاری", r"تعیین وقت سفارت",
    r"کلاس زبان", r"آموزش آیلتس",
    r"ielts class", r"immigration lawyer",
    
    # Medical wet-lab / unrelated biology
    r"wet[- ]lab", r"molecular biology",
    r"cell culture", r"gene expression",
    r"genomics", r"proteomics",
    r"immunotherapy", r"oncology",
    r"tissue staining", r"cardiovascular",
    
    # Chemistry / materials unrelated to AI or sensing
    r"organic chemistry", r"inorganic chemistry",
    r"polymer chemistry",
    
    # Purely unrelated robotics
    r"robotic arm manipulation",
    r"robot[- ]assisted surgery",
    r"prosthetics",
    r"exoskeleton",
    
    # Pure mechanical engineering
    r"pure mechanical design",
    r"fluid mechanics",
    r"thermodynamics",
    
    # Non-relevant policy / social sciences
    r"public policy",
    r"social sciences",
    r"political science"
]

ACADEMIC_SEARCH_QUERIES = [
    # Direct match
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
    
    # Estimation / optimization
    "factor graph optimization robotics",
    "state estimation robotics",
    "probabilistic robotics",
    "pose estimation",
    
    # Autonomous systems
    "autonomous navigation",
    "field robotics",
    "mobile robot localization",
    "outdoor robotics",
    
    # Embedded / AI
    "embedded AI robotics",
    "edge AI computer vision",
    "GPU accelerated computer vision",
    "CUDA computer vision"
]

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

NEW_ACADEMIC_PORTALS = [
    # UK Universities
    "https://www.jobs.ac.uk/feeds/subject-areas/electrical-and-electronic-engineering",
    "https://www.jobs.ac.uk/feeds/subject-areas/computer-science",
    
    # Netherlands
    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+robotics",
    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+computer+vision",
    "https://www.academictransfer.com/en/jobs/rss/?q=PhD+navigation",
    
    # European / International
    "https://academicpositions.com/feed/rss?field=computer-science-electrical-engineering",
    "https://academicpositions.com/feed/rss?field=robotics",
    "https://academicpositions.com/feed/rss?field=artificial-intelligence"
]

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
def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def mark_as_seen(uid: str, seen: set):
    seen.add(uid)
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(list(seen), f)
    except Exception:
        pass

def fast_prefilter(text: str) -> bool:
    lower_t = text.lower()
    if any(re.search(pat, lower_t) for pat in EXCLUDE_PATTERNS):
        return False
    return any(re.search(pat, lower_t) for pat in BROAD_PATTERNS)

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
        return datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    except Exception:
        return None

def extract_clean_json(raw_text: str) -> dict:
    try:
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            return json.loads(raw_text[start_idx:end_idx + 1])
    except Exception:
        pass
    return {"is_relevant": False, "reason": "Failed to parse JSON output"}

def evaluate_with_gemini(text: str) -> dict:
    models_to_try = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
    for attempt in range(3):
        for model_name in models_to_try:
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=f"Evaluate this PhD post:\n\n{text[:2000]}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                )
                time.sleep(4.2)
                if not response.candidates or not response.candidates[0].content.parts:
                    return {"is_relevant": False, "reason": "Blocked by Gemini Safety Filters"}
                return extract_clean_json(response.text)
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(25)
                    break
                else:
                    continue
    return {"is_relevant": False, "reason": "API error after retries"}

def send_alert(url: str, analysis: dict, snippet: str):
    if not isinstance(analysis, dict):
        return
    tier_badges = {1: "🔥 Tier 1", 2: "⚡ Tier 2", 3: "🛠 Tier 3"}
    tier_val = analysis.get("tier", 2)
    label = tier_badges.get(tier_val, "Relevant Opportunity")
    
    score = analysis.get("confidence_score", "N/A")
    title = analysis.get("title", "PhD Opportunity")
    topics_list = analysis.get("key_topics", [])
    topics = ", ".join(topics_list) if isinstance(topics_list, list) else str(topics_list)
    reason = analysis.get("reason", "Matches research profile.")

    msg = (
        f"🎯 *New PhD Match!*\n"
        f"*Category:* {label}\n"
        f"*Score:* {score}/10\n"
        f"*Title:* {title}\n"
        f"*Topics:* `{topics}`\n\n"
        f"💡 *Reason:* {reason}\n\n"
        f"🔗 [Open Vacancy / Post]({url})\n"
        f"────────────────────\n"
        f"📝 *Preview:*\n{snippet[:250]}..."
    )
    
    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False
    }
    try:
        session.post(endpoint, json=payload, timeout=20)
    except Exception as e:
        print(f"[-] Alert sending error: {e}")

def scrape_telegram_channel_deep(channel: str, seen: set):
    print(f"\n📡 Scanning @{channel}...")
    current_url = f"https://t.me/s/{channel}"
    
    for _ in range(PAGES_PER_CHANNEL):
        try:
            r = session.get(current_url, timeout=25)
            if r.status_code != 200:
                break
            soup = BeautifulSoup(r.text, "html.parser")
            messages = soup.find_all("div", class_="tgme_widget_message")
            if not messages:
                break
            
            oldest_id_on_page = None
            for msg in reversed(messages):
                msg_id = msg.get("data-post")
                if not msg_id:
                    continue
                
                raw_num = msg_id.split("/")[-1]
                if raw_num.isdigit():
                    oldest_id_on_page = int(raw_num) if oldest_id_on_page is None else min(oldest_id_on_page, int(raw_num))
                
                if msg_id in seen:
                    continue
                
                time_tag = msg.find("time")
                if time_tag and time_tag.get("datetime"):
                    dt = parse_iso_time(time_tag.get("datetime"))
                    if dt and not is_recent_date(dt):
                        mark_as_seen(msg_id, seen)
                        continue
                        
                text_elem = msg.find("div", class_="tgme_widget_message_text")
                if not text_elem:
                    mark_as_seen(msg_id, seen)
                    continue
                    
                text = text_elem.get_text(separator="\n").strip()
                if len(text) < 40 or not fast_prefilter(text):
                    mark_as_seen(msg_id, seen)
                    continue
                    
                print(f"[+] Evaluating Telegram post: {msg_id}")
                analysis = evaluate_with_gemini(text)
                
                if analysis.get("is_relevant"):
                    send_alert(f"https://t.me/{msg_id}", analysis, text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')} - Score {analysis.get('confidence_score')}/10) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                
                mark_as_seen(msg_id, seen)
                
            if oldest_id_on_page:
                current_url = f"https://t.me/s/{channel}?before={oldest_id_on_page}"
            else:
                break
        except Exception as e:
            print(f"[-] @{channel} issue: {e}")
            break

def scrape_findaphd_direct(seen: set):
    print("\n🌍 Scanning FindAPhD.com directly...")
    for kw in ACADEMIC_SEARCH_QUERIES:
        try:
            search_url = f"https://www.findaphd.com/phds/?Keywords={urllib.parse.quote_plus(kw)}"
            resp = session.get(search_url, timeout=25)
            if resp.status_code != 200:
                continue
            
            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.find_all("div", class_="phd-result-row") or soup.find_all("div", class_="w-100")
            valid_cards = [c for c in cards if c.find("a", class_="apply-link") or c.find("h3")]
            
            if valid_cards:
                print(f"  [>] FindAPhD ('{kw}'): Found {len(valid_cards)} listings.")
            
            for card in valid_cards[:6]:
                title_elem = card.find("a", class_="apply-link") or card.find("h3") or card.find("a")
                if not title_elem:
                    continue
                
                title = title_elem.get_text(strip=True)
                link = title_elem.get("href", "")
                if link and not link.startswith("http"):
                    link = f"https://www.findaphd.com{link}"
                
                desc_elem = card.find("div", class_="desc") or card.find("div", class_="phd-result-row__desc")
                desc = desc_elem.get_text(strip=True) if desc_elem else ""
                
                full_text = f"{title}\n{desc}"
                uid = link or title
                
                if uid in seen or len(full_text) < 30 or not fast_prefilter(full_text):
                    mark_as_seen(uid, seen)
                    continue

                print(f"[+] Evaluating FindAPhD vacancy: {title[:55]}...")
                analysis = evaluate_with_gemini(full_text)
                if analysis.get("is_relevant"):
                    send_alert(link, analysis, full_text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')}) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                    
                mark_as_seen(uid, seen)
        except Exception:
            pass

def scrape_euraxess_direct(seen: set):
    print("\n🇪🇺 Scanning EURAXESS (EU MSCA & Funded Positions)...")
    for kw in EURAXESS_QUERIES:
        try:
            search_url = f"https://euraxess.ec.europa.eu/jobs/search?keywords={urllib.parse.quote_plus(kw)}"
            resp = session.get(search_url, timeout=25)
            if resp.status_code != 200:
                continue
            
            soup = BeautifulSoup(resp.text, "html.parser")
            job_links = soup.find_all("a", href=re.compile(r"^/jobs/\d+"))
            
            unique_jobs = {}
            for a in job_links:
                href = a.get("href")
                title = a.get_text(strip=True)
                if len(title) > 15 and href not in unique_jobs:
                    unique_jobs[href] = title
            
            if unique_jobs:
                print(f"  [>] EURAXESS ('{kw}'): Found {len(unique_jobs)} listings.")
            
            for href, title in list(unique_jobs.items())[:6]:
                link = f"https://euraxess.ec.europa.eu{href}"
                if link in seen:
                    continue
                
                job_desc = ""
                try:
                    job_resp = session.get(link, timeout=15)
                    if job_resp.status_code == 200:
                        jsoup = BeautifulSoup(job_resp.text, "html.parser")
                        desc_div = jsoup.find("div", class_="node__content") or jsoup.find("main")
                        if desc_div:
                            job_desc = desc_div.get_text(separator="\n", strip=True)
                except Exception:
                    pass
                    
                full_text = f"{title}\n{job_desc}"[:2000]
                
                if not fast_prefilter(full_text):
                    mark_as_seen(link, seen)
                    continue

                print(f"[+] Evaluating EURAXESS vacancy: {title[:55]}...")
                analysis = evaluate_with_gemini(full_text)
                if analysis.get("is_relevant"):
                    send_alert(link, analysis, full_text)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')}) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                    
                mark_as_seen(link, seen)
        except Exception:
            pass

def scrape_academic_rss_feeds(seen: set):
    """Scrapes jobs.ac.uk, AcademicTransfer, and European academic feeds."""
    print("\n🎓 Scanning High-Yield Academic Portals (jobs.ac.uk & AcademicTransfer)...")
    for feed_url in NEW_ACADEMIC_PORTALS:
        try:
            resp = session.get(feed_url, timeout=25)
            if resp.status_code != 200:
                continue
            
            feed = feedparser.parse(resp.content)
            feed_name = feed_url.split('/')[-1]
            if feed.entries:
                print(f"  [>] Feed '{feed_name}': Found {len(feed.entries)} entries.")
            
            for entry in feed.entries:
                uid = entry.get("id", entry.link)
                if uid in seen:
                    continue
                
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if not is_recent_date(pub_dt):
                        mark_as_seen(uid, seen)
                        continue
                        
                content = f"{entry.title}\n{entry.get('summary', '')}"
                if len(content) < 30 or not fast_prefilter(content):
                    mark_as_seen(uid, seen)
                    continue
                    
                print(f"[+] Evaluating Academic Vacancy: {entry.title[:55]}...")
                analysis = evaluate_with_gemini(content)
                
                if analysis.get("is_relevant"):
                    send_alert(entry.link, analysis, content)
                    print(f"    [✓] MATCH CONFIRMED (Tier {analysis.get('tier')} - Score {analysis.get('confidence_score')}/10) -> Alert Sent!")
                else:
                    print(f"    [-] Skipped: {analysis.get('reason', 'Filtered by LLM')}")
                    
                mark_as_seen(uid, seen)
                
        except Exception as e:
            print(f"  [-] Academic feed error for '{feed_url[:40]}...': {e}")
            
def test_telegram():
    endpoint = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": "✅ PhD Finder Telegram test"
    }

    response = session.post(endpoint, json=payload, timeout=20)

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)
    
def main():
    print("🚀 Running PhD Finder Agent on Cloud...")
    seen = load_seen()
    print(f"📂 Cached database has {len(seen)} items.")
    
    # 1. Telegram Channels
    for ch in CHANNELS_TO_SCRAPE:
        scrape_telegram_channel_deep(ch, seen)
        
    # 2. International Academic Direct Scrapers
    scrape_findaphd_direct(seen)
    scrape_euraxess_direct(seen)
    
    # 3. New High-Yield Academic Portals (UK & Dutch University Portals)
    scrape_academic_rss_feeds(seen)
    
    print("✅ Run complete. Exiting cleanly.")

if __name__ == "__main__":
    test_telegram()
    main()
    
