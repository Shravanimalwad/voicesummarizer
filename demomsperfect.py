import re
import sqlite3
import threading
import hashlib
import os
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import *
from tkinter import ttk, messagebox, scrolledtext, filedialog

from youtube_transcript_api import YouTubeTranscriptApi

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "if", "so", "of", "to", "in",
    "on", "for", "with", "as", "is", "it", "its", "at", "by", "this",
    "that", "these", "those", "be", "was", "were", "are", "am", "i",
    "you", "he", "she", "we", "they", "them", "his", "her", "our", "your",
    "their", "from", "into", "about", "up", "out", "just", "like", "not",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "there", "here", "what", "which", "who", "when",
    "then", "than", "also", "very", "really", "going", "gonna", "get",
    "got", "know", "think", "one", "okay", "yeah",
}

STOPWORDS_HI = {
    "और", "का", "के", "की", "को", "में", "है", "हैं", "था", "थी", "थे",
    "पर", "एक", "यह", "वह", "ये", "वो", "तो", "भी", "से", "ही", "जो",
    "कि", "इस", "उस", "अपने", "अपनी", "हम", "आप", "तुम", "मैं", "हो",
    "गया", "गई", "गए", "रहा", "रही", "रहे", "कर", "करना", "किया", "किए",
    "नहीं", "क्या", "कैसे", "जब", "तब", "लिए", "साथ", "बात", "कुछ",
    "सब", "जैसे", "अगर", "लेकिन", "क्योंकि",
}

AUDIO_FILETYPES = [
    ("Audio files", "*.mp3 *.wav *.m4a *.ogg *.flac *.aac"),
    ("MP3 files", "*.mp3"),
    ("All files", "*.*"),
]

# Fixed number of sentences the summarizer produces (the length slider was removed).
DEFAULT_NUM_SENTENCES = 14


def get_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return None


def get_transcript(video_id, preferred_lang="en"):
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    transcript = None
    try:
        transcript = transcript_list.find_transcript([preferred_lang])
    except Exception:
        transcript = None

    if transcript is None:
        for t in transcript_list:
            transcript = t
            break

    if transcript is None:
        raise ValueError("No transcript is available for this video in any language.")

    fetched = transcript.fetch()
    full_text = " ".join(line.text for line in fetched)
    return full_text


_whisper_model_cache = {}


def transcribe_audio_file(file_path, model_size="base", status_callback=None):

    import whisper

    if status_callback:
        status_callback(f"Loading Whisper '{model_size}' model (first time may take a while)...")

    if model_size not in _whisper_model_cache:
        _whisper_model_cache[model_size] = whisper.load_model(model_size)
    model = _whisper_model_cache[model_size]

    if status_callback:
        status_callback("Transcribing audio... this can take a bit for long files.")

    result = model.transcribe(str(file_path))
    return result["text"]


# ===========================================================
# Live meeting recording — records from the default microphone
# via `sounddevice`, saves a WAV file, which then flows through
# the same transcribe_audio_file() + summarizer pipeline as any
# other uploaded/recorded meeting audio.
# ===========================================================

class LiveRecorder:
    """
    Records microphone audio in a background callback stream and saves
    it as a WAV file when stopped. Requires `sounddevice` + `numpy`
    (only imported when recording actually starts, so the app still
    launches fine if they're not installed).
    """

    def __init__(self, samplerate=44100, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self._frames = []
        self._stream = None
        self._recording = False

    @property
    def is_recording(self):
        return self._recording

    def start(self):
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError(
                "Live recording needs the 'sounddevice' and 'numpy' packages.\n\n"
                "Install them with:\n    pip install sounddevice numpy"
            ) from error
        except OSError as error:
            raise RuntimeError(
                "Live recording needs the PortAudio library, which isn't "
                "available on this system.\n\n"
                "On Windows/macOS this usually installs automatically with "
                "'pip install sounddevice'. On Linux, install it first with:\n"
                "    sudo apt-get install portaudio19-dev\n"
                "then:\n    pip install sounddevice numpy"
            ) from error

        self._frames = []
        self._recording = True

        def _callback(indata, frame_count, time_info, status):
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            callback=_callback,
        )
        self._stream.start()

    def stop_and_save(self, out_path):
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            raise ValueError("No audio was captured — check your microphone and try again.")

        import numpy as np
        import wave

        audio_data = np.concatenate(self._frames, axis=0)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 => 2 bytes per sample
            wf.setframerate(self.samplerate)
            wf.writeframes(audio_data.tobytes())

        self._frames = []
        return out_path


def detect_language(text):
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    latin_chars = re.findall(r"[a-zA-Z]", text)
    if len(hindi_chars) > len(latin_chars):
        return "hi"
    return "en"


def summarize_text_locally(transcript_text, num_sentences=14):
    """
    Simple frequency-based extractive summarizer.
    """
    transcript_text = transcript_text.strip()
    language = detect_language(transcript_text)

    if language == "hi":
        stopwords = STOPWORDS_HI
        word_pattern = r"[\u0900-\u097F]+"
    else:
        stopwords = STOPWORDS_EN
        word_pattern = r"[a-zA-Z']+"

    sentences = re.split(r"(?<=[.!?।])\s+", transcript_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    num_sentences = max(1, min(num_sentences, len(sentences)))

    if len(sentences) <= num_sentences:
        return "\n".join(f"• {s}" for s in sentences)

    word_counts = {}
    for sentence in sentences:
        words = re.findall(word_pattern, sentence if language == "hi" else sentence.lower())
        for word in words:
            if word not in stopwords and len(word) > 1:
                word_counts[word] = word_counts.get(word, 0) + 1

    if not word_counts:
        return "\n".join(f"• {s}" for s in sentences[:num_sentences])

    max_count = max(word_counts.values())
    normalized_counts = {w: c / max_count for w, c in word_counts.items()}

    sentence_scores = []
    for idx, sentence in enumerate(sentences):
        words = re.findall(word_pattern, sentence if language == "hi" else sentence.lower())
        if not words:
            score = 0.0
        else:
            score = sum(normalized_counts.get(w, 0.0) for w in words) / (len(words) ** 0.5)
        sentence_scores.append((idx, score, sentence))

    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_sentences.sort(key=lambda x: x[0])

    return "\n".join(f"• {s}" for _, _, s in top_sentences)


def save_summary_to_file(summary_text, identifier):
    out_folder = Path(__file__).parent / "summaries"
    out_folder.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_identifier = re.sub(r"[^A-Za-z0-9_-]", "_", str(identifier))[:40]
    file_path = out_folder / f"{safe_identifier}_{timestamp}.txt"
    file_path.write_text(summary_text, encoding="utf-8")
    return file_path


# ===========================================================
# Database layer (SQLite) — stores every summary ever produced
# and now also stores user accounts for the login page
# ===========================================================

DB_PATH = Path(__file__).parent / "media_summarizer.db"


# -----------------------------------------------------------
# App settings (persisted to a small JSON config file) — powers
# the Settings tab: default Whisper model + default save folder.
# -----------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "settings.json"

DEFAULT_CONFIG = {
    "whisper_model": "base",
    "default_save_folder": "",  # empty = fall back to Desktop
    "theme": "light",
}


def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_CONFIG)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def init_db():
    out_folder = Path(__file__).parent / "summaries"
    out_folder.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            source_type TEXT NOT NULL,       -- 'youtube' or 'audio'
            source_ref TEXT,                 -- URL or file path
            summary_text TEXT NOT NULL,
            sentence_count INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_summary(identifier, source_type, source_ref, summary_text, sentence_count):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO summaries (identifier, source_type, source_ref, summary_text, sentence_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            identifier,
            source_type,
            source_ref,
            summary_text,
            sentence_count,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return row_id


def fetch_history(limit=200):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT id, identifier, source_type, summary_text, created_at
        FROM summaries
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return rows


def fetch_summary_by_id(row_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT identifier, source_type, summary_text FROM summaries WHERE id = ?",
        (row_id,),
    ).fetchone()
    conn.close()
    return row


def fetch_stats():
    """Aggregate counts used by the Analytics tab."""
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    youtube_count = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE source_type = 'youtube'"
    ).fetchone()[0]
    audio_count = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE source_type = 'audio'"
    ).fetchone()[0]
    meeting_count = conn.execute(
        "SELECT COUNT(*) FROM summaries WHERE source_type = 'meeting'"
    ).fetchone()[0]
    avg_sentences = conn.execute(
        "SELECT AVG(sentence_count) FROM summaries"
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT identifier, created_at FROM summaries ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return {
        "total": total,
        "youtube_count": youtube_count,
        "audio_count": audio_count,
        "meeting_count": meeting_count,
        "avg_sentences": round(avg_sentences, 1) if avg_sentences else 0,
        "latest": latest,
    }


# -----------------------------------------------------------
# Shared "source type" display helper — used by History cards,
# PDF export, and the Analytics chart so the icon/label mapping
# for youtube / audio / meeting stays consistent everywhere.
# -----------------------------------------------------------

def describe_source_type(source_type):
    mapping = {
        "youtube": ("📺", "YouTube Video"),
        "audio": ("🎵", "Audio File"),
        "meeting": ("🎙️","Meeting Recording"),
    }
    return mapping.get(source_type, ("🎵", "Audio File"))


def delete_summary(row_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM summaries WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


# -----------------------------------------------------------
# User accounts: password hashing + create/verify helpers
# -----------------------------------------------------------

def _hash_password(password, salt):
    """PBKDF2-HMAC-SHA256 password hash — never store plaintext passwords."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    ).hex()


def create_user(username, password):
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password cannot be empty.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")

    salt = os.urandom(16).hex()
    password_hash = _hash_password(password, salt)

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("That username is already taken.")
    finally:
        conn.close()


def verify_user(username, password):
    """Returns True if the username/password combo is valid, else False."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT password_hash, salt FROM users WHERE username = ?",
        (username.strip(),),
    ).fetchone()
    conn.close()

    if row is None:
        return False

    stored_hash, salt = row
    return _hash_password(password, salt) == stored_hash


# ===========================================================
# PDF export
# ===========================================================

def _find_unicode_font():
    """
    Try to locate a system font that supports Devanagari (Hindi) glyphs,
    so PDFs render correctly regardless of language. Falls back to the
    built-in Helvetica font (Latin only) if none is found.
    """
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "C:/Windows/Fonts/Nirmala.ttf",
        "C:/Windows/Fonts/mangal.ttf",
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
    ]
    for path in candidate_paths:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("SummaryUnicodeFont", path))
                return "SummaryUnicodeFont"
            except Exception:
                continue
    return "Helvetica"


_PDF_ACCENT = colors.HexColor("#2563EB")
_PDF_MUTED = colors.HexColor("#64748B")
_PDF_TEXT = colors.HexColor("#0F172A")


def _draw_pdf_header_footer(canvas, doc, identifier, source_type):
    """
    Draws a professional letterhead-style header band and footer
    (with page number + branding) on every page of the exported PDF.
    """
    canvas.saveState()
    page_width, page_height = A4

    # --- Header band ---
    band_height = 2.0 * cm
    canvas.setFillColor(_PDF_ACCENT)
    canvas.rect(0, page_height - band_height, page_width, band_height, stroke=0, fill=1)

    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(2 * cm, page_height - 1.3 * cm, "🎬🎧 Media Summarizer")

    canvas.setFont("Helvetica", 9)
    _, type_text = describe_source_type(source_type)
    canvas.drawRightString(page_width - 2 * cm, page_height - 1.3 * cm, type_text)

    # --- Footer ---
    canvas.setFillColor(_PDF_MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm, "Generated by Media Summarizer — offline, on-device summary")
    canvas.drawRightString(page_width - 2 * cm, 1.2 * cm, f"Page {canvas.getPageNumber()}")
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.line(2 * cm, 1.5 * cm, page_width - 2 * cm, 1.5 * cm)

    canvas.restoreState()


def export_summary_to_pdf(summary_text, identifier, out_path=None, source_type="youtube"):
    out_folder = Path(__file__).parent / "summaries"
    out_folder.mkdir(exist_ok=True)

    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_identifier = re.sub(r"[^A-Za-z0-9_-]", "_", str(identifier))[:40]
        out_path = out_folder / f"{safe_identifier}_{timestamp}.pdf"

    font_name = _find_unicode_font()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=3.2 * cm,
        bottomMargin=2.2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Media Summary",
        author="Media Summarizer",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SummaryTitle",
        parent=styles["Heading1"],
        fontName=font_name if font_name != "Helvetica" else "Helvetica-Bold",
        fontSize=19,
        textColor=_PDF_TEXT,
        spaceAfter=4,
        alignment=TA_LEFT,
    )
    meta_style = ParagraphStyle(
        "SummaryMeta",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        textColor=_PDF_MUTED,
        spaceAfter=14,
        leading=13,
    )
    section_style = ParagraphStyle(
        "SummarySection",
        parent=styles["Normal"],
        fontName=font_name if font_name != "Helvetica" else "Helvetica-Bold",
        fontSize=11,
        textColor=_PDF_ACCENT,
        spaceBefore=4,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "SummaryBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        textColor=_PDF_TEXT,
        leading=17,
        spaceAfter=10,
        bulletFontName=font_name if font_name != "Helvetica" else "Helvetica-Bold",
        bulletFontSize=11,
        bulletColor=_PDF_ACCENT,
        bulletIndent=0,
        leftIndent=16,
    )

    _, source_label = describe_source_type(source_type)

    story = [
        Paragraph("Summary Report", title_style),
        Paragraph(
            f"<b>Source:</b> {identifier} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Type:</b> {source_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            meta_style,
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=14),
        Paragraph("Key Points", section_style),
    ]

    for line in summary_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip any bullet glyph the summarizer added and re-apply a
        # consistent bullet via reportlab's own bullet mechanism, which
        # gives a proper hanging indent in the PDF.
        stripped = line.lstrip("•-* ").strip()
        safe_line = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(safe_line, body_style, bulletText="●"))

    def _on_page(canvas, doc_):
        _draw_pdf_header_footer(canvas, doc_, identifier, source_type)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return out_path


# ===========================================================
# Palette — everything now runs on a clean white background
# ===========================================================

PALETTE = {
    "bg": "#FFFFFF",
    "panel": "#FFFFFF",
    "card": "#F1F5F9",
    "field": "#F1F5F9",
    "accent": "#2563EB",
    "accent2": "#10B981",
    "text": "#0F172A",
    "muted": "#475569",
    "success": "#16A34A",
    "warning": "#F59E0B",
    "error": "#DC2626",
    "border": "#E2E8F0",
    # Sidebar (dark, web-dashboard style) — contrasts with the white content area
    "sidebar_bg": "#111827",
    "sidebar_text": "#94A3B8",
    "sidebar_text_active": "#FFFFFF",
    "sidebar_hover": "#1F2937",
    "sidebar_active_bg": "#2563EB",
}


# ===========================================================
# Login / Sign-up window (SQLite-backed)
# ===========================================================

class LoginWindow:
    """
    Simple username/password gate backed by the `users` table in SQLite.
    Passwords are never stored in plaintext — see create_user/verify_user.
    On a successful login, `on_success(username)` is called, which the
    caller uses to close this window and open the main app.
    """

    def __init__(self, root, on_success):
        self.root = root
        self.on_success = on_success
        self.mode = "register"  # or "login" — new users sign up first

        self.root.title("🔐 Media Summarizer — Login")
        self.root.geometry("420x460")
        self.root.configure(bg=PALETTE["bg"])
        self.root.minsize(380, 420)

        init_db()

        self._build_ui()

    def _build_ui(self):
        self.container = Frame(self.root, bg=PALETTE["bg"])
        self.container.pack(expand=True, fill=BOTH, padx=30, pady=30)

        Label(
            self.container, text="🎬🎧 Media Summarizer",
            font=("Segoe UI", 18, "bold"), bg=PALETTE["bg"], fg=PALETTE["text"],
        ).pack(pady=(10, 0))

        self.subtitle_label = Label(
            self.container, text="Create a new account",
            font=("Segoe UI", 10), bg=PALETTE["bg"], fg=PALETTE["muted"],
        )
        self.subtitle_label.pack(pady=(2, 24))

        # Username field
        Label(
            self.container, text="Username", bg=PALETTE["bg"], fg=PALETTE["text"],
            font=("Segoe UI", 9, "bold"), anchor=W,
        ).pack(fill=X)
        self.username_entry = Entry(
            self.container, font=("Segoe UI", 11), bg=PALETTE["field"], fg=PALETTE["text"],
            insertbackground=PALETTE["text"], relief=FLAT,
            highlightthickness=1, highlightbackground=PALETTE["border"],
            highlightcolor=PALETTE["accent"],
        )
        self.username_entry.pack(fill=X, ipady=7, pady=(4, 14))

        # Password field
        Label(
            self.container, text="Password", bg=PALETTE["bg"], fg=PALETTE["text"],
            font=("Segoe UI", 9, "bold"), anchor=W,
        ).pack(fill=X)
        self.password_entry = Entry(
            self.container, font=("Segoe UI", 11), bg=PALETTE["field"], fg=PALETTE["text"],
            insertbackground=PALETTE["text"], relief=FLAT, show="•",
            highlightthickness=1, highlightbackground=PALETTE["border"],
            highlightcolor=PALETTE["accent"],
        )
        self.password_entry.pack(fill=X, ipady=7, pady=(4, 6))
        self.password_entry.bind("<Return>", lambda _e: self._submit())

        self.error_label = Label(
            self.container, text="", bg=PALETTE["bg"], fg=PALETTE["error"],
            font=("Segoe UI", 9), wraplength=340, justify=LEFT,
        )
        self.error_label.pack(fill=X, pady=(4, 10))

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Login.TButton", background=PALETTE["accent"], foreground="#ffffff",
            font=("Segoe UI", 11, "bold"), padding=8, borderwidth=0,
        )
        style.map("Login.TButton", background=[("active", "#1D4ED8")])

        self.submit_button = ttk.Button(
            self.container, text="Sign Up", style="Login.TButton", command=self._submit,
        )
        self.submit_button.pack(fill=X, pady=(6, 10))

        self.toggle_label = Label(
            self.container, text="Already have an account? Log in",
            font=("Segoe UI", 9, "underline"), bg=PALETTE["bg"], fg=PALETTE["accent"],
            cursor="hand2",
        )
        self.toggle_label.pack(pady=(6, 0))
        self.toggle_label.bind("<Button-1>", lambda _e: self._toggle_mode())

    def _toggle_mode(self):
        self.error_label.config(text="")
        if self.mode == "login":
            self.mode = "register"
            self.subtitle_label.config(text="Create a new account")
            self.submit_button.config(text="Sign Up")
            self.toggle_label.config(text="Already have an account? Log in")
        else:
            self.mode = "login"
            self.subtitle_label.config(text="Log in to continue")
            self.submit_button.config(text="Log In")
            self.toggle_label.config(text="Don't have an account? Sign up")

    def _submit(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get()

        if not username or not password:
            self.error_label.config(text="Please enter both a username and a password.")
            return

        if self.mode == "login":
            if verify_user(username, password):
                self.error_label.config(text="")
                self.on_success(username)
            else:
                self.error_label.config(text="Invalid username or password.")
        else:
            try:
                create_user(username, password)
                messagebox.showinfo("Account created", "Account created! You can now log in.")
                self._toggle_mode()
                self.password_entry.delete(0, END)
            except ValueError as error:
                self.error_label.config(text=str(error))


class MediaSummarizerApp:
    APP_VERSION = "2.0.0"
    NAV_ITEMS = [
        ("youtube", "📺", "YouTube Video"),
        ("audio", "🎵", "Audio File"),
        ("meetings", "🎙️","Meetings"),
        ("history", "🕘", "History"),
        ("analytics", "📊", "Analytics"),
        ("settings", "⚙️", "Settings"),
    ]

    def __init__(self, root, username=None, on_logout=None):
        self.root = root
        self.username = username
        self.on_logout = on_logout
        self.config = load_config()

        title_suffix = f" — logged in as {username}" if username else ""
        self.root.title(f"🎬🎧 Media Summarizer{title_suffix}")
        self.root.geometry("980x760")
        self.root.configure(bg=PALETTE["bg"])
        self.root.minsize(820, 620)

        init_db()

        self.current_identifier = None
        self.current_summary = None
        self.current_source_type = None
        self.current_source_ref = None
        self.audio_path_var = StringVar(value="No file selected")
        self.current_page = "youtube"
        self.nav_buttons = {}
        self.pages = {}

        # Live meeting recording state
        self.recorder = LiveRecorder()
        self.record_start_time = None
        self.record_timer_job = None
        self.meeting_recorded_path_var = StringVar(value="No recorded meeting selected")
        self.meeting_mode = "live"  # or "recorded"

        self._build_style()
        self._build_header()
        self._build_body()

        self._refresh_history()
        self._refresh_analytics()
        self._show_page("youtube")

    def _logout(self):
        if not messagebox.askyesno("Log Out", "Are you sure you want to log out?"):
            return
        if self.on_logout:
            self.on_logout()

    # -------------------- styling --------------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=PALETTE["card"],
            foreground=PALETTE["muted"],
            padding=(16, 10),
            font=("Segoe UI", 11, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2563EB")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("TFrame", background=PALETTE["bg"])
        style.configure(
            "Accent.TButton",
            background=PALETTE["accent"],
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            padding=8,
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#1D4ED8")])

        style.configure(
            "Secondary.TButton",
            background=PALETTE["accent2"],
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            padding=6,
            borderwidth=0,
        )
        style.map("Secondary.TButton", background=[("active", "#059669")])

        style.configure(
            "Horizontal.TScale",
            background=PALETTE["bg"],
            troughcolor=PALETTE["card"],
        )

    # -------------------- header --------------------
    def _build_header(self):
        header = Frame(self.root, bg=PALETTE["bg"])
        header.pack(fill=X, pady=(18, 6), padx=20)

        title_row = Frame(header, bg=PALETTE["bg"])
        title_row.pack(fill=X)

        Label(
            title_row,
            text="🎬🎧 Media Summarizer",
            font=("Segoe UI", 22, "bold"),
            bg=PALETTE["bg"],
            fg=PALETTE["text"],
        ).pack(side=LEFT, anchor=W)

        subtitle_text = "Summarize any YouTube video or local audio file — completely offline."
        if self.username:
            subtitle_text += f"   (logged in as {self.username})"

        Label(
            header,
            text=subtitle_text,
            font=("Segoe UI", 10),
            bg=PALETTE["bg"],
            fg=PALETTE["muted"],
        ).pack(anchor=W, pady=(2, 0))

    # -------------------- tabs (mode selector) --------------------
    # -------------------- body: sidebar (left) + pages (right) --------------------
    def _build_body(self):
        body = Frame(self.root, bg=PALETTE["bg"])
        body.pack(fill=BOTH, expand=True)

        self._build_sidebar(body)

        self.content_area = Frame(body, bg=PALETTE["bg"])
        self.content_area.pack(side=LEFT, fill=BOTH, expand=True)

        self.pages_container = Frame(self.content_area, bg=PALETTE["bg"])
        self.pages_container.pack(fill=BOTH, expand=True, padx=24, pady=(20, 0))

        self._build_youtube_page()
        self._build_audio_page()
        self._build_meetings_page()
        self._build_history_page()
        self._build_analytics_page()
        self._build_settings_page()

        self._build_result_panel()

    # -------------------- sidebar nav (web-app style) --------------------
    def _build_sidebar(self, parent):
        sidebar = Frame(parent, bg=PALETTE["sidebar_bg"], width=210)
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)

        Label(
            sidebar, text="MENU", bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"],
            font=("Segoe UI", 9, "bold"), anchor=W,
        ).pack(fill=X, padx=20, pady=(24, 10))

        for page_id, icon, label in self.NAV_ITEMS:
            btn = Button(
                sidebar, text=f"  {icon}   {label}", anchor=W, bd=0, relief=FLAT,
                bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"],
                activebackground=PALETTE["sidebar_hover"],
                activeforeground=PALETTE["sidebar_text_active"],
                highlightthickness=0, font=("Segoe UI", 10), padx=8, pady=10, cursor="hand2",
                command=lambda pid=page_id: self._show_page(pid),
            )
            btn.pack(fill=X, padx=10, pady=2)
            self.nav_buttons[page_id] = btn

            def _enter(_e, b=btn, pid=page_id):
                if pid != self.current_page:
                    b.configure(bg=PALETTE["sidebar_hover"])

            def _leave(_e, b=btn, pid=page_id):
                if pid != self.current_page:
                    b.configure(bg=PALETTE["sidebar_bg"])

            btn.bind("<Enter>", _enter)
            btn.bind("<Leave>", _leave)

        # Spacer pushes the footer to the bottom of the sidebar
        Frame(sidebar, bg=PALETTE["sidebar_bg"]).pack(fill=BOTH, expand=True)

        if self.username:
            Button(
                sidebar, text="  🚪   Log Out", anchor=W, bd=0, relief=FLAT,
                bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"],
                activebackground=PALETTE["sidebar_hover"],
                activeforeground=PALETTE["sidebar_text_active"],
                highlightthickness=0, font=("Segoe UI", 10), padx=8, pady=10, cursor="hand2",
                command=self._logout,
            ).pack(fill=X, padx=10, pady=(0, 6))

        Label(
            sidebar, text="Media Summarizer", bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"],
            font=("Segoe UI", 8), anchor=W,
        ).pack(fill=X, padx=20, pady=(0, 2))
        Label(
            sidebar, text=f"v{self.APP_VERSION}", bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"],
            font=("Segoe UI", 8), anchor=W,
        ).pack(fill=X, padx=20, pady=(0, 20))

    def _show_page(self, page_id):
        self.current_page = page_id
        for pid, btn in self.nav_buttons.items():
            if pid == page_id:
                btn.configure(bg=PALETTE["sidebar_active_bg"], fg=PALETTE["sidebar_text_active"])
            else:
                btn.configure(bg=PALETTE["sidebar_bg"], fg=PALETTE["sidebar_text"])

        for pid, frame in self.pages.items():
            frame.grid_remove()
        self.pages[page_id].grid(row=0, column=0, sticky="nsew")

        # The result panel (summary output) is only relevant on the
        # YouTube / Audio / Meetings pages — hide it elsewhere, like a web app would.
        if page_id in ("youtube", "audio", "meetings"):
            self.result_panel.pack(fill=BOTH, expand=True, padx=24, pady=(16, 20))
        else:
            self.result_panel.pack_forget()

        if page_id == "history":
            self._refresh_history()
        elif page_id == "analytics":
            self._refresh_analytics()

    def _make_page(self, page_id):
        page = Frame(self.pages_container, bg=PALETTE["panel"])
        page.grid(row=0, column=0, sticky="nsew")
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)
        self.pages[page_id] = page
        return page

    def _page_title(self, page, text):
        Label(
            page, text=text, bg=PALETTE["panel"], fg=PALETTE["text"],
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor=W, pady=(4, 16))

    # -------------------- YouTube page --------------------
    def _build_youtube_page(self):
        page = self._make_page("youtube")
        self._page_title(page, "📺 Summarize a YouTube Video")

        Label(
            page, text="Paste a YouTube URL:", bg=PALETTE["panel"], fg=PALETTE["text"],
            font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(0, 4))

        yt_row = Frame(page, bg=PALETTE["panel"])
        yt_row.pack(fill=X)

        self.url_entry = Entry(
            yt_row, font=("Segoe UI", 11), bg=PALETTE["field"], fg=PALETTE["text"],
            insertbackground=PALETTE["text"], relief=FLAT,
        )
        self.url_entry.pack(side=LEFT, expand=True, fill=X, ipady=6, padx=(0, 10))

        ttk.Button(
            yt_row, text="✨ Summarize Video", style="Accent.TButton",
            command=self.start_youtube_thread,
        ).pack(side=LEFT)

    # -------------------- Audio page --------------------
    def _build_audio_page(self):
        page = self._make_page("audio")
        self._page_title(page, "🎵 Summarize an Audio File")

        Label(



            
            page, text="Pick a local audio file (mp3, wav, m4a, ogg, flac):",
            bg=PALETTE["panel"], fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(0, 4))

        audio_row = Frame(page, bg=PALETTE["panel"])
        audio_row.pack(fill=X, pady=(0, 6))

        Label(
            audio_row, textvariable=self.audio_path_var, bg=PALETTE["field"], fg=PALETTE["muted"],
            anchor=W, font=("Segoe UI", 10), relief=FLAT,
        ).pack(side=LEFT, expand=True, fill=X, ipady=6, ipadx=8, padx=(0, 10))

        ttk.Button(
            audio_row, text="📂 Browse...", style="Secondary.TButton",
            command=self.browse_audio_file,
        ).pack(side=LEFT)

        model_row = Frame(page, bg=PALETTE["panel"])
        model_row.pack(fill=X, pady=(6, 4))

        Label(
            model_row, text="Whisper model size:", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Segoe UI", 9),
        ).pack(side=LEFT)

        self.whisper_model_var = StringVar(value=self.config.get("whisper_model", "base"))
        model_dropdown = ttk.Combobox(
            model_row, textvariable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium"], state="readonly", width=10,
        )
        model_dropdown.pack(side=LEFT, padx=8)

        Label(
            model_row, text="(tiny = fastest, medium = most accurate but slower)",
            bg=PALETTE["panel"], fg=PALETTE["muted"], font=("Segoe UI", 8, "italic"),
        ).pack(side=LEFT)

        audio_btn_row = Frame(page, bg=PALETTE["panel"])
        audio_btn_row.pack(fill=X, pady=(6, 14))

        ttk.Button(
            audio_btn_row, text="✨ Transcribe & Summarize", style="Accent.TButton",
            command=self.start_audio_thread,
        ).pack(side=LEFT)

    # -------------------- Meetings page: Live Recording + Recorded Meeting --------------------
    def _build_meetings_page(self):
        page = self._make_page("meetings")
        self._page_title(page, "🎙️ Meetings")

        # Segmented toggle between the two modes
        toggle_row = Frame(page, bg=PALETTE["panel"])
        toggle_row.pack(fill=X, pady=(0, 16))

        self.meeting_live_btn = ttk.Button(
            toggle_row, text="🔴 Live Recording", style="Accent.TButton",
            command=lambda: self._set_meeting_mode("live"),
        )
        self.meeting_live_btn.pack(side=LEFT, padx=(0, 8))

        self.meeting_recorded_btn = ttk.Button(
            toggle_row, text="📁 Recorded Meeting", style="Secondary.TButton",
            command=lambda: self._set_meeting_mode("recorded"),
        )
        self.meeting_recorded_btn.pack(side=LEFT)

        # --- Live Recording sub-panel ---
        self.meeting_live_panel = Frame(page, bg=PALETTE["panel"])

        Label(
            self.meeting_live_panel, text="Meeting name:", bg=PALETTE["panel"],
            fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(0, 4))

        self.meeting_name_entry = Entry(
            self.meeting_live_panel, font=("Segoe UI", 11), bg=PALETTE["field"],
            fg=PALETTE["text"], insertbackground=PALETTE["text"], relief=FLAT,
        )
        self.meeting_name_entry.pack(fill=X, ipady=6, pady=(0, 16))
        self.meeting_name_entry.insert(0, "Team Meeting")

        record_row = Frame(self.meeting_live_panel, bg=PALETTE["panel"])
        record_row.pack(fill=X)

        self.record_button = ttk.Button(
            record_row, text="🔴 Start Recording", style="Accent.TButton",
            command=self.toggle_recording,
        )
        self.record_button.pack(side=LEFT, padx=(0, 14))

        self.record_timer_label = Label(
            record_row, text="00:00", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Segoe UI", 14, "bold"),
        )
        self.record_timer_label.pack(side=LEFT)

        Label(
            self.meeting_live_panel,
            text="Records audio from your default microphone. Click Stop when the "
                 "meeting ends to transcribe and summarize it automatically.",
            bg=PALETTE["panel"], fg=PALETTE["muted"], font=("Segoe UI", 9),
            wraplength=560, justify=LEFT,
        ).pack(anchor=W, pady=(14, 0))

        # --- Recorded Meeting sub-panel (upload a past recording) ---
        self.meeting_recorded_panel = Frame(page, bg=PALETTE["panel"])

        Label(
            self.meeting_recorded_panel,
            text="Pick a previously recorded meeting file (mp3, wav, m4a, ogg, flac):",
            bg=PALETTE["panel"], fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(0, 4))

        recorded_row = Frame(self.meeting_recorded_panel, bg=PALETTE["panel"])
        recorded_row.pack(fill=X, pady=(0, 6))

        Label(
            recorded_row, textvariable=self.meeting_recorded_path_var, bg=PALETTE["field"],
            fg=PALETTE["muted"], anchor=W, font=("Segoe UI", 10), relief=FLAT,
        ).pack(side=LEFT, expand=True, fill=X, ipady=6, ipadx=8, padx=(0, 10))

        ttk.Button(
            recorded_row, text="📂 Browse...", style="Secondary.TButton",
            command=self.browse_recorded_meeting,
        ).pack(side=LEFT)

        Label(
            self.meeting_recorded_panel, text="Meeting name (optional):",
            bg=PALETTE["panel"], fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(12, 4))

        self.meeting_recorded_name_entry = Entry(
            self.meeting_recorded_panel, font=("Segoe UI", 11), bg=PALETTE["field"],
            fg=PALETTE["text"], insertbackground=PALETTE["text"], relief=FLAT,
        )
        self.meeting_recorded_name_entry.pack(fill=X, ipady=6, pady=(0, 16))

        ttk.Button(
            self.meeting_recorded_panel, text="✨ Transcribe & Summarize", style="Accent.TButton",
            command=self.start_recorded_meeting_thread,
        ).pack(anchor=W)

        self._set_meeting_mode("live")

    def _set_meeting_mode(self, mode):
        self.meeting_mode = mode
        if mode == "live":
            self.meeting_live_btn.configure(style="Accent.TButton")
            self.meeting_recorded_btn.configure(style="Secondary.TButton")
            self.meeting_recorded_panel.pack_forget()
            self.meeting_live_panel.pack(fill=X)
        else:
            self.meeting_recorded_btn.configure(style="Accent.TButton")
            self.meeting_live_btn.configure(style="Secondary.TButton")
            self.meeting_live_panel.pack_forget()
            self.meeting_recorded_panel.pack(fill=X)

    def browse_recorded_meeting(self):
        path = filedialog.askopenfilename(title="Choose a recorded meeting file", filetypes=AUDIO_FILETYPES)
        if path:
            self.meeting_recorded_path_var.set(path)

    # -------------------- Live recording controls --------------------
    def toggle_recording(self):
        if not self.recorder.is_recording:
            self._start_recording()
        else:
            self._stop_recording_and_summarize()

    def _start_recording(self):
        try:
            self.recorder.start()
        except RuntimeError as error:
            messagebox.showerror("Missing dependency", str(error))
            return

        self.record_start_time = datetime.now()
        self.record_button.config(text="⏹ Stop & Summarize")
        self.set_status("🔴 Recording... speak into your microphone.")
        self._tick_recording_timer()

    def _tick_recording_timer(self):
        if not self.recorder.is_recording:
            return
        elapsed = int((datetime.now() - self.record_start_time).total_seconds())
        minutes, seconds = divmod(elapsed, 60)
        self.record_timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
        self.record_timer_job = self.root.after(1000, self._tick_recording_timer)

    def _stop_recording_and_summarize(self):
        if self.record_timer_job:
            self.root.after_cancel(self.record_timer_job)
            self.record_timer_job = None

        meeting_name = self.meeting_name_entry.get().strip() or "Meeting"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", meeting_name)[:40]
        recordings_folder = Path(__file__).parent / "recordings"
        out_path = recordings_folder / f"{safe_name}_{timestamp}.wav"

        self.record_button.config(text="🔴 Start Recording")
        self.record_timer_label.config(text="00:00")

        try:
            self.set_status("💾 Saving recording...")
            saved_path = self.recorder.stop_and_save(out_path)
        except ValueError as error:
            self.set_status("Recording failed.", kind="error")
            messagebox.showerror("Recording Error", str(error))
            return

        self._show_page("meetings")  # ensure result panel is visible
        thread = threading.Thread(
            target=self._transcribe_and_summarize_meeting,
            args=(saved_path, meeting_name),
            daemon=True,
        )
        thread.start()

    def _transcribe_and_summarize_meeting(self, audio_path, meeting_name):
        self._set_busy(True)
        try:
            model_size = self.config.get("whisper_model", "base")
            transcript_text = transcribe_audio_file(
                audio_path, model_size=model_size, status_callback=self.set_status
            )
            if not transcript_text.strip():
                raise ValueError("Transcription was empty. Try recording again with clearer audio.")

            self._finish_with_transcript(
                transcript_text, identifier=meeting_name,
                source_type="meeting", source_ref=str(audio_path),
            )
        except ModuleNotFoundError:
            self.set_status("Whisper not installed.", kind="error")
            messagebox.showerror(
                "Missing dependency",
                "Meeting transcription needs the 'openai-whisper' package.\n\n"
                "Install it with:\n    pip install openai-whisper\n\n"
                "You'll also need ffmpeg installed on your system.",
            )
        except Exception as error:
            self.set_status("Something went wrong.", kind="error")
            messagebox.showerror("Error", str(error))
        finally:
            self._set_busy(False)

    # -------------------- Recorded meeting (upload) flow --------------------
    def start_recorded_meeting_thread(self):
        thread = threading.Thread(target=self.summarize_recorded_meeting, daemon=True)
        thread.start()

    def summarize_recorded_meeting(self):
        path_str = self.meeting_recorded_path_var.get()
        if not path_str or path_str == "No recorded meeting selected":
            messagebox.showwarning("No file", "Please choose a recorded meeting file first.")
            return

        audio_path = Path(path_str)
        if not audio_path.exists():
            messagebox.showerror("File not found", f"Could not find:\n{audio_path}")
            return

        meeting_name = self.meeting_recorded_name_entry.get().strip() or audio_path.stem

        self._set_busy(True)
        try:
            model_size = self.config.get("whisper_model", "base")
            transcript_text = transcribe_audio_file(
                audio_path, model_size=model_size, status_callback=self.set_status
            )
            if not transcript_text.strip():
                raise ValueError("Transcription was empty. Try a different file.")

            self._finish_with_transcript(
                transcript_text, identifier=meeting_name,
                source_type="meeting", source_ref=str(audio_path),
            )
        except ModuleNotFoundError:
            self.set_status("Whisper not installed.", kind="error")
            messagebox.showerror(
                "Missing dependency",
                "Meeting transcription needs the 'openai-whisper' package.\n\n"
                "Install it with:\n    pip install openai-whisper\n\n"
                "You'll also need ffmpeg installed on your system.",
            )
        except Exception as error:
            self.set_status("Something went wrong.", kind="error")
            messagebox.showerror("Error", str(error))
        finally:
            self._set_busy(False)

    # -------------------- History page: scrollable web-style card feed --------------------
    def _build_history_page(self):
        page = self._make_page("history")

        history_header = Frame(page, bg=PALETTE["panel"])
        history_header.pack(fill=X, pady=(0, 12))

        Label(
            history_header, text="🕘 Past Summaries", bg=PALETTE["panel"], fg=PALETTE["text"],
            font=("Segoe UI", 16, "bold"),
        ).pack(side=LEFT)

        ttk.Button(
            history_header, text="🔄 Refresh", style="Secondary.TButton",
            command=self._refresh_history,
        ).pack(side=RIGHT)

        feed_outer = Frame(page, bg=PALETTE["panel"], highlightthickness=1,
                            highlightbackground=PALETTE["border"])
        feed_outer.pack(fill=BOTH, expand=True)

        self.history_canvas = Canvas(feed_outer, bg=PALETTE["card"], highlightthickness=0)
        feed_scrollbar = Scrollbar(feed_outer, orient=VERTICAL, command=self.history_canvas.yview)
        self.history_canvas.configure(yscrollcommand=feed_scrollbar.set)

        feed_scrollbar.pack(side=RIGHT, fill=Y)
        self.history_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        self.history_feed = Frame(self.history_canvas, bg=PALETTE["card"])
        self._history_feed_window = self.history_canvas.create_window(
            (0, 0), window=self.history_feed, anchor="nw"
        )

        def _sync_scrollregion(_event=None):
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

        def _sync_feed_width(event):
            self.history_canvas.itemconfig(self._history_feed_window, width=event.width)

        self.history_feed.bind("<Configure>", _sync_scrollregion)
        self.history_canvas.bind("<Configure>", _sync_feed_width)

        def _on_mousewheel(event):
            delta = -1 if event.num == 5 or event.delta < 0 else 1
            self.history_canvas.yview_scroll(-delta, "units")

        # Cross-platform mouse wheel scrolling
        self.history_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.history_canvas.bind_all("<Button-4>", _on_mousewheel)
        self.history_canvas.bind_all("<Button-5>", _on_mousewheel)

    # -------------------- Analytics page --------------------
    def _build_analytics_page(self):
        page = self._make_page("analytics")
        self._page_title(page, "📊 Usage Overview")

        self.stats_cards_frame = Frame(page, bg=PALETTE["panel"])
        self.stats_cards_frame.pack(fill=X)

        self.analytics_chart_canvas = Canvas(
            page, bg=PALETTE["panel"], highlightthickness=0, height=180,
        )
        self.analytics_chart_canvas.pack(fill=X, pady=(18, 6))

        self.analytics_latest_label = Label(
            page, text="", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Segoe UI", 9, "italic"),
        )
        self.analytics_latest_label.pack(anchor=W, pady=(4, 14))

        ttk.Button(
            page, text="🔄 Refresh Stats", style="Secondary.TButton",
            command=self._refresh_analytics,
        ).pack(anchor=W)

    # -------------------- Settings page --------------------
    def _build_settings_page(self):
        page = self._make_page("settings")
        self._page_title(page, "⚙️ Preferences")

        Label(
            page, text="Default Whisper model for audio transcription:",
            bg=PALETTE["panel"], fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(6, 4))

        self.settings_model_var = StringVar(value=self.config.get("whisper_model", "base"))
        ttk.Combobox(
            page, textvariable=self.settings_model_var,
            values=["tiny", "base", "small", "medium"], state="readonly", width=14,
        ).pack(anchor=W)

        Label(
            page, text="Default folder for saving PDFs (blank = Desktop):",
            bg=PALETTE["panel"], fg=PALETTE["text"], font=("Segoe UI", 10),
        ).pack(anchor=W, pady=(18, 4))

        folder_row = Frame(page, bg=PALETTE["panel"])
        folder_row.pack(fill=X)

        self.settings_folder_var = StringVar(value=self.config.get("default_save_folder", ""))
        Entry(
            folder_row, textvariable=self.settings_folder_var, font=("Segoe UI", 10),
            bg=PALETTE["field"], fg=PALETTE["text"], insertbackground=PALETTE["text"], relief=FLAT,
        ).pack(side=LEFT, expand=True, fill=X, ipady=6, padx=(0, 8))

        ttk.Button(
            folder_row, text="📂 Browse...", style="Secondary.TButton",
            command=self._choose_default_folder,
        ).pack(side=LEFT)

        self.settings_status_label = Label(
            page, text="", bg=PALETTE["panel"], fg=PALETTE["success"],
            font=("Segoe UI", 9, "italic"),
        )
        self.settings_status_label.pack(anchor=W, pady=(10, 4))

        ttk.Button(
            page, text="💾 Save File", style="Accent.TButton",
            command=self._save_settings,
        ).pack(anchor=W)

    def _choose_default_folder(self):
        folder = filedialog.askdirectory(title="Choose default save folder")
        if folder:
            self.settings_folder_var.set(folder)

    def _save_settings(self):
        self.config["whisper_model"] = self.settings_model_var.get()
        self.config["default_save_folder"] = self.settings_folder_var.get().strip()
        save_config(self.config)
        self.whisper_model_var.set(self.config["whisper_model"])
        self.settings_status_label.config(text="✅ Settings saved.")

    def browse_audio_file(self):
        path = filedialog.askopenfilename(title="Choose an audio file", filetypes=AUDIO_FILETYPES)
        if path:
            self.audio_path_var.set(path)

    # -------------------- shared result panel (status + summary + save/pdf) --------------------
    def _build_result_panel(self):
        self.result_panel = Frame(self.content_area, bg=PALETTE["bg"])
        # Packed/unpacked dynamically by _show_page() — only shown on the
        # YouTube / Audio pages, like a web app's contextual results panel.

        self.status_label = Label(
            self.result_panel, text="Ready.", bg=PALETTE["bg"], fg=PALETTE["muted"],
            font=("Segoe UI", 9, "italic"),
        )
        self.status_label.pack(anchor=W, pady=(0, 6))

        Label(
            self.result_panel, text="📝 Summary:", bg=PALETTE["bg"], fg=PALETTE["text"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor=W, pady=(0, 2))

        self.output_box = scrolledtext.ScrolledText(
            self.result_panel, wrap=WORD, font=("Georgia", 12),
            bg=PALETTE["field"], fg=PALETTE["text"], insertbackground=PALETTE["text"],
            relief=FLAT, padx=14, pady=12, spacing1=2, spacing2=2, height=10,
        )
        self.output_box.pack(expand=True, fill=BOTH, pady=(0, 10))

        # Tag used for the bullet glyph itself — bold, accent-colored, slightly larger
        self.output_box.tag_configure(
            "bullet_mark",
            font=("Georgia", 14, "bold"),
            foreground=PALETTE["accent"],
        )
        # Tag used for each bullet's text — nice serif font, generous line/paragraph
        # spacing and a hanging indent so wrapped lines line up under the bullet.
        self.output_box.tag_configure(
            "bullet_text",
            font=("Georgia", 12),
            foreground=PALETTE["text"],
            lmargin1=28,
            lmargin2=46,
            spacing1=4,
            spacing3=12,
        )
        # Fallback tag for any non-bulleted line (e.g. plain paragraphs)
        self.output_box.tag_configure(
            "plain_text",
            font=("Georgia", 12),
            foreground=PALETTE["text"],
            lmargin1=4,
            lmargin2=4,
            spacing1=4,
            spacing3=8,
        )

        save_row = Frame(self.result_panel, bg=PALETTE["bg"])
        save_row.pack()

        self.save_button = ttk.Button(
            save_row, text="💾 Save Summary to File", style="Secondary.TButton",
            command=self.save_summary, state=DISABLED,
        )
        self.save_button.pack(side=LEFT, padx=(0, 10))

        self.pdf_button = ttk.Button(
            save_row, text="⬇️ Download PDF", style="Accent.TButton",
            command=self.export_pdf, state=DISABLED,
        )
        self.pdf_button.pack(side=LEFT)

    def set_status(self, text, kind="info"):
        color = {
            "info": PALETTE["muted"],
            "success": PALETTE["success"],
            "error": PALETTE["error"],
        }.get(kind, PALETTE["muted"])
        self.status_label.config(text=text, fg=color)
        self.root.update_idletasks()

    def _render_summary(self, summary_text):
        """
        Render summary text into the output box with nicely styled bullet
        points: a bold accent-colored bullet glyph followed by the point
        text in a readable serif font, with a hanging indent so wrapped
        lines stay aligned under the bullet.
        """
        self.output_box.delete("1.0", END)
        for raw_line in summary_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(("•", "-", "*")):
                point = line.lstrip("•-* ").strip()
                self.output_box.insert(END, "●  ", "bullet_mark")
                self.output_box.insert(END, point + "\n", "bullet_text")
            else:
                self.output_box.insert(END, line + "\n", "plain_text")

    # -------------------- YouTube flow --------------------
    def start_youtube_thread(self):
        thread = threading.Thread(target=self.summarize_youtube, daemon=True)
        thread.start()

    def summarize_youtube(self):
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please paste a YouTube URL first.")
            return

        video_id = get_video_id(url)
        if not video_id:
            messagebox.showerror("Invalid URL", "Could not find a video ID in that URL.")
            return

        # Open the pasted video in the user's default web browser so it
        # starts playing right away, while the transcript/summary work
        # continues in the background.
        try:
            webbrowser.open(f"https://www.youtube.com/watch?v={video_id}", new=2)
        except Exception:
            pass  # Don't block summarization if the browser can't be opened

        self._set_busy(True)
        try:
            self.set_status("📥 Fetching transcript...")
            transcript_text = get_transcript(video_id)

            if not transcript_text.strip():
                raise ValueError("Transcript was empty.")

            self._finish_with_transcript(
                transcript_text, identifier=video_id, source_type="youtube", source_ref=url
            )

        except Exception as error:
            self.set_status("Something went wrong.", kind="error")
            messagebox.showerror("Error", str(error))
        finally:
            self._set_busy(False)

    # -------------------- Audio file flow --------------------
    def start_audio_thread(self):
        thread = threading.Thread(target=self.summarize_audio, daemon=True)
        thread.start()

    def summarize_audio(self):
        path_str = self.audio_path_var.get()
        if not path_str or path_str == "No file selected":
            messagebox.showwarning("No file", "Please choose an audio file first.")
            return

        audio_path = Path(path_str)
        if not audio_path.exists():
            messagebox.showerror("File not found", f"Could not find:\n{audio_path}")
            return

        self._set_busy(True)
        try:
            model_size = self.whisper_model_var.get()
            transcript_text = transcribe_audio_file(
                audio_path, model_size=model_size, status_callback=self.set_status
            )

            if not transcript_text.strip():
                raise ValueError("Transcription was empty. Try a different model size or file.")

            self._finish_with_transcript(
                transcript_text,
                identifier=audio_path.stem,
                source_type="audio",
                source_ref=str(audio_path),
            )

        except ModuleNotFoundError:
            self.set_status("Whisper not installed.", kind="error")
            messagebox.showerror(
                "Missing dependency",
                "Audio transcription needs the 'openai-whisper' package.\n\n"
                "Install it with:\n    pip install openai-whisper\n\n"
                "You'll also need ffmpeg installed on your system.",
            )
        except Exception as error:
            self.set_status("Something went wrong.", kind="error")
            messagebox.showerror("Error", str(error))
        finally:
            self._set_busy(False)

    # -------------------- shared finish step --------------------
    def _finish_with_transcript(self, transcript_text, identifier, source_type, source_ref):
        self.set_status("🧠 Summarizing...")
        num_sentences = DEFAULT_NUM_SENTENCES
        summary = summarize_text_locally(transcript_text, num_sentences=num_sentences)

        self.current_identifier = identifier
        self.current_summary = summary
        self.current_source_type = source_type
        self.current_source_ref = source_ref

        self._render_summary(summary)
        self.save_button.config(state=NORMAL)
        self.pdf_button.config(state=NORMAL)

        try:
            insert_summary(
                identifier=identifier,
                source_type=source_type,
                source_ref=source_ref,
                summary_text=summary,
                sentence_count=num_sentences,
            )
            self._refresh_history()
            self._refresh_analytics()
            self.set_status("✅ Done! Saved to database.", kind="success")
        except Exception as db_error:
            self.set_status(f"✅ Done, but DB save failed: {db_error}", kind="error")

    def _set_busy(self, busy):
        self.save_button.config(state=DISABLED if busy else (NORMAL if self.current_summary else DISABLED))
        self.pdf_button.config(state=DISABLED if busy else (NORMAL if self.current_summary else DISABLED))
        if not busy:
            return
        self.output_box.delete("1.0", END)

    def save_summary(self):
        if not self.current_summary:
            return
        try:
            path = save_summary_to_file(self.current_summary, self.current_identifier)
            messagebox.showinfo("Saved", f"Summary saved to:\n{path}")
        except Exception as error:
            messagebox.showerror("Error", f"Could not save file: {error}")

    # -------------------- PDF export --------------------
    @staticmethod
    def _get_desktop_path():
        """Best-effort cross-platform guess at the user's Desktop folder."""
        home = Path.home()
        candidates = [home / "Desktop", home / "OneDrive" / "Desktop"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return home

    def _get_default_save_folder(self):
        """Respects the Settings-tab 'default save folder' if the user set
        one; otherwise falls back to the Desktop."""
        configured = self.config.get("default_save_folder", "").strip()
        if configured and Path(configured).exists():
            return Path(configured)
        return self._get_desktop_path()

    def export_pdf(self):
        if not self.current_summary:
            return

        safe_identifier = re.sub(r"[^A-Za-z0-9_-]", "_", str(self.current_identifier))[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{safe_identifier}_summary_{timestamp}.pdf"

        # Let the user pick exactly where to save — defaults to Desktop so
        # "Save to Desktop" is a single click away, but any folder works.
        chosen_path = filedialog.asksaveasfilename(
            title="Save Summary as PDF",
            initialdir=str(self._get_default_save_folder()),
            initialfile=default_filename,
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not chosen_path:
            return  # user cancelled the dialog

        try:
            self.set_status("📄 Building PDF...")
            path = export_summary_to_pdf(
                self.current_summary,
                self.current_identifier,
                out_path=Path(chosen_path),
                source_type=self.current_source_type or "youtube",
            )
            self.set_status("✅ PDF exported.", kind="success")
            messagebox.showinfo("PDF Exported", f"Summary saved as PDF to:\n{path}")
        except Exception as error:
            self.set_status("PDF export failed.", kind="error")
            messagebox.showerror("Error", f"Could not export PDF: {error}")

    # -------------------- Analytics tab logic --------------------
    def _refresh_analytics(self):
        stats = fetch_stats()

        for widget in self.stats_cards_frame.winfo_children():
            widget.destroy()

        cards = [
            ("Total Summaries", stats["total"], PALETTE["accent"]),
            ("YouTube", stats["youtube_count"], PALETTE["error"]),
            ("Audio Files", stats["audio_count"], PALETTE["accent2"]),
            ("Meetings", stats["meeting_count"], PALETTE["warning"]),
        ]
        for title, value, color in cards:
            card = Frame(
                self.stats_cards_frame, bg=PALETTE["card"], highlightthickness=1,
                highlightbackground=PALETTE["border"],
            )
            card.pack(side=LEFT, expand=True, fill=X, padx=(0, 10))
            Label(
                card, text=str(value), bg=PALETTE["card"], fg=color,
                font=("Segoe UI", 20, "bold"),
            ).pack(pady=(12, 0))
            Label(
                card, text=title, bg=PALETTE["card"], fg=PALETTE["muted"],
                font=("Segoe UI", 9),
            ).pack(pady=(0, 12))

        # --- Lightweight bar chart (YouTube vs Audio vs Meetings) on a Canvas ---
        canvas = self.analytics_chart_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 300)
        max_count = max(stats["youtube_count"], stats["audio_count"], stats["meeting_count"], 1)
        bar_area_height = 120
        base_y = 150

        bars = [
            ("YouTube", stats["youtube_count"], PALETTE["error"]),
            ("Audio", stats["audio_count"], PALETTE["accent2"]),
            ("Meetings", stats["meeting_count"], PALETTE["warning"]),
        ]
        bar_width = 80
        gap = 60
        start_x = (width - (bar_width * len(bars) + gap * (len(bars) - 1))) / 2

        for i, (label, count, color) in enumerate(bars):
            x0 = start_x + i * (bar_width + gap)
            x1 = x0 + bar_width
            bar_height = (count / max_count) * bar_area_height if max_count else 0
            y0 = base_y - bar_height
            canvas.create_rectangle(x0, y0, x1, base_y, fill=color, outline="")
            canvas.create_text(
                (x0 + x1) / 2, y0 - 12, text=str(count),
                fill=PALETTE["text"], font=("Segoe UI", 11, "bold"),
            )
            canvas.create_text(
                (x0 + x1) / 2, base_y + 16, text=label,
                fill=PALETTE["muted"], font=("Segoe UI", 9),
            )
        canvas.create_line(20, base_y, width - 20, base_y, fill=PALETTE["border"])

        if stats["latest"]:
            latest_id, latest_time = stats["latest"]
            self.analytics_latest_label.config(
                text=f"Most recent: '{latest_id}' — {latest_time}"
            )
        else:
            self.analytics_latest_label.config(text="No summaries yet.")

    # -------------------- History tab logic (web-style card feed) --------------------
    def _refresh_history(self):
        for widget in self.history_feed.winfo_children():
            widget.destroy()

        self._history_rows = fetch_history()

        if not self._history_rows:
            Label(
                self.history_feed, text="No summaries yet — your history will show up here.",
                bg=PALETTE["card"], fg=PALETTE["muted"], font=("Segoe UI", 10, "italic"),
            ).pack(pady=30)
            self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))
            return

        for row_id, identifier, source_type, summary_text, created_at in self._history_rows:
            self._build_history_card(row_id, identifier, source_type, summary_text, created_at)

        self.history_feed.update_idletasks()
        self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all"))

    def _build_history_card(self, row_id, identifier, source_type, summary_text, created_at):
        icon, type_label = describe_source_type(source_type)
        snippet = summary_text.replace("\n", "  ").lstrip("• ").strip()
        if len(snippet) > 140:
            snippet = snippet[:140].rstrip() + "…"

        # Outer wrapper gives a subtle "card shadow" edge on a white card
        wrapper = Frame(self.history_feed, bg=PALETTE["card"])
        wrapper.pack(fill=X, padx=12, pady=6)

        card = Frame(
            wrapper, bg=PALETTE["panel"], highlightthickness=1,
            highlightbackground=PALETTE["border"],
        )
        card.pack(fill=X)

        inner = Frame(card, bg=PALETTE["panel"])
        inner.pack(fill=X, padx=16, pady=12)

        top_row = Frame(inner, bg=PALETTE["panel"])
        top_row.pack(fill=X)

        Label(
            top_row, text=f"{icon} {identifier}", bg=PALETTE["panel"], fg=PALETTE["text"],
            font=("Segoe UI", 11, "bold"), anchor=W,
        ).pack(side=LEFT)

        Label(
            top_row, text=f"{type_label} · {created_at}", bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Segoe UI", 8), anchor=E,
        ).pack(side=RIGHT)

        Label(
            inner, text=snippet, bg=PALETTE["panel"], fg=PALETTE["muted"],
            font=("Segoe UI", 9), anchor=W, justify=LEFT, wraplength=640,
        ).pack(fill=X, pady=(6, 10))

        btn_row = Frame(inner, bg=PALETTE["panel"])
        btn_row.pack(fill=X)

        ttk.Button(
            btn_row, text="👁️ View", style="Secondary.TButton",
            command=lambda rid=row_id: self.load_history_entry(rid),
        ).pack(side=LEFT, padx=(0, 8))

        ttk.Button(
            btn_row, text="⬇️ Download PDF", style="Accent.TButton",
            command=lambda rid=row_id: self.download_history_entry_pdf(rid),
        ).pack(side=LEFT, padx=(0, 8))

        ttk.Button(
            btn_row, text="🗑️ Delete", style="Secondary.TButton",
            command=lambda rid=row_id: self.delete_history_entry(rid),
        ).pack(side=LEFT)

        # Simple hover highlight so the feed feels interactive, like a web list
        def _on_enter(_e):
            card.configure(highlightbackground=PALETTE["accent"])

        def _on_leave(_e):
            card.configure(highlightbackground=PALETTE["border"])

        for widget in (card, inner, top_row):
            widget.bind("<Enter>", _on_enter)
            widget.bind("<Leave>", _on_leave)

    def load_history_entry(self, row_id):
        row = fetch_summary_by_id(row_id)
        if not row:
            return
        identifier, source_type, summary_text = row

        self.current_identifier = identifier
        self.current_summary = summary_text
        self.current_source_type = source_type
        self.current_source_ref = None

        self._render_summary(summary_text)
        self.save_button.config(state=NORMAL)
        self.pdf_button.config(state=NORMAL)
        self._show_page("youtube" if source_type == "youtube" else "audio")
        self.set_status(f"📖 Loaded '{identifier}' from history.", kind="success")

    def download_history_entry_pdf(self, row_id):
        """Download a PDF directly from a history card, without first
        loading it into the main summary view."""
        row = fetch_summary_by_id(row_id)
        if not row:
            return
        identifier, source_type, summary_text = row

        safe_identifier = re.sub(r"[^A-Za-z0-9_-]", "_", str(identifier))[:40]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{safe_identifier}_summary_{timestamp}.pdf"

        chosen_path = filedialog.asksaveasfilename(
            title="Save Summary as PDF",
            initialdir=str(self._get_default_save_folder()),
            initialfile=default_filename,
            defaultextension=".pdf",
            filetypes=[("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not chosen_path:
            return

        try:
            self.set_status("📄 Building PDF...")
            path = export_summary_to_pdf(
                summary_text, identifier, out_path=Path(chosen_path), source_type=source_type,
            )
            self.set_status("✅ PDF exported.", kind="success")
            messagebox.showinfo("PDF Exported", f"Summary saved as PDF to:\n{path}")
        except Exception as error:
            self.set_status("PDF export failed.", kind="error")
            messagebox.showerror("Error", f"Could not export PDF: {error}")

    def delete_history_entry(self, row_id):
        if not messagebox.askyesno("Confirm delete", "Delete this summary from the database?"):
            return
        delete_summary(row_id)
        self._refresh_history()
        self._refresh_analytics()
        self.set_status("🗑️ Entry deleted.", kind="info")


# ---------------------------------------------------------
# Run the app — shows the login window first, then the
# main Media Summarizer window on successful login. Logging
# out returns the user to the login window (no full restart).
# ---------------------------------------------------------

def _launch_login(root):
    for widget in root.winfo_children():
        widget.destroy()
    root.config(menu="")  # clear the app's menu bar when back at login
    LoginWindow(root, on_success=lambda username: _launch_main_app(root, username))


def _launch_main_app(root, username):
    for widget in root.winfo_children():
        widget.destroy()
    MediaSummarizerApp(root, username=username, on_logout=lambda: _launch_login(root))


if __name__ == "__main__":
    root = Tk()
    _launch_login(root)
    root.mainloop()