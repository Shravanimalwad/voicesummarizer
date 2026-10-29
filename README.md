# 🎙️ Offline Video, Audio & Meeting Summarizer

An offline desktop application built with **Python, Tkinter, OpenAI Whisper, and SQLite** to transcribe and summarize YouTube videos, audio files, and meeting recordings. The application provides user account management, transcription history, analytics, and PDF export through a simple graphical interface.

## ✨ Features

- **🎥 YouTube Video Summarization** — Extract and summarize audio from YouTube videos.
- **🎵 Audio File Transcription** — Convert supported audio files into text.
- **🎤 Meeting Recording Summarization** — Process recorded meetings and generate concise summaries.
- **🤖 AI-Powered Transcription** — Use Whisper for automatic speech recognition.
- **📝 Extractive Text Summarization** — Generate concise summaries by identifying important sentences from transcripts.
- **👥 Multi-User Accounts** — Maintain separate user accounts and user-specific activity.
- **🗃️ SQLite Database** — Store user information, transcripts, summaries, and processing history locally.
- **📊 Analytics & History** — Track previous summarization tasks and review activity.
- **📄 PDF Export** — Export generated summaries for offline access and sharing.
- **🖥️ Desktop GUI** — Interact with application features through a Tkinter-based interface.
- **🔒 Local Data Storage** — Keep application records on the local machine without requiring a cloud database.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Tkinter | Desktop graphical user interface |
| OpenAI Whisper | Speech-to-text transcription |
| SQLite | Local database and user history |
| Extractive Summarization | Generate concise summaries from transcripts |
| PDF Generation Library | Export summaries as PDF documents |

## 🏗️ Application Workflow

1. **Select Input:** Provide a YouTube video link, an audio file, or a recorded meeting.
2. **Extract Audio:** Retrieve or prepare the audio for processing.
3. **Transcribe:** Use Whisper to convert spoken content into text.
4. **Summarize:** Apply an extractive summarization algorithm to identify important sentences.
5. **Store Results:** Save transcripts, summaries, and task history in SQLite.
6. **Review & Export:** View previous results, access analytics, and export summaries as PDF files.

## 📂 Project Structure

The following is a suggested structure. Adjust the filenames to match your actual project.

```text
offline-video-audio-summarizer/
│
├── main.py                 # Application entry point
├── gui.py                  # Tkinter user interface
├── transcription.py        # Whisper transcription logic
├── summarizer.py            # Extractive summarization
├── database.py              # SQLite database operations
├── youtube_handler.py       # YouTube audio processing
├── pdf_export.py             # PDF generation and export
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── models/                   # Optional local Whisper models
├── exports/                  # Generated PDF summaries
├── data/                     # Local database files
└── assets/                   # Icons and other UI resources
```

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or a compatible version supported by your dependencies.
- FFmpeg installed and available on your system PATH.
- Sufficient disk space for Whisper model files.
- A microphone and suitable audio input configuration if live recording is supported.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/offline-video-audio-summarizer.git
cd offline-video-audio-summarizer
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

If your project includes a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Typical dependencies may include:

```text
openai-whisper
torch
yt-dlp
nltk
reportlab
```

Tkinter and SQLite are included with many standard Python installations, although Tkinter may require a separate operating-system package on some Linux distributions.

Only include dependencies that your implementation actually uses.

### 4. Install FFmpeg

Whisper requires FFmpeg to process many audio formats.

- **Windows:** Install FFmpeg and add its `bin` directory to PATH.
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

Verify the installation:

```bash
ffmpeg -version
```

### 5. Download the Whisper Model

Whisper model weights are typically downloaded the first time a model is loaded. For a fully offline setup, download the required model beforehand and verify that it can be loaded without internet access.

Available model sizes include `tiny`, `base`, `small`, `medium`, and `large`. Smaller models generally require fewer resources, while larger models can provide better transcription accuracy.

### 6. Run the Application

Use the appropriate entry-point filename for your project:

```bash
python main.py
```

## 📴 Offline Usage

The application is designed for local processing and storage wherever supported by the implementation.

For offline operation:

- Install Python dependencies before disconnecting from the internet.
- Download the required Whisper model files in advance.
- Configure the application to load the local model instead of downloading it.
- Store the SQLite database and exported PDFs locally.
- Ensure that all audio-processing utilities are installed.

**Note:** YouTube video processing normally requires an internet connection to retrieve the video or its audio. To use the application completely offline, use locally available audio files or previously downloaded recordings. The application should not be described as fully offline for every input type unless this has been verified.

## 🗄️ Database

SQLite provides persistent local storage for application data.

Depending on the implementation, the database may contain:

- User account information
- Transcription records
- Generated summaries
- Processing timestamps
- User-specific history
- Activity and usage statistics

A typical conceptual schema might include:

| Table | Purpose |
|---|---|
| `users` | User accounts and authentication data |
| `transcriptions` | Transcript content and source information |
| `summaries` | Generated summaries |
| `history` | Processing activity and timestamps |

The actual table names and relationships depend on your database implementation.

## 📄 PDF Export

Users can export generated summaries into PDF documents for:

- Meeting notes
- Lecture summaries
- Video content reviews
- Personal study material
- Documentation and reporting

Exported files can be saved locally and accessed without an internet connection.

## 🔐 Privacy & Security

- Application records are stored in a local SQLite database.
- Local Whisper inference can process audio without sending it to a cloud transcription service.
- User passwords should be stored as secure password hashes rather than plain text.
- Database access and file operations should be handled carefully.
- Sensitive transcripts and recordings should be protected from unauthorized local access.

YouTube retrieval, dependency installation, and initial model downloads may require internet access.

## 🚧 Limitations

- Transcription accuracy depends on audio quality, language, accents, and background noise.
- Large Whisper models may require substantial RAM and processing time.
- CPU-based transcription may be slower than GPU-based transcription.
- Extractive summarization selects existing sentences and may not produce the natural wording of an abstractive language model.
- Live meeting transcription depends on the application's recording and audio-capture implementation.
- YouTube processing depends on network availability and the video's accessibility.

## 🔮 Future Enhancements

- Support for additional audio and video formats.
- Multilingual transcription and language selection.
- Speaker diarization to identify different speakers.
- Searchable transcript history and advanced filtering.
- Improved summarization with configurable summary length.
- Export to TXT, DOCX, and other document formats.
- Progress indicators and background processing for long recordings.
- Packaging as a standalone executable for Windows, Linux, and macOS.
- Enhanced database security and role-based access control.


---

*Built with Python to make video, audio, and meeting content easier to understand, organize, and review.*
