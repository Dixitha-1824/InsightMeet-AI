# 🎥InsightMeet AI – Intelligent Meeting Analysis RAG Assistan

An AI-powered Meeting Assistant that processes YouTube videos and local media files to generate accurate transcriptions, concise summaries, and intelligent question-answering using Retrieval-Augmented Generation (RAG). The application leverages OpenAI Whisper for speech-to-text conversion and Mistral AI for natural language understanding, providing an interactive interface built with Streamlit.

---

## 🚀 Features

- Process YouTube videos and local audio/video files.
- Convert speech to text using OpenAI Whisper.
- Generate AI-powered meeting summaries with Mistral AI.
- Ask questions about meeting content using Retrieval-Augmented Generation (RAG).
- Perform semantic search using vector embeddings for context-aware responses.
- Interactive and user-friendly Streamlit interface.
- Supports efficient knowledge retrieval from long meeting recordings.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- Mistral AI
- LangChain
- FAISS (Vector Database)
- yt-dlp
- FFmpeg

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/VideoAssistant.git
cd VideoAssistant
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
MISTRAL_API_KEY=your_api_key_here
```

### Run the application

```bash
streamlit run app.py
```

---

## 💡 How It Works

1. Upload a local audio/video file or provide a YouTube URL.
2. Extract audio from the media.
3. Transcribe speech into text using OpenAI Whisper.
4. Generate vector embeddings and store them in a FAISS vector database.
5. Retrieve relevant context using semantic search (RAG).
6. Generate summaries and answer user queries using Mistral AI.

---

## 🎯 Use Cases

- Meeting Summarization
- Lecture and Webinar Analysis
- Interview Analysis
- YouTube Video Understanding
- Knowledge Retrieval from Recorded Sessions
- AI-Powered Meeting Assistant

---
