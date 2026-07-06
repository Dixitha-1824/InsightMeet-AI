import whisper
import os
import requests
from pydub import AudioSegment
import re



WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():

    global _model  

    if _model is None: 
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL) 
        print("Whisper model loaded.")
    return _model 


def transcribe_chunk(chunk_path: str,translate: bool = False) -> str:

    model = load_model()  

    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        chunk_path,
        task=task,
        fp16=False,
        language="en",
        temperature=0,
        condition_on_previous_text=True,
    )
    return result["text"]  

def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, translate=translate)

        full_transcript += text

    print("Transcription completed")

    return clean_transcript(full_transcript)


def clean_transcript(transcript: str) -> str:
    """
    Clean Whisper transcript before sending it to the LLM.
    """

    # Remove extra spaces
    transcript = re.sub(r"\s+", " ", transcript)

    # Remove repeated punctuation
    transcript = re.sub(r"([.,!?])\1+", r"\1", transcript)

    # Remove extra blank lines
    transcript = re.sub(r"\n\s*\n", "\n", transcript)

    return transcript.strip()