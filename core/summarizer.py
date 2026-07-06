from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.1,
    )


def split_transcript(transcript: str) -> list:
    """Split long transcript into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Generate a professional meeting summary."""

    llm = get_llm()

    # -----------------------------
    # Step 1 : Summarize each chunk
    # -----------------------------
    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert AI Meeting Assistant.

Summarize this section of the meeting transcript.

Focus only on:

• Main discussion points
• Important decisions
• Action items
• Deadlines
• Risks or blockers
• Next steps

Ignore:
- Greetings
- Small talk
- Repeated conversations
- Filler words

Return concise bullet points.
""",
        ),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)

    # -----------------------------
    # Step 2 : Combine summaries
    # -----------------------------
    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert AI Meeting Assistant.

Using the partial meeting summaries, generate a professional meeting summary.

Use the following format:

# Meeting Overview
A short paragraph describing the purpose of the meeting.

# Key Discussion Points
- Bullet points

# Decisions Made
- Bullet points

# Action Items
- Task — Owner (if known) — Deadline (if mentioned)

# Risks / Blockers
- Bullet points
(If none, write "No risks discussed.")

# Next Steps
- Bullet points

Rules:
- Do not invent information.
- Include only facts supported by the transcript.
- Keep the summary concise and professional.
""",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    return combined_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    """Generate a short professional meeting title."""

    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                """
Generate a concise professional meeting title.

Rules:
- Maximum 8 words.
- Capture the primary topic of the meeting.
- Do not include quotation marks.
- Do not end with punctuation.
- Return ONLY the title.
""",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:3000])