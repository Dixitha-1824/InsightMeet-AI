from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.0,
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert AI Meeting Assistant.

Analyze the meeting transcript carefully and extract EVERY action item.

Rules:
- Do not invent information.
- If an owner is not mentioned, write "Unknown".
- If a deadline is not mentioned, write "Not specified".
- Include only genuine tasks assigned during the meeting.
- Ignore suggestions that were not agreed upon.

For every action item provide:

1. Task
2. Owner
3. Deadline
4. Priority (High / Medium / Low if inferable, otherwise Unknown)
5. Evidence (brief sentence from the transcript)

Return the results as a numbered list.

If no action items exist, return exactly:

No action items found.
"""
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert AI Meeting Assistant.

Extract all important decisions that were finalized during the meeting.

Rules:
- Include only confirmed decisions.
- Ignore ideas that were merely discussed.
- Do not invent information.

For each decision include:

1. Decision
2. Reason (if mentioned)
3. Decision Maker (if known)

Return as a numbered list.

If no confirmed decisions exist, return exactly:

No key decisions found.
"""
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert AI Meeting Assistant.

Extract all unanswered questions, unresolved issues, or topics requiring follow-up.

Rules:
- Ignore questions that were fully answered.
- Include pending discussions.
- Do not invent information.

For each item include:

1. Question / Issue
2. Raised By (if known)
3. Suggested Follow-up (if mentioned)

Return as a numbered list.

If no unresolved questions exist, return exactly:

No open questions found.
"""
    )

    return chain.invoke(transcript)