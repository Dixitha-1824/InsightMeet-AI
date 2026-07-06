from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()

import time


def run_pipeline(source: str) -> dict:
    print("=" * 60)
    print("🚀 Starting AI Meeting Assistant")
    print("=" * 60)

    start_time = time.time()

    try:
        # Step 1 : Process input
        chunks = process_input(source)

        # Step 2 : Transcribe
        transcript = transcribe_all(chunks)
        print(f"\nTranscript Preview:\n{transcript[:300]}...\n")

        # Step 3 : Generate outputs
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        # Step 4 : Build RAG
        rag_chain = build_rag_chain(transcript)

        elapsed = time.time() - start_time
        print(f"Pipeline completed successfully in {elapsed:.2f} seconds.\n")

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
if __name__ == "__main__":
    print("helo")

    source = input("Enter YouTube URL or local file path:\n").strip()

    result = run_pipeline(source)

    print("\n" + "=" * 70)
    print("📄 MEETING REPORT")
    print("=" * 70)

    print(f"\n📌 Title\n{result['title']}")

    print(f"\n📝 Summary\n{result['summary']}")

    print(f"\n✅ Action Items\n{result['action_items']}")

    print(f"\n📌 Key Decisions\n{result['key_decisions']}")

    print(f"\n❓ Open Questions\n{result['open_questions']}")

    print("\n" + "=" * 70)

    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("👋 Thank you for using AI Meeting Assistant.")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\nAssistant: {answer}\n")