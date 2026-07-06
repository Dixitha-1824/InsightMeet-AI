import tempfile
import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ AI Meeting Assistant")
st.write("Upload a meeting recording or paste a YouTube URL.")

# ---------------- Sidebar ---------------- #

st.sidebar.title("Input Source")

input_type = st.sidebar.radio(
    "Choose Input",
    ["Upload File", "YouTube URL"]
)

source = None

if input_type == "Upload File":

    uploaded_file = st.sidebar.file_uploader(
        "Upload Audio / Video",
        type=[
            "mp3",
            "wav",
            "mp4",
            "m4a",
            "mov",
            "avi",
            "mkv"
        ]
    )

    if uploaded_file:

        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.read())
        source = temp.name

else:

    youtube_url = st.sidebar.text_input(
        "YouTube URL"
    )

    if youtube_url:
        source = youtube_url

# ---------------- Run Pipeline ---------------- #

if st.sidebar.button("Generate Report"):

    if source is None:

        st.warning("Please upload a file or enter a YouTube URL.")

    else:

        with st.spinner("Processing meeting..."):

            result = run_pipeline(source)

        st.session_state["result"] = result

        st.success("Meeting processed successfully!")

# ---------------- Results ---------------- #

if "result" in st.session_state:

    result = st.session_state["result"]

    st.header("📌 Meeting Title")

    st.info(result["title"])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary",
        "Action Items",
        "Key Decisions",
        "Open Questions",
        "Transcript"
    ])

    with tab1:

        st.markdown(result["summary"])

    with tab2:

        st.markdown(result["action_items"])

    with tab3:

        st.markdown(result["key_decisions"])

    with tab4:

        st.markdown(result["open_questions"])

    with tab5:

        st.text_area(
            "Transcript",
            result["transcript"],
            height=450
        )

    st.divider()

    st.header("💬 Chat with Meeting")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.chat_input("Ask something about this meeting...")

    if question:

        rag_chain = result["rag_chain"]

        answer = ask_question(
            rag_chain,
            question
        )

        st.session_state.chat_history.append(
            ("You", question)
        )

        st.session_state.chat_history.append(
            ("Assistant", answer)
        )

    for sender, message in st.session_state.chat_history:

        with st.chat_message(sender):

            st.write(message)