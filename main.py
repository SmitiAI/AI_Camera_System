import streamlit as st
import cv2
from ultralytics import YOLO
import io
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETTINGS & CACHING ---
st.set_page_config(page_title="AI Engineering Workbench", layout="wide")
st.title("Electronics AI Vision & Document Assistant")


@st.cache_resource
def get_models():
    vision_model = YOLO('yolo11n.pt')
    # Local Llama3 configuration
    llm = ChatOllama(model="llama3", base_url="http://127.0.0.1:11434")
    # Configured a dedicated embedding model (Make sure you ran 'ollama pull nomic-embed-text' in terminal)
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://127.0.0.1:11434")
    return vision_model, llm, embeddings


vision_model, llm, embeddings = get_models()

# Ensure retriever state stays active across user interactions
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- 2. SIDEBAR: DOCUMENT BRAIN ---
with st.sidebar:
    st.header("Document Assistant")
    uploaded_file = st.file_uploader("Upload Component Datasheet", type="pdf")
    chat_query = st.text_input("Ask a technical question:", key="user_query")

    if uploaded_file:
        @st.cache_resource
        def process_pdf_with_pdfminer(file_bytes):
            from pdfminer import high_level

            # Read directly from RAM bytes stream
            pdf_file = io.BytesIO(file_bytes)

            # High-level text extraction to handle formatting elegantly
            raw_text = high_level.extract_text(pdf_file)

            if not raw_text.strip():
                st.error("Could not extract legible text layer from this PDF.")
                return None

            # Splitting text into dense blocks optimized for reference parameters
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=50
            )

            splits = text_splitter.create_documents([raw_text])

            # Index securely into FAISS Vector DB
            vectorstore = FAISS.from_documents(splits, embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 2})


        retriever_obj = process_pdf_with_pdfminer(uploaded_file.getvalue())

        if retriever_obj:
            st.session_state.retriever = retriever_obj
            st.success("Datasheet Successfully Indexed!")

# --- 3. MAIN AREA: AI VISION FEED ---
col1, col2 = st.columns([2, 1])

# Initialize toggle state before rendering the widget
if "cam_toggle" not in st.session_state:
    st.session_state.cam_toggle = False

# Automatically turn off camera switch if user submits a text question
if chat_query and st.session_state.retriever:
    st.session_state.cam_toggle = False

with col1:
    st.subheader("Live Component Vision")
    run_vision = st.toggle("Power On Camera", key="cam_toggle")


# Fragment engine controls independent camera updates
@st.fragment(run_every=0.1)
def camera_engine():
    if st.session_state.cam_toggle:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            results = vision_model(frame)
            annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
            st.image(annotated_frame)


# Call the engine loop
camera_engine()

# --- 4. SEPARATE CHAT LOGIC ---
if chat_query and st.session_state.retriever:
    with st.spinner("Analyzing datasheet records..."):
        # Retrieve context from FAISS
        docs = st.session_state.retriever.invoke(chat_query)
        context = "\n\n".join([d.page_content for d in docs])

        # Structure the query block
        prompt = (
            f"You are a helpful expert electronics assistant. Use the following context items "
            f"extracted from a component datasheet to accurately answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {chat_query}\n\n"
            f"Answer:"
        )

        response = llm.invoke(prompt)
        st.markdown(f"### Analysis Result:\n{response.content}")