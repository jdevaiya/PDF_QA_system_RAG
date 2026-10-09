import os
import streamlit as st
from dotenv import load_dotenv

from src.loading import loader
from src.splitting import splitter
from src.embedding import embeddings
from src.retriever import get_retriever
from src.llm import get_llm
from src.rag_chain import get_rag_chain

load_dotenv()

st.set_page_config(
    page_title="PDF Q&A System using RAG",
    page_icon="📄",
    layout="wide"
)

# --- Initialize Session State ---
if "db" not in st.session_state:
    st.session_state.db = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "processed_provider" not in st.session_state:
    st.session_state.processed_provider = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "prefill_question" not in st.session_state:
    st.session_state.prefill_question = ""


# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ System Configuration")

    # Embedding Provider Configuration
    st.subheader("1. 🧠 Embedding Provider")
    embedding_provider = st.selectbox(
        "Choose Embedding Provider:",
        options=["auto", "google", "huggingface"],
        format_func=lambda x: {
            "auto": "⚡ Auto (Google with HuggingFace Fallback)",
            "google": "🌐 Google Gemini Embeddings",
            "huggingface": "💻 Hugging Face (all-MiniLM-L6-v2 - Local & Free)"
        }[x],
        index=0,
        help="In 'Auto' mode, uses Google Gemini embeddings and automatically switches to local HuggingFace if Google is rate-limited (429) or down."
    )

    # LLM Generation Provider Configuration
    st.subheader("2. 🤖 LLM / Generation Provider")
    model_provider = st.selectbox(
        "Choose LLM Provider:",
        options=["auto", "google", "groq"],
        format_func=lambda x: {
            "auto": "⚡ Auto (Google with Groq Fallback)",
            "google": "🌐 Google Gemini Only",
            "groq": "🚀 Groq (openai-oss-120b) Only"
        }[x],
        index=0,
        help="In 'Auto' mode, queries run on Google Gemini and automatically switch to Groq if Google's key is exhausted or down."
    )

    st.divider()
    st.subheader("🔑 API Keys Status")
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    has_google = bool(google_key and not google_key.startswith("your_"))
    has_groq = bool(groq_key and not groq_key.startswith("your_"))

    st.write(f"• **Google Gemini:** {'🟢 Configured' if has_google else '🔴 Missing'}")
    st.write(f"• **Groq:** {'🟢 Configured' if has_groq else '🔴 Missing'}")
    st.write(f"• **HuggingFace Local:** 🟢 Ready (No Key Required)")

    st.divider()
    if st.button("🗑️ Clear Q&A History", use_container_width=True):
        st.session_state.qa_history = []
        st.rerun()


# --- Main Application Header ---
st.title("📄 PDF Q&A System using RAG")
st.caption("Upload your document, ask questions with dedicated sending controls, and receive answers powered by Gemini & Groq.")

# --- Document Upload Section ---
st.subheader("1. 📂 Upload Document")
file = st.file_uploader("Upload a PDF file to analyze:", type=['pdf'])

# Process file only when a new file is uploaded or provider changed
if file:
    needs_processing = (
        st.session_state.db is None or
        st.session_state.processed_file != file.name or
        st.session_state.processed_provider != embedding_provider
    )

    if needs_processing:
        with st.spinner("⏳ Loading, chunking, and embedding document into vector store..."):
            documents = loader(file)
            chunks = splitter(documents)
            db = embeddings(chunks, provider=embedding_provider)

            if db is None:
                st.error("❌ Embedding generation failed. Please check your configuration.")
                st.stop()

            st.session_state.db = db
            st.session_state.processed_file = file.name
            st.session_state.processed_provider = embedding_provider
            st.session_state.qa_history = []  # Reset Q&A history for new document
            st.success(f"✅ Successfully processed **{file.name}** ({len(chunks)} chunks indexed)!")
    else:
        st.success(f"✅ Ready: **{file.name}** is loaded and ready for questions.")

    st.divider()

    # --- Question Sending Section ---
    st.subheader("2. 💬 Ask a Question")

    # Quick question suggestions
    st.markdown("**Quick Suggestions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 Summarize this PDF", use_container_width=True):
            st.session_state.prefill_question = "Can you provide a comprehensive summary of this entire document?"
            st.rerun()
    with col2:
        if st.button("🔑 Key Takeaways", use_container_width=True):
            st.session_state.prefill_question = "What are the main key points, conclusions, and takeaways in this document?"
            st.rerun()
    with col3:
        if st.button("🔍 Explain Main Topics", use_container_width=True):
            st.session_state.prefill_question = "What are the primary topics covered in this document and how are they explained?"
            st.rerun()

    # Question Input Form with Send Button
    current_val = st.session_state.prefill_question or ""
    with st.form(key="qa_form", clear_on_submit=False):
        user_query = st.text_input(
            "Enter your question about the PDF:",
            value=current_val,
            placeholder="e.g., Explain the core concepts discussed in this document...",
            key="input_question"
        )
        
        c_left, c_right = st.columns([4, 1])
        with c_left:
            show_chunks = st.checkbox("Show retrieved source chunks with answer", value=True)
        with c_right:
            submit_button = st.form_submit_button("🚀 Send Question", type="primary", use_container_width=True)

    # Process Question when Send button is pressed
    if submit_button:
        # Clear the prefill state so it doesn't persist
        st.session_state.prefill_question = ""
        
        cleaned_query = user_query.strip()
        if not cleaned_query:
            st.warning("⚠️ Please type a question before sending.")
        else:
            retrieved_docs = get_retriever(st.session_state.db)

            try:
                llm = get_llm(provider=model_provider)
                rag_chain = get_rag_chain(retrieved_docs, llm)
            except Exception as e:
                st.error(f"❌ Failed to initialize model: {str(e)}")
                st.stop()

            with st.spinner("🔍 Retrieving context and generating answer..."):
                try:
                    # Retrieve relevant context documents
                    docs = retrieved_docs.invoke(cleaned_query)
                    
                    # Generate the answer using RAG chain
                    answer = rag_chain.invoke(cleaned_query)
                    
                    # Determine provider label for display
                    provider_label = {
                        "auto": "⚡ Auto (Gemini / Groq)",
                        "google": "🌐 Google Gemini",
                        "groq": "🚀 Groq Llama 3.3"
                    }.get(model_provider, model_provider)

                    # Store in Q&A session history
                    st.session_state.qa_history.insert(0, {
                        "question": cleaned_query,
                        "answer": answer,
                        "docs": docs,
                        "provider": provider_label,
                        "show_chunks": show_chunks
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

    # --- Show Answer Section ---
    if st.session_state.qa_history:
        st.divider()
        st.subheader("3. 💡 Answers & Q&A History")

        # Display latest answer prominently
        latest = st.session_state.qa_history[0]
        
        with st.container(border=True):
            st.markdown(f"### ❓ Question:\n**{latest['question']}**")
            st.caption(f"Provider: {latest['provider']}")
            
            st.markdown("### 💡 Answer:")
            st.markdown(latest["answer"])

            # Show retrieved source chunks if requested
            if latest.get("show_chunks") and latest.get("docs"):
                with st.expander("📚 View Retrieved Source Context Chunks"):
                    for i, doc in enumerate(latest["docs"]):
                        page_num = doc.metadata.get("page", None)
                        page_info = f"Page {page_num + 1}" if isinstance(page_num, int) else "Page N/A"
                        st.markdown(f"**Chunk {i+1} ({page_info}):**")
                        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                        st.divider()

        # Display prior questions & answers in an accordion if multiple questions were asked
        if len(st.session_state.qa_history) > 1:
            st.markdown("#### 📜 Previous Questions & Answers:")
            for idx, item in enumerate(st.session_state.qa_history[1:], start=2):
                with st.expander(f"Q#{idx}: {item['question'][:80]}..."):
                    st.markdown(f"**Question:** {item['question']}")
                    st.caption(f"Provider: {item['provider']}")
                    st.markdown(f"**Answer:**\n{item['answer']}")
                    if item.get("docs"):
                        st.markdown("**Sources:**")
                        for d_idx, doc in enumerate(item["docs"]):
                            page_num = doc.metadata.get("page", None)
                            page_info = f"Page {page_num + 1}" if isinstance(page_num, int) else "Page N/A"
                            st.caption(f"Chunk {d_idx+1} ({page_info}): {doc.page_content[:150]}...")

else:
    # No file uploaded yet
    st.info("👆 Please upload a PDF file above to start asking questions.")
