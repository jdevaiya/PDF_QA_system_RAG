import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)


def get_google_embeddings(model="gemini-embedding-001"):
    """
    Initializes and returns Google Generative AI Embeddings.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY is not set or invalid in environment/.env file.")
    
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key
    )


def get_huggingface_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    Initializes and returns local HuggingFace embeddings (runs on CPU, no API key required).
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )


def embeddings(chunks, provider="auto", index_path="faiss_index"):
    
    provider = (provider or "auto").lower().strip()

    # 1. HuggingFace explicit selection
    if provider == "huggingface":
        try:
            embed_model = get_huggingface_embeddings()
            db = FAISS.from_documents(chunks, embed_model)
            db.save_local(index_path)
            return db
        except Exception as e:
            logger.error(f"Hugging Face embeddings failed: {e}")
            _notify_ui_error(f"❌ Hugging Face embedding error: {str(e)}")
            return None

    # 2. Google explicit selection
    elif provider == "google":
        try:
            embed_model = get_google_embeddings()
            db = FAISS.from_documents(chunks, embed_model)
            db.save_local(index_path)
            return db
        except Exception as e:
            logger.error(f"Google embeddings failed: {e}")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                _notify_ui_error("🚫 Google API quota/rate limit exhausted (429)! Please try later or switch to Hugging Face.")
            else:
                _notify_ui_error(f"❌ Google embedding error: {str(e)}")
            return None

    # 3. Auto mode: Google primary with HuggingFace fallback
    elif provider == "auto":
        # First attempt Google Gemini
        try:
            embed_model = get_google_embeddings()
            db = FAISS.from_documents(chunks, embed_model)
            db.save_local(index_path)
            return db
        except Exception as google_err:
            logger.warning(f"Google embedding failed ({google_err}). Falling back to local Hugging Face model...")
            _notify_ui_warning(
                "⚠️ Google embedding API is down, rate-limited (429), or missing key. "
                "Automatically switching to local Hugging Face embeddings (all-MiniLM-L6-v2)..."
            )

            # Fallback to local HuggingFace embeddings
            try:
                embed_model = get_huggingface_embeddings()
                db = FAISS.from_documents(chunks, embed_model)
                db.save_local(index_path)
                _notify_ui_success("✅ Embedded successfully using local Hugging Face fallback model!")
                return db
            except Exception as hf_err:
                logger.error(f"Local Hugging Face fallback also failed: {hf_err}")
                _notify_ui_error(
                    f"❌ Both Google and Hugging Face embedding attempts failed.\n"
                    f"Google error: {google_err}\n"
                    f"Hugging Face error: {hf_err}"
                )
                return None
    else:
        raise ValueError(f"Unknown embedding provider '{provider}'. Must be 'auto', 'google', or 'huggingface'.")


def _notify_ui_warning(message):
    try:
        import streamlit as st
        st.warning(message)
    except Exception:
        pass


def _notify_ui_success(message):
    try:
        import streamlit as st
        st.success(message)
    except Exception:
        pass


def _notify_ui_error(message):
    try:
        import streamlit as st
        st.error(message)
    except Exception:
        pass