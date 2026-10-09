import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_google(model="gemini-3.8-flash", temperature=0.3, max_tokens=500):
    """
    Initializes and returns the Google Gemini LLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GOOGLE_API_KEY is not set or invalid in environment/.env file.")
    
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        google_api_key=api_key
    )


def get_llm_groq(model="openai/gpt-oss-120b", temperature=0.3, max_tokens=500):
    """
    Initializes and returns the Groq LLM.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.startswith("your_"):
        raise ValueError("GROQ_API_KEY is not set or invalid in environment/.env file.")
    
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=api_key
    )


def _wrap_groq_fallback(groq_llm):
    """
    Wraps Groq LLM with a notification toast if called within Streamlit as a fallback.
    """
    def _fallback_invoke(prompt_input):
        logger.warning("Google Gemini call failed. Falling back to Groq Llama model...")
        try:
            import streamlit as st
            st.toast("⚠️ Google Gemini unavailable or rate-limited. Automatically switched to Groq Llama 3.3!", icon="🚀")
        except Exception:
            pass
        return groq_llm.invoke(prompt_input)
    
    return RunnableLambda(_fallback_invoke)


def get_llm(
    provider="auto",
    google_model="gemini-3.8-flash",
    groq_model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=500
):
    
    provider = (provider or "auto").lower().strip()

    if provider == "google":
        return get_llm_google(model=google_model, temperature=temperature, max_tokens=max_tokens)

    if provider == "groq":
        return get_llm_groq(model=groq_model, temperature=temperature, max_tokens=max_tokens)

    if provider == "auto":
        google_llm = None
        groq_llm = None
        google_err = None
        groq_err = None

        try:
            google_llm = get_llm_google(model=google_model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            google_err = e
            logger.warning(f"Google Gemini init skipped: {e}")

        try:
            groq_llm = get_llm_groq(model=groq_model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            groq_err = e
            logger.warning(f"Groq init skipped: {e}")

        # If both are available, chain Google with Groq as fallback
        if google_llm and groq_llm:
            fallback_target = _wrap_groq_fallback(groq_llm)
            return google_llm.with_fallbacks([fallback_target])
        elif google_llm:
            return google_llm
        elif groq_llm:
            return groq_llm
        else:
            raise ValueError(
                f"Neither Google Gemini nor Groq could be initialized.\n"
                f"Google: {google_err}\n"
                f"Groq: {groq_err}\n"
                f"Please ensure at least one valid API key (GOOGLE_API_KEY or GROQ_API_KEY) is set in your .env file."
            )

    raise ValueError(f"Unknown provider '{provider}'. Must be 'auto', 'google', or 'groq'.")