"""
LLM Connector Module
====================

This module manages interaction with a local Ollama LLM instance via LangChain.
It handles model initialization, prompt submission, and response generation.

Example
-------
    from rag.llm_connector import get_llm_response

    answer = get_llm_response("Summarize my resume skills.", model_name="gemma:2b")
    print(answer)
"""

from langchain_ollama import OllamaLLM  # or ChatOllama
from src.rag.exceptions import RAGBaseException
from src.rag.logger import get_logger

logger = get_logger(__name__)


class LLMInitializationError(RAGBaseException):
    """Raised when an Ollama LLM model fails to initialize."""
    def __init__(self, message="Failed to initialize local LLM via Ollama."):
        super().__init__(message)


def get_local_llm(model_name: str = "gemma:2b", temperature: float = 0.2) -> OllamaLLM:
    """
    Initialize and return a local Ollama model instance.

    Parameters
    ----------
    model_name : str, optional
        The name of the Ollama model to use (default is "gemma:2b").
    temperature : float, optional
        Sampling temperature controlling creativity (default is 0.3).

    Returns
    -------
    Ollama
        An instance of LangChain's Ollama wrapper.

    Raises
    ------
    LLMInitializationError
        If the model cannot be initialized properly.
    """
    try:
        logger.info(f"Initializing Ollama model '{model_name}' with temperature={temperature}")
        llm = OllamaLLM(model=model_name, temperature=temperature)
        return llm
    except Exception as e:
        logger.exception("Error while initializing Ollama model.")
        raise LLMInitializationError(str(e)) from e


def get_llm_response(prompt: str, model_name: str = "gemma:2b", temperature: float = 0.2) -> str:
    """
    Generate a response from the local Ollama LLM for a given prompt.

    Parameters
    ----------
    prompt : str
        User prompt or query text.
    model_name : str, optional
        The Ollama model name to use.
    temperature : float, optional
        Sampling temperature for generation.

    Returns
    -------
    str
        Generated response text.

    Raises
    ------
    RAGBaseException
        If generation fails.
    """
    llm = get_local_llm(model_name=model_name, temperature=temperature)
    try:
        logger.debug("Generating response from LLM for the given prompt >>>")
        response = llm.invoke(prompt)
        logger.info("LLM response successfully generated.")
        return response
    except Exception as e:
        logger.exception("Error during LLM text generation.")
        raise RAGBaseException(f"LLM response generation failed: {e}") from e
