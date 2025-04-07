from typing import Literal, Optional
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig:
    """Centralized configuration for LLMs and embeddings"""
    
    def __init__(
        self,
        llm_type: Literal["google", "openai", "ollama"] = "google",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        chunk_size: int = 1000
    ):
        self.llm_type = llm_type
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.max_tokens = max_tokens
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            chunk_size=chunk_size
        )
        
        # Initialize LLM based on type
        if llm_type == "google":
            self.llm = ChatGoogleGenerativeAI(
                model=model_name or os.getenv("GOOGLE_MODEL_NAME"),
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif llm_type == "openai":
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT_NAME"),
                model_name= model_name or os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL_NAME"),
                temperature=temperature,
                max_tokens=max_tokens,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
        elif llm_type == "ollama":
            self.llm = Ollama(
                model=model_name or "gemma3",
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}") 