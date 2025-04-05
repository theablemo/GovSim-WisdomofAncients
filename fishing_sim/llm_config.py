from typing import Literal, Optional
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI, ChatGooglePalm
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMConfig:
    """Centralized configuration for LLMs and embeddings"""
    
    def __init__(
        self,
        llm_type: Literal["google", "azure", "ollama"] = "google",
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        chunk_size: int = 1000
    ):
        self.llm_type = llm_type
        self.model_name = model_name
        self.temperature = temperature
        self.chunk_size = chunk_size
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            chunk_size=chunk_size
        )
        
        # Initialize LLM based on type
        if llm_type == "google":
            self.llm = ChatGooglePalm(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model_name=model_name or "models/chat-bison-001",
                temperature=temperature
            )
        elif llm_type == "azure":
            self.llm = ChatOpenAI(
                model_name=model_name or "gpt-4",
                temperature=temperature,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            )
        elif llm_type == "ollama":
            self.llm = Ollama(
                model=model_name or "llama2",
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}") 