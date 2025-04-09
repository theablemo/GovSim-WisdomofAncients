from typing import Literal, Optional, TypeVar, Type
from langchain_openai import AzureOpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
import random
from pydantic import BaseModel
from .config import SimulationConfig

# Load environment variables
load_dotenv(override=True)

# Type variable for Pydantic models
T = TypeVar("T", bound=BaseModel)


class LLMConfig:
    """Centralized configuration for LLMs and embeddings"""

    def __init__(
        self,
        config: SimulationConfig,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.llm_type = config.llm_type
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.chunk_size = config.chunk_size
        self.seed = seed

        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-small",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            chunk_size=self.chunk_size,
        )

        # Initialize LLM based on type with provided seed
        if self.llm_type == "google":
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name or os.getenv("GOOGLE_MODEL_NAME"),
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                generation_config=(
                    {"seed": self.seed} if self.temperature > 0 and self.seed is not None else None
                ),
            )
        elif self.llm_type == "openai":
            self.llm = AzureChatOpenAI(
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_DEPLOYMENT_NAME"),
                model_name=self.model_name or os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL_NAME"),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                seed=self.seed if self.temperature > 0 and self.seed is not None else None,
            )
        elif self.llm_type == "ollama":
            self.llm = OllamaLLM(
                model=self.model_name or "gemma3",
                temperature=self.temperature,
                seed=self.seed if self.temperature > 0 and self.seed is not None else None,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def get_response_content(self, response):
        """Standardize response content across different LLM types"""
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)

    def extract_json_response(
        self, response, model_class: Type[T], default: Optional[T] = None
    ) -> T:
        """Extract and parse JSON from LLM response into a Pydantic model.

        Args:
            response: The raw LLM response
            model_class: Pydantic model class to parse the response into
            default: Default instance to return if parsing fails

        Returns:
            An instance of the specified Pydantic model
        """
        try:
            # Get standardized response content
            cleaned_content = self.get_response_content(response).strip()

            # Clean up any markdown formatting
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]  # Remove ```
            cleaned_content = cleaned_content.strip()

            # Parse the JSON response using the provided model class
            return model_class.model_validate_json(cleaned_content)

        except Exception as e:
            if self.config.verbose:
                print(f"Error parsing {model_class.__name__} from response: {e}")

            if default is not None:
                return default
            raise  # Re-raise the exception if no default is provided
