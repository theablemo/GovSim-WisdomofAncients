from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SimulationConfig(BaseModel):
    """Configuration for the fishing simulation"""

    # Basic simulation parameters
    num_fishermen: int = Field(default=5, description="Number of fishermen in the simulation")
    lake_capacity: int = Field(default=100, description="Maximum fish population in the lake")
    reproduction_rate: float = Field(
        default=2.0, description="Fish reproduction rate per month (e.g., 2.0 means 100% increase)"
    )
    num_months: int = Field(default=12, description="Number of months to run the simulation")
    collapse_threshold: int = Field(
        default=5, description="Minimum fish population before collapse"
    )
    num_runs: int = Field(default=5, description="Number of simulation runs")

    # Memory settings
    enable_social_memory: bool = Field(
        default=True, description="Whether to enable social memory and norms"
    )
    enable_inheritance: bool = Field(
        default=True, description="Whether to enable memory inheritance between generations"
    )
    personal_memory_size: int = Field(
        default=5, description="Number of memories to retrieve from personal memory"
    )
    social_memory_size: int = Field(
        default=2, description="Number of norms to retrieve from social memory"
    )
    inheritance_rate: float = Field(
        default=0.7, description="Percentage of memories inherited from parent"
    )

    # LLM settings
    llm_type: Literal["google", "openai", "ollama"] = Field(
        default="ollama", description="Type of LLM to use (google, openai, or ollama)"
    )
    model_name: Optional[str] = Field(
        default=None, description="Model name to use (defaults to provider-specific default)"
    )
    temperature: float = Field(
        default=0.3, description="Temperature for LLM responses (higher = more creative)"
    )
    max_tokens: int = Field(default=1000, description="Maximum number of tokens in LLM responses")
    chunk_size: int = Field(default=1000, description="Chunk size for embeddings")

    # Logging settings
    log_dir: str = Field(default="logs", description="Directory to store simulation logs")
    results_dir: str = Field(default="results", description="Directory to store simulation results")
    verbose: bool = Field(default=False, description="Whether to print detailed logs")

    @property
    def simulation_dir_name(self) -> str:
        """Generate a directory name for the simulation results based on configuration"""
        social_memory_setting = (
            "SocialMemoryEnabled" if self.enable_social_memory else "SocialMemoryDisabled"
        )
        inheritance_setting = (
            "InheritanceEnabled" if self.enable_inheritance else "InheritanceDisabled"
        )
        setting = f"{social_memory_setting}_{inheritance_setting}"
        model = self.model_name if self.model_name else self.llm_type
        return f"{model}_{setting}_{self.lake_capacity}_{self.num_fishermen}_{self.num_months}_{self.num_runs}"
