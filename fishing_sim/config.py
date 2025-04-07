from pydantic import BaseModel, Field
from typing import List, Optional

class SimulationConfig(BaseModel):
    """Configuration for the fishing simulation"""
    # Basic simulation parameters
    num_fishermen: int = Field(default=5, description="Number of fishermen in the simulation")
    lake_capacity: int = Field(default=100, description="Maximum fish population in the lake")
    reproduction_rate: float = Field(default=2.0, description="Fish reproduction rate per month (e.g., 2.0 means 100% increase)")
    num_months: int = Field(default=12, description="Number of months to run the simulation")
    collapse_threshold: int = Field(default=5, description="Minimum fish population before collapse")
    num_runs: int = Field(default=5, description="Number of simulation runs")
    
    # Memory settings
    enable_social_memory: bool = Field(default=True, description="Whether to enable social memory and norms")
    enable_inheritance: bool = Field(default=True, description="Whether to enable memory inheritance between generations")
    personal_memory_size: int = Field(default=5, description="Number of memories to retrieve from personal memory")
    social_memory_size: int = Field(default=2, description="Number of norms to retrieve from social memory")
    inheritance_rate: float = Field(default=0.7, description="Percentage of memories inherited from parent")
    
    # Logging settings
    log_dir: str = Field(default="logs", description="Directory to store simulation logs")
    verbose: bool = Field(default=False, description="Whether to print detailed logs") 