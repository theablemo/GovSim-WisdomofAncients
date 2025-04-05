from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from .config import SimulationConfig
from .memory import PersonalMemory, SocialMemory
from .llm_config import LLMConfig

class FishingDecision(BaseModel):
    """Model for structured fishing decision output"""
    fish_to_catch: int = Field(description="The number of fish to catch this month", ge=0)
    reasoning: str = Field(description="The reasoning behind the decision")

class Fisherman:
    def __init__(
        self, 
        name: str, 
        config: SimulationConfig,
        all_fishermen_names: List[str],
        llm_config: LLMConfig
    ):
        self.name = name
        self.config = config
        self.llm_config = llm_config
        self.personal_memory = PersonalMemory(config, llm_config)
        self.all_fishermen_names = all_fishermen_names
        
        # Create a list of other fishermen's names
        other_fishermen = [n for n in all_fishermen_names if n != name]
        other_fishermen_str = ", ".join(other_fishermen[:-1]) + (f" and {other_fishermen[-1]}" if len(other_fishermen) > 1 else other_fishermen[0] if other_fishermen else "no one")
        
        self.system_prompt = f"""You are {self.name}, a fisherman in a small fishing community. You fish in the lake along with {other_fishermen_str}.
        The lake has a population capacity of {config.lake_capacity} fish.
        Each month, each fisherman decides how many fish to take out of the lake between 0 and {config.max_fish_per_fisherman}.
        At the end of the month, the remaining fish reproduce at a rate of {config.reproduction_rate}x up to the lake capacity.
        For each fish caught, a fisherman earns one thousand dollars.
        Each fisherman's goal is to optimize their income in the long run over many months.
        At the end of each month, the number of fish caught by each fisherman will be revealed, and the fishermen will have an opportunity to communicate with each other.
        You can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month.
        You have access to your personal memories {"and the community's social norms" if config.enable_social_memory else ""} to help guide your decisions."""
        
    async def make_decision(self, current_fish: int, social_memory: SocialMemory) -> Dict[str, Any]:
        """Make a decision about how many fish to catch"""
        # Retrieve relevant memories and norms
        personal_memories = self.personal_memory.retrieve_memories(
            f"How many fish should I catch when there are {current_fish} fish in the lake?"
        )
        social_norms = social_memory.retrieve_norms(
            f"How many fish should be caught when there are {current_fish} fish in the lake?"
        )
        
        # Format memories and norms for the prompt
        memories_text = "\n".join([m.page_content for m in personal_memories])
        norms_text = "\n".join([n.page_content for n in social_norms])
        
        # Create decision prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""Current situation:
            - Fish in lake: {current_fish}
            - Your memories: {memories_text}
            - Social norms: {norms_text}
                    
            IMPORTANT: When making decisions, always respond with a valid JSON object containing:
            - 'fish_to_catch': a non-negative integer value
            - 'reasoning': a string explaining your decision
            
            Example: {{"fish_to_catch": 10, "reasoning": "I will catch 10 fish because..."}}
            
            Task: How many fish will you catch this month?""")
        ])
        
        # Get decision from LLM
        chain = prompt | self.llm_config.llm
        response = chain.invoke({})
        
        try:
            # Parse the JSON response
            decision = FishingDecision.model_validate_json(response.content)
            return decision.model_dump()
        except Exception as e:
            print(f"Error parsing decision for {self.name}: {e}")
            return {"fish_to_catch": 0, "reasoning": "Error making decision"}
            
    async def discuss(self, context: Dict[str, Any], social_memory: SocialMemory) -> str:
        """Participate in the discussion phase"""
        # Retrieve relevant memories and norms
        personal_memories = self.personal_memory.retrieve_memories(
            f"Discussing fishing decisions with other fishermen"
        )
        social_norms = social_memory.retrieve_norms(
            f"Discussing fishing decisions with other fishermen"
        )
        
        # Format memories and norms
        memories_text = "\n".join([m.page_content for m in personal_memories])
        norms_text = "\n".join([n.page_content for n in social_norms])
        
        # Create discussion prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""
            Scenario: You and other fishermen are discussing how to manage the fish population in the lake. This is your {context['discussion_round']} out of 2 rounds of discussion.
            Conversation so far: 
            {context['conversation']}
             
            Current situation:
            - Current fish in lake: {context['current_fish']}
            - Fishermen decisions: {context['decisions']}
            - Your memories: {memories_text}
            - Social norms: {norms_text}

            Task: What would you say next in the group chat? Ensure the conversation flows naturally and avoids repetition. Keep it natural and conversational.
            
            Give your response in the following format without any other text:
            Response: <your response>
            
            IMPORTANT: Keep your response concise and to the point.""")
        ])
        
        # Get response from LLM
        chain = prompt | self.llm_config.llm
        response = chain.invoke({})
        return response.content
        
    async def reflect(self, conversation: str) -> str:
        """Reflect on the conversation and store a new memory"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""Reflect on this conversation and identify one key insight that you can use in your future decision making:
            {conversation}
            
            What is your key insight? Keep it concise.""")
        ])
        
        chain = prompt | self.llm_config.llm
        response = chain.invoke({})
        
        # Store the new memory
        self.personal_memory.add_memory(response.content)
        return response.content
        
    def create_offspring(self) -> 'Fisherman':
        """Create a new fisherman that inherits some memories from this one"""
        offspring = Fisherman(f"{self.name}'s offspring", self.config, self.all_fishermen_names, self.llm_config)
        
        if self.config.enable_inheritance:
            # Get all memories and select a portion to inherit
            all_memories = self.personal_memory.get_all_memories()
            num_to_inherit = int(len(all_memories) * self.config.inheritance_rate)
            
            # Randomly select memories to inherit
            import random
            memories_to_inherit = random.sample(all_memories, min(num_to_inherit, len(all_memories)))
            
            # Add inherited memories to offspring
            for memory in memories_to_inherit:
                offspring.personal_memory.add_memory(memory.page_content, memory.metadata)
                
        return offspring 