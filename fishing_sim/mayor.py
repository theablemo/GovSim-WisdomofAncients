from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from fishing_sim.config import SimulationConfig
from fishing_sim.memory import SocialMemory
from .llm_config import LLMConfig

class Mayor:
    def __init__(self, config: SimulationConfig, llm_config: LLMConfig):
        self.config = config
        self.llm = llm_config.llm
        self.system_prompt = """You are the mayor of a small fishing community. Your role is to:
        1. Start discussions about fishing practices
        2. Summarize the community's decisions
        3. Help establish and update social norms based on community discussions
        4. Guide the community towards sustainable fishing practices
        
        When proposing new norms or updating existing ones, consider:
        - The current state of the lake and fishing practices
        - Past norms and their effectiveness
        - The community's recent discussions and decisions
        - The long-term sustainability of the fishing community
        
        IMPORTANT: When proposing norms, also assign an importance score between 0.0 and 1.0:
        - 0.0-0.3: Low importance (suggestions or guidelines)
        - 0.4-0.7: Medium importance (recommendations)
        - 0.8-1.0: High importance (rules or principles)"""
        
    def start_discussion(self, current_fish: int, decisions: Dict[str, int]) -> str:
        """Start the discussion by providing a report"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""Current situation:
            - Fish in lake: {current_fish}
            - Fishermen decisions:
            {self._format_decisions(decisions)}
            
            Start the discussion by providing a report and your initial thoughts. Keep it conversational.
            IMPORTANT: Keep your response concise and to the point.
            """)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        return response.content
        
    def update_norms(self, conversation: str, social_memory: SocialMemory) -> tuple[str, float]:
        """Analyze the conversation and propose a new or updated norm with importance score"""
        # Get relevant past norms
        past_norms = social_memory.retrieve_norms(
            f"Analyzing conversation about fishing practices: {conversation}"
        )
        
        # Format past norms for the prompt
        norms_text = "\n".join([
            f"- {doc.page_content} (Importance: {doc.metadata['importance']:.1f}, Recency: {doc.metadata['recency']:.1f})"
            for doc in past_norms
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", f"""Based on this conversation:
            {conversation}
            
            Relevant past norms:
            {norms_text}
            
            What new social norm or update to existing norms would you propose? 
            Also assign an importance score between 0.0 and 1.0.
            
            Respond in the following format:
            Norm: <your proposed norm>
            Importance: <importance score between 0.0 and 1.0>
            """)
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({})
        
        # Parse the response
        lines = response.content.split('\n')
        norm = ""
        importance = 0.5  # Default importance
        
        for line in lines:
            if line.startswith("Norm:"):
                norm = line[6:].strip()
            elif line.startswith("Importance:"):
                try:
                    importance = float(line[11:].strip())
                    importance = max(0.0, min(1.0, importance))  # Clamp between 0 and 1
                except ValueError:
                    pass
        
        return norm, importance
        
    def _format_decisions(self, decisions: Dict[str, int]) -> str:
        """Format the decisions dictionary into a readable string"""
        return "\n".join([f"- {name}: {fish} fish" for name, fish in decisions.items()]) 