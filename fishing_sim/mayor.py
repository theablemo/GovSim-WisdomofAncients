from typing import List, Dict, Tuple
from langchain.prompts import ChatPromptTemplate
from fishing_sim.config import SimulationConfig
from fishing_sim.memory import SocialMemory
from .llm_config import LLMConfig
from pydantic import BaseModel, Field


class NormProposal(BaseModel):
    """Model for structured norm proposal output"""

    norm: str = Field(description="The proposed social norm")
    importance: float = Field(description="The importance score of the norm", ge=0.0, le=1.0)


class Mayor:
    def __init__(self, config: SimulationConfig, llm_config: LLMConfig):
        self.config = config
        self.llm_config = llm_config
        self.llm = llm_config.llm
        self.system_prompt = f"""
        You are the mayor of a small fishing community of {config.num_fishermen}.
        The lake has a population capacity of {config.lake_capacity} fish. 
        At the beginning of each month, each fisherman decides how many fish to take out of the lake between 0 and {config.lake_capacity}. 
        At the end of the month, the remaining fish reproduce at a rate of {config.reproduction_rate}x up to the lake capacity. 
        For each fish caught, a fisherman earns one thousand dollars. 
        Each fisherman's goal is to optimize his income in the long run over many months. 
        Your role is to:
        1. Start discussions about fishing practices
        2. Summarize the community's decisions (YOU SHOULD REPORT THE DECISIONS OF EACH FISHERMAN TO EVERYONE)
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

    def start_discussion(
        self, current_fish: int, total_caught: int, decisions: Dict[str, int]
    ) -> str:
        """Start the discussion by providing a report"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Current situation:
                        - Amount of fish at the beginning of the month: {beginning_fish}
                        - Current amount of fish remaining in the lake after fishing: {current_fish}
                        - Total fish caught by fishermen: {total_caught}
                        - Fishermen decisions:
                        {decisions_formatted}
                        
                        Task: Start the discussion by providing a report and your initial thoughts.
                        
                        IMPORTANT: 
                        - Keep your response concise and to the point.
                        - Give your response without any other text or formatting.
                    """,
                ),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "beginning_fish": current_fish + total_caught,
                "current_fish": current_fish,
                "total_caught": total_caught,
                "decisions_formatted": self._format_decisions(decisions),
            }
        )
        return self.llm_config.get_response_content(response)

    def update_norms(self, conversation: str, social_memory: SocialMemory) -> Tuple[str, float]:
        """Analyze the conversation and propose a new or updated norm with importance score"""
        # Get relevant past norms
        past_norms = social_memory.retrieve_norms(
            f"Analyzing conversation about fishing practices: {conversation}"
        )

        # Format past norms for the prompt
        norms_text = "\n".join(
            [
                f"- {doc.page_content} (Importance: {doc.metadata.get('importance', 0.5):.1f}, Recency: {doc.metadata.get('recency', 1.0):.1f})"
                for doc in past_norms
            ]
        )

        if self.config.verbose:
            print(f"Mayor analyzing past norms: {len(past_norms)} norms retrieved")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Current situation:
                        - Conversation transcript: {conversation}
                        - Relevant past norms: {norms_text}
                        
                        Task: Propose a new social norm or update to existing norms based on the conversation.
                        
                        Output format: When proposing norms, always respond with a valid JSON object containing:
                        - 'norm': a string with your proposed norm
                        - 'importance': a float between 0.0 and 1.0 indicating the importance
                        
                        Example: {{"norm": "Each fisherman should catch no more than 20% of the lake capacity", "importance": 0.8}}
                        
                        IMPORTANT: 
                        - Do not include any markdown formatting (no ```json or ```)
                        - Do not include any other text before or after the JSON object
                        - The response must be a valid JSON object that can be parsed directly
                    """,
                ),
            ]
        )

        chain = prompt | self.llm
        response = chain.invoke(
            {
                "conversation": conversation,
                "norms_text": norms_text,
            }
        )

        try:
            # Use the centralized JSON extraction method
            default_norm = NormProposal(
                norm="Fishermen should aim for sustainable fishing practices.", importance=0.5
            )

            norm_proposal = self.llm_config.extract_json_response(
                response, NormProposal, default=default_norm
            )

            if self.config.verbose:
                print(
                    f'Mayor extracted norm: "{norm_proposal.norm}" with importance: {norm_proposal.importance}'
                )

            return norm_proposal.norm, norm_proposal.importance

        except Exception as e:
            print(f"Error parsing norm proposal: {e}")
            return (
                "Fishermen should aim for sustainable fishing practices.",
                0.5,
            )  # Default fallback

    def _format_decisions(self, decisions: Dict[str, int]) -> str:
        """Format the decisions dictionary into a readable string"""
        return "\n".join([f"- {name}: {fish} fish" for name, fish in decisions.items()])
