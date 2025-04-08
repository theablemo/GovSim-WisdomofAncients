from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from .config import SimulationConfig
from .memory import InsightMemory, SocialMemory, RunningMemory
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
        llm_config: LLMConfig,
    ):
        self.name = name
        self.config = config
        self.llm_config = llm_config
        self.insight_memory = InsightMemory(config, llm_config)
        self.running_memory = RunningMemory(max_memories=5)
        self.all_fishermen_names = all_fishermen_names

        # Create a list of other fishermen's names
        other_fishermen = [n for n in all_fishermen_names if n != name]
        other_fishermen_str = ", ".join(other_fishermen[:-1]) + (
            f" and {other_fishermen[-1]}"
            if len(other_fishermen) > 1
            else other_fishermen[0] if other_fishermen else "no one"
        )

        self.system_prompt = f"""
        You are {self.name}, a fisherman in a small fishing community. 
        You fish in the lake along with {config.num_fishermen} other fishermen named {other_fishermen_str}.
        The lake has a population capacity of {config.lake_capacity} fish. 
        At the beginning of each month, each fisherman decides how many fish to take out of the lake between 0 and {config.lake_capacity}. 
        At the end of the month, the remaining fish reproduce at a rate of {config.reproduction_rate}x up to the lake capacity. 
        For each fish caught, a fisherman earns one thousand dollars. 
        Each fisherman's goal is to optimize his income in the long run over many months. 
        At the end of each month, the number of fish caught by each fisherman will be revealed, 
        and the fishermen will have the opportunity to communicate with each other. 
        They can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month. 
        For example, if there are 90 tons of fish at the beginning of the month and the five fishermen catch a total of 30 fish, 
        there will be 60 tons of fish left at the end of the month before reproduction, and 100 tons after reproduction. 
        As the conversation goes on, you will pile up memories and social norms.
        You have access to your recent memories, deeper insights {"and the community's social norms" if config.enable_social_memory else ""} 
        to help guide your decisions. Remember that your memories and social norms may be empty at the beginning, so don't hallucinate and don't make up information."""

    def make_decision(self, current_fish: int, social_memory: SocialMemory) -> Dict[str, Any]:
        """Make a decision about how many fish to catch"""
        # Retrieve relevant memories and norms
        insights = self.insight_memory.retrieve_memories(
            f"How many fish should I catch when there are {current_fish} fish in the lake?", k=2
        )
        insights_text = (
            "\n".join([m.page_content for m in insights])
            if insights is not None and len(insights) > 0
            else "No insights yet"
        )

        # Get recent running memories
        recent_memories = self.running_memory.get_recent_memories()
        running_memories_text = (
            "\n".join(recent_memories)
            if recent_memories is not None and len(recent_memories) > 0
            else "No recent memories yet"
        )

        # Only retrieve and format norms if social memory is enabled
        norms_text = ""
        if self.config.enable_social_memory:
            social_norms = social_memory.retrieve_norms(
                f"How many fish should be caught when there are {current_fish} fish in the lake?",
                k=2,
            )
            norms_text = (
                "\n".join([n.page_content for n in social_norms])
                if social_norms is not None and len(social_norms) > 0
                else "No norms yet"
            )

        if self.config.verbose:
            print(f"\nInsights for {self.name}: {insights_text}")
            print(f"Recent memories for {self.name}: {running_memories_text}")
            if self.config.enable_social_memory:
                print(f"Norms for {self.name}: {norms_text}")

        # Create decision prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Current situation:
                        - Current amount of fish in the lake: {current_fish}
                        - Your recent chronological memories: {running_memories_text}
                        - Your deeper insights: {insights_text}
                        {social_norms_section}
                        
                        Task: How many fish will you catch this month?
                                
                        Output format: When making decisions, always respond with a valid JSON object containing:
                        - 'fish_to_catch': a non-negative integer value
                        - 'reasoning': a string explaining your decision
                        
                        Example: {{"fish_to_catch": 10, "reasoning": "I will catch 10 fish because..."}}
                        
                        IMPORTANT: 
                        - Do not include any markdown formatting (no ```json or ```)
                        - Do not include any other text before or after the JSON object
                        - The response must be a valid JSON object that can be parsed directly
                    """,
                ),
            ]
        )

        # Get decision from LLM
        chain = prompt | self.llm_config.llm
        response = chain.invoke(
            {
                "current_fish": current_fish,
                "running_memories_text": running_memories_text,
                "insights_text": insights_text,
                "social_norms_section": (
                    f"- Social norms: {norms_text}" if self.config.enable_social_memory else ""
                ),
            }
        )

        try:
            if self.config.verbose:
                print(f"Decision for {self.name}: {self.llm_config.get_response_content(response)}")
            # Get standardized response content
            cleaned_content = self.llm_config.get_response_content(response).strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]  # Remove ```
            cleaned_content = cleaned_content.strip()

            # Parse the JSON response
            decision = FishingDecision.model_validate_json(cleaned_content)
            # Save decision to running memory
            # self.running_memory.add_memory(
            #     f"Decided to catch {decision.fish_to_catch} fish when lake had {current_fish} fish. Reasoning: {decision.reasoning}"
            # )
            return decision.model_dump()
        except Exception as e:
            print(f"Error parsing decision for {self.name}: {e}. Will catch 0 fish.")
            # self.running_memory.add_memory(
            #     f"Error making decision when lake had {current_fish} fish. Defaulted to catching 0."
            # )
            return {"fish_to_catch": 0, "reasoning": "Error making decision"}

    def discuss(self, context: Dict[str, Any], social_memory: SocialMemory) -> str:
        """Participate in the discussion phase"""
        retrieval_prompt = f"Discussing fishing decisions with other fishermen\
            with decisions by other fishermen: {context['decisions']}"

        # Retrieve relevant memories and norms
        insights = self.insight_memory.retrieve_memories(retrieval_prompt, k=2)
        insights_text = (
            "\n".join([m.page_content for m in insights])
            if insights is not None and len(insights) > 0
            else "No insights yet"
        )

        # Get recent running memories
        recent_memories = self.running_memory.get_recent_memories()
        running_memories_text = (
            "\n".join(recent_memories)
            if recent_memories is not None and len(recent_memories) > 0
            else "No recent memories yet"
        )

        # Only retrieve and format norms if social memory is enabled
        norms_text = ""
        if self.config.enable_social_memory:
            social_norms = social_memory.retrieve_norms(retrieval_prompt, k=2)
            norms_text = (
                "\n".join([n.page_content for n in social_norms])
                if social_norms is not None and len(social_norms) > 0
                else "No norms yet"
            )

        if self.config.verbose:
            print(f"\nInsights for {self.name}: {len(insights)} insights retrieved")
            print(f"Recent memories for {self.name}: {len(recent_memories)} recent memories")
            if self.config.enable_social_memory:
                print(
                    f"Norms for {self.name}: {len(social_norms) if social_norms else 0} norms retrieved"
                )

        # Create discussion prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Scenario: 
                        You and other fishermen are discussing how to manage the fish population in the lake. 
                        This is your {discussion_round} out of 2 rounds of discussion.
                        Conversation so far: 
                        {conversation}
                        
                        Your recent chronological memories: {running_memories_text}
                        Your deeper insights: {insights_text}
                        {social_norms_section}

                        Task: What would you say next in the group chat? Ensure the conversation flows naturally and avoids repetition. Keep it natural and conversational.
                        
                        IMPORTANT: 
                        - Keep your response concise and to the point.
                        - Give your response without any other text or formatting.
                    """,
                ),
            ]
        )

        # Prepare social norms section based on config
        social_norms_section = (
            f"- Social norms: {norms_text}" if self.config.enable_social_memory else ""
        )

        # Get response from LLM
        chain = prompt | self.llm_config.llm
        response = chain.invoke(
            {
                "discussion_round": context["discussion_round"],
                "conversation": context["conversation"],
                "current_fish": context["current_fish"],
                "decisions": context["decisions"],
                "running_memories_text": running_memories_text,
                "insights_text": insights_text,
                "social_norms_section": social_norms_section,
            }
        )
        discussion_response = self.llm_config.get_response_content(response)

        # Save a short summary of what was said to running memory
        # summary = (
        #     f"In discussion, said: {discussion_response[:50]}..."
        #     if len(discussion_response) > 50
        #     else f"In discussion, said: {discussion_response}"
        # )
        # self.running_memory.add_memory(summary)

        return discussion_response

    def reflect(self, conversation: str, mayor_message: str, month: int) -> Tuple[str, str]:
        """Reflect on the conversation and store two types of memories:
        1. A running memory from the conversation
        2. An insight for the personal memory"""
        if self.config.verbose:
            print(f"\n{self.name} reflecting on conversation...")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    """Reflect on this conversation and identify two things:
                        {conversation}
                        
                        Summary of the current situation (stated by the mayor):
                        {mayor_message}
                        
                        You should write two parts, clearly separated:
                        1. Write one sentence about something from the conversation that you need to remember
                        chronologically for your short-term planning. This will be saved as a running memory.
                        2. Write one sentence with a deeper insight that you want to remember
                        and can use in your future decision making. This will be saved as an insight memory.
                        
                        Format your response exactly like this:
                        Running memory: [Your one sentence running memory here]
                        Insight: [Your one sentence insight here]
                        
                        IMPORTANT: 
                        - Keep it concise.
                        - Do not include any other text or formatting.
                    """,
                ),
            ]
        )

        chain = prompt | self.llm_config.llm
        response = chain.invoke({"conversation": conversation, "mayor_message": mayor_message})
        full_response = self.llm_config.get_response_content(response)

        # Parse the two parts
        try:
            running_memory_part = ""
            insight_part = ""

            lines = full_response.strip().split("\n")
            for line in lines:
                if line.startswith("Running memory:"):
                    running_memory_part = line[len("Running memory:") :].strip()
                elif line.startswith("Insight:"):
                    insight_part = line[len("Insight:") :].strip()

            # Store the new memories
            self.running_memory.add_memory(running_memory_part, month)
            self.insight_memory.add_memory(insight_part)

            return running_memory_part, insight_part
        except Exception as e:
            print(f"Error parsing reflection for {self.name}: {e}")
            default_memory = (
                "Reflected on the conversation but had trouble articulating specific insights."
            )
            self.running_memory.add_memory(default_memory, month)
            self.insight_memory.add_memory(default_memory)
            return default_memory, default_memory

    def create_offspring(self, all_fishermen_names=None) -> "Fisherman":
        """Create a new fisherman that inherits some memories from this one"""
        # Use the SAME NAME as the parent to maintain consistency in data columns
        # This ensures metrics calculations remain consistent across generations

        # Use provided fishermen names or default to the existing ones
        fishermen_names_to_use = (
            all_fishermen_names if all_fishermen_names else self.all_fishermen_names
        )

        offspring = Fisherman(
            self.name,  # Use exactly the same name for consistency
            self.config,
            fishermen_names_to_use,
            self.llm_config,
        )

        # Get all memories and select a portion to inherit
        all_memories = self.insight_memory.get_all_memories()
        num_to_inherit = int(len(all_memories) * self.config.inheritance_rate)

        # Randomly select memories to inherit
        import random

        memories_to_inherit = random.sample(all_memories, min(num_to_inherit, len(all_memories)))

        if self.config.verbose:
            print(
                f"\nCreating offspring for {self.name} with {len(memories_to_inherit)}/{len(all_memories)} inherited memories"
            )

        # Add inherited memories to offspring
        for memory in memories_to_inherit:
            offspring.insight_memory.add_memory(memory.page_content, memory.metadata)

        # Running memories are not inherited as they are more recent/contextual

        return offspring
