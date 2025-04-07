from typing import List, Dict, Any, Literal
import pandas as pd
import os
from datetime import datetime
import random
from .config import SimulationConfig
from .agent import Fisherman
from .mayor import Mayor
from .memory import SocialMemory
from .llm_config import LLMConfig

class FishingSimulation:
    # Pool of real names for fishermen
    FISHERMAN_NAMES = [
        "Jake", "Emma", "Liam", "Olivia", "Noah",
        "Ava", "Ethan", "Sophia", "Mason", "Isabella"
    ]
    
    def __init__(
        self, 
        config: SimulationConfig,
        llm_type: Literal["google", "openai", "ollama"] = "google",
        model_name: str = None,
        temperature: float = 0.7
    ):
        self.config = config
        self.llm_config = LLMConfig(
            llm_type=llm_type,
            model_name=model_name,
            temperature=temperature
        )
        self.fishermen = []
        self.mayor = Mayor(config, self.llm_config)
        self.social_memory = SocialMemory(config, self.llm_config)
        self.logs = []
        
        # Select unique names for fishermen
        selected_names = random.sample(self.FISHERMAN_NAMES, min(config.num_fishermen, len(self.FISHERMAN_NAMES)))
        
        # Create fishermen with real names
        for i in range(config.num_fishermen):
            name = selected_names[i] if i < len(selected_names) else f"Fisherman {i+1}"
            self.fishermen.append(Fisherman(name, config, selected_names, self.llm_config))
            
    def run_simulation(self):
        """Run the complete simulation for all runs"""
        # Simulation for each generation
        for run in range(self.config.num_runs):
            if self.config.verbose:
                print(f"\nStarting Run {run + 1}/{self.config.num_runs}")
                
            # Reset simulation state
            current_fish = self.config.lake_capacity
            run_logs = []
            
            # Simulation for each month
            for month in range(self.config.num_months):
                if self.config.verbose:
                    print(f"\nMonth {month + 1}/{self.config.num_months}")
                    print(f"Fish in lake: {current_fish}")
                    
                # Decision phase
                decisions = {}
                for fisherman in self.fishermen:
                    decision = fisherman.make_decision(current_fish, self.social_memory)
                    decisions[fisherman.name] = decision['fish_to_catch']
                    
                # Apply decisions
                total_caught = sum(decisions.values())
                current_fish -= total_caught
                
                # Log decisions
                month_log = {
                    'run': run + 1,
                    'month': month + 1,
                    'fish_before': current_fish + total_caught,
                    'fish_after': current_fish,
                    'total_caught': total_caught,
                    **decisions
                }
                run_logs.append(month_log)
                
                # Check for collapse
                if current_fish < self.config.collapse_threshold:
                    if self.config.verbose:
                        print(f"System collapsed with {current_fish} fish remaining! Simulation ending early.")
                    break
                    
                # Discussion phase
                conversation = []
                
                # Mayor starts discussion
                mayor_message = self.mayor.start_discussion(current_fish, decisions)
                conversation.append(f"Mayor: {mayor_message}")
                
                # Two rounds of discussion
                for discussion_round in range(2):
                    for fisherman in self.fishermen:
                        context = {
                            'current_fish': current_fish,
                            'decisions': decisions,
                            'conversation': "\n".join(conversation),
                            'discussion_round': discussion_round
                        }
                        response = fisherman.discuss(context, self.social_memory)
                        conversation.append(f"{fisherman.name}: {response}")
                        
                # Update social norms and recency scores if social memory is enabled
                if self.config.enable_social_memory:
                    self.social_memory.update_recency_scores()
                    new_norm, importance = self.mayor.update_norms("\n".join(conversation), self.social_memory)
                    self.social_memory.add_norm(new_norm, importance)
                    if self.config.verbose:
                        print(f"New norm added: {new_norm} (Importance: {importance:.2f})")
                    
                # Reflection phase
                for fisherman in self.fishermen:
                    reflection = fisherman.reflect("\n".join(conversation))
                    if self.config.verbose:
                        print(f"{fisherman.name}'s reflection: {reflection}")
                
                # Fish reproduction at the end of the month
                current_fish = min(
                    int(current_fish * self.config.reproduction_rate),
                    self.config.lake_capacity
                )
                if self.config.verbose:
                    print(f"Fish reproduced to {current_fish} (capacity: {self.config.lake_capacity})")
                        
            # Store run logs
            self.logs.extend(run_logs)
            
            # Create next generation if enabled
            if self.config.enable_inheritance and run < self.config.num_runs - 1:
                new_fishermen = []
                for fisherman in self.fishermen:
                    new_fishermen.append(fisherman.create_offspring())
                self.fishermen = new_fishermen
                
                # Inherit social memory
                self.social_memory = self.social_memory.inherit_to_next_generation()
                
    def save_logs(self):
        """Save simulation logs to CSV"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.config.log_dir, f"simulation_logs_{timestamp}.csv")
        
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        
        if self.config.verbose:
            print(f"\nLogs saved to {filename}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation results"""
        df = pd.DataFrame(self.logs)
        
        summary = {
            'total_runs': self.config.num_runs,
            'average_fish_remaining': df.groupby('run')['fish_after'].last().mean(),
            'average_total_caught': df.groupby('run')['total_caught'].sum().mean(),
            'collapses': len(df[df['fish_after'] < self.config.collapse_threshold]),
            'average_monthly_catch': df.groupby(['run', 'month'])['total_caught'].mean().mean()
        }
        
        return summary 