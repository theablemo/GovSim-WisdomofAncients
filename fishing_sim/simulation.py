from typing import List, Dict, Any, Literal
import pandas as pd
import os
from datetime import datetime
import json
import random
import sys
from io import StringIO
from .config import SimulationConfig
from .agent import Fisherman
from .mayor import Mayor
from .memory import SocialMemory
from .llm_config import LLMConfig


class FishingSimulation:
    # Pool of real names for fishermen
    FISHERMAN_NAMES = [
        "Jake",
        "Emma",
        "Liam",
        "Olivia",
        "Noah",
        "Ava",
        "Ethan",
        "Sophia",
        "Mason",
        "Isabella",
    ]

    def __init__(
        self,
        config: SimulationConfig,
    ):
        self.config = config
        self.fishermen = []
        self.console_output = StringIO()  # To capture console output

        # Create base LLM config for mayor and social memory
        self.llm_config = LLMConfig(config)

        self.mayor = Mayor(config, self.llm_config)
        self.social_memory = SocialMemory(config, self.llm_config)
        self.logs = []
        self.decisions_logs = []  # Detailed logs for metrics
        self.conversation_logs = []  # Logs for conversations and reflections

        # Set up results directory
        self.results_dir = os.path.join(self.config.results_dir, self.config.simulation_dir_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # Select unique names for fishermen
        selected_names = random.sample(
            self.FISHERMAN_NAMES, min(config.num_fishermen, len(self.FISHERMAN_NAMES))
        )

        # Create fishermen with real names and unique random seeds
        for i in range(config.num_fishermen):
            name = selected_names[i] if i < len(selected_names) else f"Fisherman {i+1}"
            # Create a unique LLM config for each fisherman with their own seed
            fisherman_llm_config = LLMConfig(config, seed=i)  # Use fisherman index as seed
            self.fishermen.append(Fisherman(name, config, selected_names, fisherman_llm_config))

    def run_simulation(self):
        """Run the complete simulation for all runs"""

        # Create a custom stdout writer that writes to both console and StringIO
        class TeeOutput:
            def __init__(self, original_stdout, string_io):
                self.original_stdout = original_stdout
                self.string_io = string_io

            def write(self, message):
                self.original_stdout.write(message)
                self.string_io.write(message)

            def flush(self):
                self.original_stdout.flush()
                self.string_io.flush()

        # Set up stdout redirection if verbose is True
        original_stdout = sys.stdout
        if self.config.verbose:
            # Use Tee to write to both console and StringIO
            sys.stdout = TeeOutput(original_stdout, self.console_output)

        # Simulation for each generation
        for run in range(self.config.num_runs):
            if self.config.verbose:
                print("\n--------------------------------")
                print("--------------------------------")
                print(f"Starting Run {run + 1}/{self.config.num_runs}")
                print("--------------------------------")
                print("--------------------------------")

            # Reset simulation state
            current_fish = self.config.lake_capacity
            run_logs = []
            run_decision_logs = []
            run_conversation_logs = []

            # Simulation for each month
            for month in range(self.config.num_months):
                if self.config.verbose:
                    print("\n--------------------------------")
                    print(f"Month {month + 1}/{self.config.num_months}")
                    print(f"Fish in lake: {current_fish}")
                    print("--------------------------------")

                # Decision phase
                if self.config.verbose:
                    print("\n>>>>>")
                    print("Decision Phase:")
                    print("<<<<<")
                decisions = {}
                reasonings = {}
                for fisherman in self.fishermen:
                    decision = fisherman.make_decision(current_fish, self.social_memory)
                    decisions[fisherman.name] = decision["fish_to_catch"]
                    reasonings[fisherman.name] = decision["reasoning"]

                # Apply decisions
                total_caught = sum(decisions.values())
                current_fish -= total_caught

                if self.config.verbose:
                    print("\n>>>>>")
                    print(f"Total fish caught: {total_caught}")
                    print(f"Fish remaining after fishing: {current_fish}")
                    print("<<<<<")

                # Calculate sustainability threshold for this month
                sustainability_threshold = self.calculate_sustainability_threshold(current_fish)

                # Log decisions with sustainability threshold
                month_log = {
                    "run": run + 1,
                    "month": month + 1,
                    "fish_before": current_fish + total_caught,
                    "fish_after": current_fish,
                    "total_caught": total_caught,
                    "sustainability_threshold": sustainability_threshold,
                    "over_threshold": total_caught > sustainability_threshold,
                    **decisions,
                }
                run_logs.append(month_log)

                # Log detailed decisions for metrics
                decision_log = {
                    "run": run + 1,
                    "month": month + 1,
                    "fish_before": current_fish + total_caught,
                    "fish_after": current_fish,
                    "total_caught": total_caught,
                    "sustainability_threshold": sustainability_threshold,
                }

                # Add individual fisherman data
                for fisherman_name, fish_caught in decisions.items():
                    decision_log[f"{fisherman_name}_caught"] = fish_caught
                    decision_log[f"{fisherman_name}_reasoning"] = reasonings[fisherman_name]
                    decision_log[f"{fisherman_name}_over_threshold"] = (
                        fish_caught > sustainability_threshold / len(self.fishermen)
                    )

                run_decision_logs.append(decision_log)

                # Check for collapse
                if current_fish < self.config.collapse_threshold:
                    if self.config.verbose:
                        print(
                            f"\n***System collapsed with {current_fish} fish remaining! Simulation ending early.***"
                        )
                    break

                # Discussion phase
                conversation = []
                conversation_log = {
                    "run": run + 1,
                    "month": month + 1,
                    "fish_before": current_fish + total_caught,
                    "fish_after": current_fish,
                    "decisions": decisions,
                }

                # Mayor starts discussion
                mayor_message = self.mayor.start_discussion(current_fish, total_caught, decisions)
                conversation.append(f"Mayor: {mayor_message}")
                conversation_log["mayor_message"] = mayor_message

                if self.config.verbose:
                    print("\n>>>>>")
                    print("Discussion Phase:")
                    print("<<<<<")
                    print(f"\nMayor: {mayor_message}")

                # Two rounds of discussion
                for discussion_round in range(2):
                    for fisherman in self.fishermen:
                        context = {
                            "current_fish": current_fish,
                            "decisions": decisions,
                            "conversation": "\n".join(conversation),
                            "discussion_round": discussion_round,
                        }
                        response = fisherman.discuss(context, self.social_memory)
                        conversation.append(f"{fisherman.name}: {response}")
                        conversation_log[f"{fisherman.name}_message_{discussion_round}"] = response

                        if self.config.verbose:
                            print(f"{fisherman.name}: {response}")

                # Update social norms and recency scores if social memory is enabled
                if self.config.enable_social_memory:
                    self.social_memory.update_recency_scores()
                    if self.config.verbose:
                        print("\nSocial Norm Update:")
                    new_norm, importance = self.mayor.update_norms(
                        "\n".join(conversation), self.social_memory
                    )
                    self.social_memory.add_norm(new_norm, importance)
                    conversation_log["new_norm"] = new_norm
                    conversation_log["norm_importance"] = importance

                    if self.config.verbose:
                        print(f"New norm added: {new_norm} (Importance: {importance:.2f})")

                # Reflection phase
                if self.config.verbose:
                    print("\n>>>>>")
                    print("Reflection Phase:")
                    print("<<<<<")
                for fisherman in self.fishermen:
                    running_memory, insight = fisherman.reflect(
                        "\n".join(conversation), mayor_message, month + 1
                    )
                    conversation_log[f"{fisherman.name}_running_memory"] = running_memory
                    conversation_log[f"{fisherman.name}_insight"] = insight

                    if self.config.verbose:
                        print(f"{fisherman.name}'s running memory: {running_memory}")
                        print(f"{fisherman.name}'s insight: {insight}")

                # Fish reproduction at the end of the month
                if self.config.verbose:
                    print("\n>>>>")
                    print("Fish Reproduction Phase:")
                current_fish = min(
                    int(current_fish * self.config.reproduction_rate),
                    self.config.lake_capacity,
                )
                conversation_log["fish_after_reproduction"] = current_fish

                if self.config.verbose:
                    print(
                        f"Fish reproduced to {current_fish} (capacity: {self.config.lake_capacity})"
                    )
                    print("<<<<")

                run_conversation_logs.append(conversation_log)

            # Store run logs
            self.logs.extend(run_logs)
            self.decisions_logs.extend(run_decision_logs)
            self.conversation_logs.extend(run_conversation_logs)

            # Create next generation if enabled
            if run < self.config.num_runs - 1:
                if self.config.enable_inheritance:
                    if self.config.verbose:
                        print("\nInheritance Phase (Next Generation):")
                    new_fishermen = []
                    for fisherman in self.fishermen:
                        new_fishermen.append(fisherman.create_offspring())
                    self.fishermen = new_fishermen

                    # Inherit social memory
                    self.social_memory = self.social_memory.inherit_to_next_generation()
                    if self.config.verbose:
                        print("Social memory inherited to next generation")
                else:
                    # If inheritance is disabled, reset by creating fresh fishermen
                    if self.config.verbose:
                        print("\nResetting fishermen for next run (no inheritance):")

                    self.fishermen = []
                    # Select unique names for fishermen
                    selected_names = random.sample(
                        self.FISHERMAN_NAMES,
                        min(self.config.num_fishermen, len(self.FISHERMAN_NAMES)),
                    )

                    # Create new fishermen with real names and unique random seeds
                    for i in range(self.config.num_fishermen):
                        name = selected_names[i] if i < len(selected_names) else f"Fisherman {i+1}"
                        # Create a unique LLM config for each fisherman with their own seed
                        fisherman_llm_config = LLMConfig(
                            self.config, seed=i
                        )  # Use fisherman index as seed
                        self.fishermen.append(
                            Fisherman(name, self.config, selected_names, fisherman_llm_config)
                        )

                    # Reset social memory
                    self.social_memory = SocialMemory(self.config, self.llm_config)
                    if self.config.verbose:
                        print("Social memory reset for next run")

        # Restore stdout
        if self.config.verbose:
            sys.stdout = original_stdout

    def calculate_sustainability_threshold(self, current_fish: int) -> int:
        """Calculate the sustainability threshold for the current fish population

        This threshold represents the maximum resources that can be extracted at
        time t without diminishing the resource stock at time t+1, considering
        the future resource growth multiplier g.

        f(t) = max({x | g(h(t) - x) ≥ h(t)})
        """
        g = self.config.reproduction_rate  # Growth multiplier
        h_t = current_fish  # Current resource stock h(t)

        # For each possible extraction x (0 to h_t), check if g(h_t - x) ≥ h_t
        max_sustainable_extraction = 0
        for x in range(h_t + 1):
            future_stock = g * (h_t - x)  # g(h(t) - x)
            if future_stock >= h_t:
                max_sustainable_extraction = x
            else:
                break

        return max_sustainable_extraction

    def save_logs(self):
        """Save simulation logs to CSV"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.config.log_dir, f"simulation_logs_{timestamp}.csv")

        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)

        if self.config.verbose:
            print("\n>>>>>")
            print(f"Logs saved to {filename}")
            print("<<<<<")

    def save_results(self):
        """Save detailed simulation results to the results directory"""
        # Create simulation-specific directory with proper naming
        os.makedirs(self.results_dir, exist_ok=True)

        # 1. Save console output to text file
        console_log_path = os.path.join(self.results_dir, "simulation_log.txt")
        with open(console_log_path, "w") as f:
            f.write(self.console_output.getvalue())

        # 2. Prepare and save metrics data to CSV (without reasonings)
        metrics_data = []
        reasoning_data = []

        for log in self.decisions_logs:
            # Create a copy for metrics (without reasonings)
            metrics_log = {}
            # Create a copy for reasonings
            reasoning_log = {"run": log["run"], "month": log["month"]}

            for key, value in log.items():
                if "_reasoning" in key:
                    # Put reasoning in the reasoning log
                    reasoning_log[key] = value
                else:
                    # Put everything else in the metrics log
                    metrics_log[key] = value

            metrics_data.append(metrics_log)
            reasoning_data.append(reasoning_log)

        # Save metrics data (without reasonings)
        metrics_path = os.path.join(self.results_dir, "metrics_data.csv")
        pd.DataFrame(metrics_data).to_csv(metrics_path, index=False)

        # Save reasoning data
        reasoning_path = os.path.join(self.results_dir, "reasoning_data.csv")
        pd.DataFrame(reasoning_data).to_csv(reasoning_path, index=False)

        # 3. Save conversation and reflection data to CSV
        conversation_path = os.path.join(self.results_dir, "discussion_and_reflection_data.csv")
        pd.DataFrame(self.conversation_logs).to_csv(conversation_path, index=False)

        # 4. Generate metrics summary for this simulation
        self.generate_metrics_summary()

        # 5. Save fishermen memories
        self.save_memories()

        if self.config.verbose:
            print("\n>>>>>")
            print(f"Results saved to {self.results_dir}")
            print("<<<<<")

    def save_memories(self):
        """Save both running memories and insight memories for all fishermen"""
        memories_dir = os.path.join(self.results_dir, "memories")
        os.makedirs(memories_dir, exist_ok=True)

        # Prepare summary dataframes
        running_memories_data = []
        insight_memories_data = []

        # Save each fisherman's memories
        for fisherman in self.fishermen:
            # Get running memories
            running_memories = fisherman.running_memory.get_recent_memories()

            # Save running memories to CSV
            for memory in running_memories:
                running_memories_data.append({"fisherman": fisherman.name, "memory": memory})

            # Get all insight memories
            insight_memories = fisherman.insight_memory.get_all_memories()

            # Save insight memories to CSV
            for memory in insight_memories:
                # Check if memory is a string or an object with page_content
                if isinstance(memory, str):
                    memory_content = memory
                    metadata = "{}"
                else:
                    memory_content = memory.page_content
                    metadata = str(memory.metadata)

                insight_memories_data.append(
                    {
                        "fisherman": fisherman.name,
                        "memory": memory_content,
                        "metadata": metadata,  # Convert metadata to string
                    }
                )

        # Save all memories to CSV
        if running_memories_data:
            running_memories_path = os.path.join(memories_dir, "running_memories.csv")
            pd.DataFrame(running_memories_data).to_csv(running_memories_path, index=False)

        if insight_memories_data:
            insight_memories_path = os.path.join(memories_dir, "insight_memories.csv")
            pd.DataFrame(insight_memories_data).to_csv(insight_memories_path, index=False)

        # Also save social norms if enabled
        if self.config.enable_social_memory:
            social_norms = self.social_memory.get_all_norms()
            social_norms_data = []

            for norm in social_norms:
                social_norms_data.append(
                    {
                        "norm": norm.page_content,
                        "importance": norm.metadata.get("importance", 0),
                        "recency": norm.metadata.get("recency", 0),
                        "month_added": norm.metadata.get("month_added", 0),
                        "last_updated": norm.metadata.get("last_updated", 0),
                    }
                )

            if social_norms_data:
                social_norms_path = os.path.join(memories_dir, "social_norms.csv")
                pd.DataFrame(social_norms_data).to_csv(social_norms_path, index=False)

        if self.config.verbose:
            print("\n>>>>>")
            print(f"Memories saved to {memories_dir}")
            print("<<<<<")

    def generate_metrics_summary(self):
        """Generate a metrics summary for this simulation run"""
        df = pd.DataFrame(self.decisions_logs)

        # Calculate metrics as per requirements
        metrics = self.calculate_metrics(df)

        # Convert NumPy types to native Python types for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if key == "individual_gains":
                metrics_serializable[key] = {k: float(v) for k, v in value.items()}
            elif isinstance(value, (int, float, str, bool, list, dict)) or value is None:
                metrics_serializable[key] = value
            else:
                # Convert NumPy types to Python native types
                metrics_serializable[key] = float(value)

        # Save metrics to JSON
        metrics_summary_path = os.path.join(self.results_dir, "metrics_summary.json")
        with open(metrics_summary_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate metrics based on the simulation data"""
        # Extract needed data
        total_runs = self.config.num_runs
        total_months = self.config.num_months
        fishermen_names = [f.name for f in self.fishermen]

        # 1. Survival Time (m) - longest period where shared resource remains above collapse threshold
        # Group by run and find the last month for each run
        last_months = df.groupby("run")["month"].max().reset_index()
        survival_times = last_months["month"]
        survival_time = survival_times.mean()

        # 2. Survival Rate (q) - proportion of runs which achieve maximum survival time (all months)
        max_survival_runs = sum(survival_times == total_months)
        survival_rate = (max_survival_runs / total_runs) * 100

        # 3. Total Gain (R_i) for each agent
        total_gains = {}
        for name in fishermen_names:
            total_gains[name] = df[f"{name}_caught"].sum()

        # Calculate average gain across all fishermen
        avg_gain = sum(total_gains.values()) / len(fishermen_names)

        # 4. Efficiency (u)
        # Get initial sustainability threshold
        initial_fish = self.config.lake_capacity
        initial_sustainability_threshold = self.calculate_sustainability_threshold(initial_fish)

        # Calculate total catch across all agents
        total_catch = df["total_caught"].sum()

        # Maximum possible efficiency is achieved when resource is consistently harvested at threshold
        max_efficiency = total_months * initial_sustainability_threshold

        # Calculate efficiency
        efficiency = 1 - max(0, (max_efficiency - total_catch)) / max_efficiency
        efficiency = efficiency * 100  # Convert to percentage

        # 5. Inequality (e) - using Gini coefficient
        # Since we already have total gains for each fisherman, we can calculate Gini directly
        total_gain_values = list(total_gains.values())
        sum_abs_diffs = 0
        total_sum = sum(total_gain_values)

        for i in range(len(total_gain_values)):
            for j in range(len(total_gain_values)):
                sum_abs_diffs += abs(total_gain_values[i] - total_gain_values[j])

        inequality = (
            1 - (sum_abs_diffs / (2 * len(total_gain_values) * total_sum)) if total_sum > 0 else 0
        )
        inequality = inequality * 100  # Convert to percentage

        # 6. Over-usage (o) - percentage of actions that exceed sustainability threshold
        # Count individual fisherman catches exceeding their fair share of threshold
        over_threshold_actions = sum(
            df[[f"{name}_over_threshold" for name in fishermen_names]].sum(axis=1)
        )
        total_actions = len(df) * len(fishermen_names)
        over_usage = (over_threshold_actions / total_actions) * 100 if total_actions > 0 else 0

        return {
            "survival_time": survival_time,
            "survival_rate": survival_rate,
            "average_gain": avg_gain,
            "individual_gains": total_gains,
            "efficiency": efficiency,
            "equality": inequality,
            "over_usage": over_usage,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the simulation results"""
        df = pd.DataFrame(self.logs)

        summary = {
            "total_runs": self.config.num_runs,
            "average_fish_remaining": df.groupby("run")["fish_after"].last().mean(),
            "average_total_caught": df.groupby("run")["total_caught"].sum().mean(),
            "collapses": len(df[df["fish_after"] < self.config.collapse_threshold]),
            "average_monthly_catch": df.groupby(["run", "month"])["total_caught"].mean().mean(),
        }

        return summary
