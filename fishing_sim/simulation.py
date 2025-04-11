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
import pickle
import numpy as np
import time


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
        self.start_time = 0
        self.running_time = 0
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
        self.start_time = time.time()

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

        # Get the names of the fishermen for reuse in future runs
        selected_names = [fisherman.name for fisherman in self.fishermen]

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
                    print(
                        f"Month {month + 1}/{self.config.num_months} -- Run {run + 1}/{self.config.num_runs}"
                    )
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
                    # Correct over_threshold check: check against full threshold, not fair share
                    decision_log[f"{fisherman_name}_over_threshold"] = (
                        fish_caught > sustainability_threshold
                    )

                run_decision_logs.append(decision_log)

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

                # Check for collapse
                if current_fish < self.config.collapse_threshold:
                    if self.config.verbose:
                        print(
                            f"\n***System collapsed with {current_fish} fish remaining! Simulation ending early.***"
                        )
                    break

                # Fish reproduction at the end of the month (if not collapsed)
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

            # Create next generation
            if run < self.config.num_runs - 1:
                if self.config.enable_inheritance:
                    if self.config.verbose:
                        print("\nInheritance Phase (Next Generation):")
                    new_fishermen = []
                    for fisherman in self.fishermen:
                        # Pass the consistent selected_names to maintain name consistency
                        new_fishermen.append(fisherman.create_offspring(selected_names))
                    self.fishermen = new_fishermen

                    # Inherit social memory
                    # self.social_memory = self.social_memory.inherit_to_next_generation(
                    #     self.llm_config
                    # )
                    if self.config.verbose:
                        print("Social memory inherited to next generation")
                else:
                    # If inheritance is disabled, reset by creating fresh fishermen with same names
                    if self.config.verbose:
                        print("\nResetting fishermen for next run (no inheritance):")

                    # Store the current fishermen names to maintain consistency
                    current_names = [fisherman.name for fisherman in self.fishermen]

                    self.fishermen = []
                    # Create new fishermen with the SAME names as before for consistency
                    for i in range(self.config.num_fishermen):
                        name = current_names[i] if i < len(current_names) else f"Fisherman {i+1}"
                        # Create a unique LLM config for each fisherman with their own seed
                        fisherman_llm_config = LLMConfig(
                            self.config, seed=i
                        )  # Use fisherman index as seed
                        self.fishermen.append(
                            Fisherman(name, self.config, selected_names, fisherman_llm_config)
                        )

                    # Reset social memory - properly indented outside the fishermen creation loop
                    if self.config.enable_social_memory:
                        # Reset social memory
                        self.social_memory = SocialMemory(self.config, self.llm_config)
                        if self.config.verbose:
                            print("Social memory reset for next run")

        # Restore stdout
        if self.config.verbose:
            sys.stdout = original_stdout
        end_time = time.time()
        self.running_time = end_time - self.start_time

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
            # Save FAISS vectorstore using proper FAISS methods
            if fisherman.insight_memory.memory is not None:
                fisherman_faiss_dir = os.path.join(memories_dir, f"{fisherman.name}_faiss")
                os.makedirs(fisherman_faiss_dir, exist_ok=True)
                fisherman.insight_memory.save_to_disk(fisherman_faiss_dir)

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
        if self.config.enable_social_memory and self.social_memory.memory is not None:
            # Save FAISS vectorstore for social memory
            social_memory_faiss_dir = os.path.join(memories_dir, "social_memory_faiss")
            os.makedirs(social_memory_faiss_dir, exist_ok=True)
            self.social_memory.save_to_disk(social_memory_faiss_dir)

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

        # Add running time to metrics
        metrics_serializable["running_time_seconds"] = self.running_time

        # Save metrics to JSON
        metrics_summary_path = os.path.join(self.results_dir, "metrics_summary.json")
        with open(metrics_summary_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics by:
        1) Grouping by run
        2) Computing each run's metrics
        3) Averaging them across runs
        """

        run_ids = df["run"].unique()
        run_results = []

        # --- We identify all fisherman columns just once ---
        fishermen_columns = [col for col in df.columns if col.endswith("_caught")]
        fishermen_names = [col.replace("_caught", "") for col in fishermen_columns]

        for run_id in run_ids:
            run_df = df[df["run"] == run_id].copy()

            # -----------------------
            # (1) SURVIVAL TIME, SURVIVAL RATE
            # -----------------------
            # If you keep track of the resource level each month (e.g. in run_df["resource"]),
            # find the largest month index where the resource is still above the collapse threshold.
            # If you only store 'month' in the dataframe, you can do e.g.:
            last_month = run_df["month"].max()
            # But more accurately, you might do:
            #   valid_months = run_df[run_df["resource"] > self.config.collapse_threshold]["month"]
            #   survival_time_this_run = valid_months.max() if not valid_months.empty else 0
            # For simplicity, assume we just use the final month as "survived" if it never collapsed:
            survival_time_this_run = last_month
            # The run "fully survives" if it reaches num_months without collapse:
            survival_rate_this_run = (
                1.0 if survival_time_this_run == self.config.num_months else 0.0
            )

            # -----------------------
            # (2) TOTAL GAINS & AVERAGE GAIN
            # -----------------------
            # Sum up each fisherman’s catch for this run:
            fisherman_gains = {}
            for name in fishermen_names:
                fisherman_gains[name] = run_df[f"{name}_caught"].sum()

            # Mean across fishermen (if you want the “average gain per fisherman”):
            average_gain_this_run = sum(fisherman_gains.values()) / len(fishermen_names)

            # -----------------------
            # (3) EFFICIENCY
            # -----------------------
            # Compare total catch to the maximum "ideal" catch that keeps the resource stable.
            # For a simple measure, compare to "num_months * sustainability_threshold(initial_fish)".
            total_catch_this_run = run_df["total_caught"].sum()
            initial_sustainability_thresh = self.calculate_sustainability_threshold(
                self.config.lake_capacity
            )
            max_possible = self.config.num_months * initial_sustainability_thresh

            efficiency_this_run = 1 - max(0, (max_possible - total_catch_this_run)) / max_possible
            efficiency_this_run *= 100

            # -----------------------
            # (4) INEQUALITY (GINI)
            # -----------------------
            # Compute Gini for the final totals of each fisherman in this single run:
            total_gain_values = list(fisherman_gains.values())
            sum_all = sum(total_gain_values)
            if sum_all <= 0 or len(total_gain_values) < 2:
                inequality_this_run = 0
            else:
                sum_abs = 0
                for i in range(len(total_gain_values)):
                    for j in range(len(total_gain_values)):
                        sum_abs += abs(total_gain_values[i] - total_gain_values[j])
                # 1 - sum_of_diffs / (2*N*sum_all)
                inequality_this_run = 1 - (sum_abs / (2 * len(total_gain_values) * sum_all))
                inequality_this_run *= 100

            # -----------------------
            # (5) OVER‐USAGE
            # -----------------------
            # Over‐usage is the fraction of months for which total_caught > sustainability_threshold
            # at that time.  *Group total*, not each person’s catch individually.
            # We'll do one row per month in this run:
            # (assuming each month has exactly one row with the sum in 'total_caught')
            # If your DataFrame has multiple rows per month (one per fisherman), then
            # you'd first group by month, or look for 'if row['month'] changes ...'
            over_count = 0
            total_months_run = 0
            for m in run_df["month"].unique():
                month_rows = run_df[run_df["month"] == m]
                # we assume there's exactly one row that has total_caught for the group:
                row = month_rows.iloc[0]
                if row["total_caught"] > row["sustainability_threshold"]:
                    over_count += 1
                total_months_run += 1

            over_usage_this_run = (
                (over_count / total_months_run) * 100 if total_months_run > 0 else 0
            )

            # Save results
            run_results.append(
                {
                    "survival_time": survival_time_this_run,
                    "survival_rate": survival_rate_this_run,
                    "average_gain": average_gain_this_run,
                    "efficiency": efficiency_this_run,
                    "inequality": inequality_this_run,
                    "over_usage": over_usage_this_run,
                    "fisherman_gains": fisherman_gains,
                }
            )

        # Now average across runs
        survival_time = np.mean([r["survival_time"] for r in run_results])
        survival_rate = np.mean([r["survival_rate"] for r in run_results]) * 100
        average_gain = np.mean([r["average_gain"] for r in run_results])
        efficiency = np.mean([r["efficiency"] for r in run_results])
        inequality = np.mean([r["inequality"] for r in run_results])
        over_usage = np.mean([r["over_usage"] for r in run_results])

        # Optionally, aggregate fisherman gains across runs (e.g. mean per run):
        all_names = set()
        for rr in run_results:
            all_names.update(rr["fisherman_gains"].keys())
        final_gains = {}
        for name in all_names:
            final_gains[name] = np.mean([rr["fisherman_gains"][name] for rr in run_results])

        return {
            "survival_time": survival_time,
            "survival_rate": survival_rate,
            "average_gain": average_gain,
            "individual_gains": final_gains,
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
