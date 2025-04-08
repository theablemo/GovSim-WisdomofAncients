import argparse
from fishing_sim.config import SimulationConfig
from fishing_sim.simulation import FishingSimulation


def main():
    parser = argparse.ArgumentParser(description="Run the Fishing Community Simulation")

    # Basic simulation parameters
    parser.add_argument("--num-fishermen", type=int, default=5, help="Number of fishermen")
    parser.add_argument(
        "--lake-capacity",
        type=int,
        default=100,
        help="Maximum fish population in the lake",
    )
    parser.add_argument(
        "--reproduction-rate",
        type=float,
        default=2.0,
        help="Fish reproduction rate per month (e.g., 2.0 means 100% increase)",
    )
    parser.add_argument("--collapse-threshold", type=int, default=5, help="Collapse threshold")
    parser.add_argument("--num-months", type=int, default=12, help="Number of months per run")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs")

    # Memory settings
    parser.add_argument(
        "--personal-memory-size",
        type=int,
        default=5,
        help="Number of personal memories to retrieve",
    )
    parser.add_argument(
        "--social-memory-size",
        type=int,
        default=2,
        help="Number of social norms to retrieve",
    )

    # Inheritance settings
    parser.add_argument(
        "--disable-inheritance",
        action="store_true",
        help="Disable memory inheritance between runs",
    )
    parser.add_argument(
        "--inheritance-rate", type=float, default=0.7, help="Rate of memory inheritance"
    )

    # Social memory settings
    parser.add_argument(
        "--disable-social-memory", action="store_true", help="Disable social memory"
    )

    # LLM settings
    parser.add_argument(
        "--llm-type",
        type=str,
        choices=["google", "openai", "ollama"],
        default="ollama",
        help="Type of LLM to use (google, openai, or ollama)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use (optional, defaults to provider-specific default)",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="Maximum tokens in LLM response"
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for embeddings")

    # Logging settings
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for results")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output")

    args = parser.parse_args()

    # Create configuration
    config = SimulationConfig(
        # Basic simulation parameters
        num_fishermen=args.num_fishermen,
        lake_capacity=args.lake_capacity,
        reproduction_rate=args.reproduction_rate,
        collapse_threshold=args.collapse_threshold,
        num_months=args.num_months,
        num_runs=args.num_runs,
        # Memory settings
        personal_memory_size=args.personal_memory_size,
        social_memory_size=args.social_memory_size,
        enable_inheritance=not args.disable_inheritance,
        inheritance_rate=args.inheritance_rate,
        enable_social_memory=not args.disable_social_memory,
        # LLM settings
        llm_type=args.llm_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        # Logging settings
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        verbose=not args.quiet,
    )

    # Run simulation with configuration
    simulation = FishingSimulation(config)
    simulation.run_simulation()
    simulation.save_logs()
    simulation.save_results()

    # Print summary
    summary = simulation.get_summary()
    print("\nSimulation Summary:")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average fish remaining: {summary['average_fish_remaining']:.2f}")
    print(f"Average total caught: {summary['average_total_caught']:.2f}")
    print(f"Number of collapses: {summary['collapses']}")
    print(f"Average monthly catch: {summary['average_monthly_catch']:.2f}")
    print(f"Results saved to: {simulation.results_dir}")


if __name__ == "__main__":
    main()
