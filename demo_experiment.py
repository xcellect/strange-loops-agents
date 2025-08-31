import numpy as np
import matplotlib.pyplot as plt
from strange_loops_agent import StrangeLoopAgent, calculate_simple_phi
from visualization import (visualize_strange_loop, plot_phi_evolution,
                          create_animation, plot_belief_heatmap)
import os


def create_environment_agent():
    """
    Create an environment that responds to the agent's self-awareness.

    Returns:
        function: Environment function that takes agent state and returns observations
    """
    def environment_step(agent, step_count):
        """
        Environment that gives different observations based on agent's recursion depth.
        This creates a feedback loop that can trigger strange loop formation.
        """
        recursion_depth = agent.recursion_depth

        # Base observation (world state)
        world_obs = np.random.randint(3)  # Random world observation

        # Self observation influenced by recursion depth
        if recursion_depth > 2:
            # "Enlightened" observation when agent is self-aware
            self_obs = 2
        elif recursion_depth > 1:
            # "Aware" observation
            self_obs = 1
        else:
            # "Unaware" observation
            self_obs = 0

        # Add some noise to make it more realistic
        if np.random.random() < 0.1:  # 10% chance of noise
            self_obs = np.random.randint(3)

        return [world_obs, self_obs]

    return environment_step


def run_experiment(agent, environment_fn, steps=100, verbose=True):
    """
    Run the strange loop experiment.

    Args:
        agent: StrangeLoopAgent instance
        environment_fn: Function that returns observations
        steps: Number of steps to run
        verbose: Whether to print progress

    Returns:
        dict: Experiment results and metrics
    """
    print(f"Starting strange loop experiment for {steps} steps...")

    history = []
    phi_history = []
    action_history = []

    for step in range(steps):
        # Get observation from environment
        obs = environment_fn(agent, step)

        # Agent takes a step with strange loop logic
        action = agent.quine_step(obs)

        # Calculate consciousness measure
        phi = calculate_simple_phi(agent)

        # Record data
        history.append({
            'step': step,
            'recursion_depth': agent.recursion_depth,
            'beliefs': agent.qs.copy() if agent.qs else None,
            'meta_beliefs': agent.self_model_history[-1].get('meta_beliefs') if agent.self_model_history else None,
            'action': action,
            'obs': obs,
            'phi': phi
        })

        phi_history.append(phi)
        action_history.append(action)

        if verbose and step % 20 == 0:
            print(f"Step {step}: Recursion={agent.recursion_depth:.1f}, Phi={phi:.3f}")

    # Calculate final metrics
    metrics = agent.get_recursion_metrics()
    metrics.update({
        'total_steps': steps,
        'final_phi': phi_history[-1] if phi_history else 0,
        'avg_phi': np.mean(phi_history) if phi_history else 0,
        'max_phi': max(phi_history) if phi_history else 0,
        'phi_variance': np.var(phi_history) if phi_history else 0,
        'phi_increase': (phi_history[-1] - phi_history[0]) if len(phi_history) > 1 else 0
    })

    results = {
        'history': history,
        'phi_history': phi_history,
        'action_history': action_history,
        'metrics': metrics,
        'agent': agent
    }

    print("\nExperiment completed!")
    print(f"Final recursion depth: {metrics['max_recursion_depth']:.1f}")
    print(f"Average Φ: {metrics['avg_phi']:.3f}")
    print(f"Φ increase: {metrics['phi_increase']:.3f}")

    return results


def demo_strange_loop():
    """
    Main demo function showing strange loop formation.
    """
    print("=" * 60)
    print("STRANGE LOOPS ACTIVE INFERENCE DEMO")
    print("=" * 60)
    print()

    # Create agent and environment
    agent = StrangeLoopAgent()
    environment = create_environment_agent()

    # Run experiment
    results = run_experiment(agent, environment, steps=100, verbose=True)

    # Create visualizations
    print("\nGenerating visualizations...")

    # Create output directory
    output_dir = "/workspace/strange_loops_agent/results"
    os.makedirs(output_dir, exist_ok=True)

    # Main visualization
    fig1 = visualize_strange_loop(results['history'],
                                 save_path=f"{output_dir}/strange_loops_demo.png")
    plt.show()

    # Phi evolution
    fig2 = plot_phi_evolution(results['history'], results['phi_history'],
                             save_path=f"{output_dir}/phi_evolution.png")
    plt.show()

    # Belief heatmap
    fig3 = plot_belief_heatmap(results['history'],
                              save_path=f"{output_dir}/belief_heatmap.png")
    plt.show()

    # Print detailed results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    metrics = results['metrics']
    print(f"Max Recursion Depth: {metrics['max_recursion_depth']:.2f}")
    print(f"Average Recursion Depth: {metrics['avg_recursion_depth']:.2f}")
    print(f"Recursion Stability: {metrics['recursion_stability']:.2f}")
    print(f"Total Loops Detected: {metrics['total_loops_detected']}")
    print()
    print(f"Final Φ (Integration): {metrics['final_phi']:.3f}")
    print(f"Average Φ: {metrics['avg_phi']:.3f}")
    print(f"Max Φ: {metrics['max_phi']:.3f}")
    print(f"Φ Increase: {metrics['phi_increase']:+.3f}")
    print()
    print(f"Total Steps: {metrics['total_steps']}")

    # Analyze strange loop formation
    recursion_depths = [h['recursion_depth'] for h in results['history']]
    phi_values = results['phi_history']

    print("\n" + "=" * 60)
    print("STRANGE LOOP ANALYSIS")
    print("=" * 60)

    # Check for correlation between recursion and consciousness
    if len(recursion_depths) > 1 and len(phi_values) > 1:
        correlation = np.corrcoef(recursion_depths, phi_values)[0, 1]
        print(f"Correlation between recursion and Φ: {correlation:.3f}")

    # Check when strange loops first form
    loop_start = None
    for i, depth in enumerate(recursion_depths):
        if depth > 0.5:
            loop_start = i
            break

    if loop_start is not None:
        print(f"Strange loops first detected at step: {loop_start}")
        print(f"Initial Φ: {phi_values[0]:.3f}, Φ at loop start: {phi_values[loop_start]:.3f}")
    else:
        print("No significant strange loop formation detected")

    # Check for attractor behavior (stable high recursion)
    high_recursion_periods = sum(1 for d in recursion_depths if d > 2.0)
    if high_recursion_periods > 0:
        print(f"Periods of deep recursion: {high_recursion_periods} steps")
        print("Attractor behavior detected - agent maintains self-reference!")

    return results


def run_multiple_experiments(num_experiments=5, steps_per_experiment=50):
    """
    Run multiple experiments to test reproducibility.

    Args:
        num_experiments: Number of experiments to run
        steps_per_experiment: Steps per experiment
    """
    print(f"\nRunning {num_experiments} experiments for statistical analysis...")

    all_metrics = []

    for exp in range(num_experiments):
        print(f"\nExperiment {exp + 1}/{num_experiments}")
        agent = StrangeLoopAgent()
        environment = create_environment_agent()
        results = run_experiment(agent, environment, steps=steps_per_experiment, verbose=False)
        all_metrics.append(results['metrics'])

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    metric_names = ['max_recursion_depth', 'avg_recursion_depth', 'final_phi', 'avg_phi', 'max_phi']

    for metric in metric_names:
        values = [m[metric] for m in all_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric:25s}: {mean_val:.3f} ± {std_val:.3f}")

    # Check consistency
    phi_increases = [m['phi_increase'] for m in all_metrics]
    positive_increases = sum(1 for inc in phi_increases if inc > 0)
    print(f"\nExperiments with Φ increase: {positive_increases}/{num_experiments} ({100*positive_increases/num_experiments:.1f}%)")

    return all_metrics


if __name__ == "__main__":
    # Run main demo
    results = demo_strange_loop()

    # Run multiple experiments for statistics
    if input("\nRun multiple experiments for statistical analysis? (y/n): ").lower() == 'y':
        run_multiple_experiments()

    print("\nDemo completed! Check the 'results' directory for visualizations.")

