import numpy as np
import matplotlib.pyplot as plt
from simple_strange_loop_agent import SimpleStrangeLoopAgent, calculate_simple_phi
import os


def create_simple_environment():
    """
    Create a simple environment that responds to the agent's self-awareness.
    
    Returns:
        function: Environment function that takes agent state and returns observations
    """
    def environment_step(agent, step_count):
        """
        Environment that gives different observations based on agent's recursion depth.
        """
        recursion_depth = agent.recursion_depth
        
        # Base observation influenced by recursion depth
        if recursion_depth > 2:
            # "Enlightened" observation when agent is self-aware
            obs = 2
        elif recursion_depth > 1:
            # "Aware" observation
            obs = 1
        else:
            # "Unaware" observation
            obs = 0
            
        # Add some noise
        if np.random.random() < 0.2:  # 20% chance of noise
            obs = np.random.randint(agent.num_obs)
            
        return obs
    
    return environment_step


def run_simple_experiment(agent, environment_fn, steps=100, verbose=True):
    """
    Run the simple strange loop experiment.
    
    Args:
        agent: SimpleStrangeLoopAgent instance
        environment_fn: Function that returns observations
        steps: Number of steps to run
        verbose: Whether to print progress
        
    Returns:
        dict: Experiment results and metrics
    """
    print(f"Starting simple strange loop experiment for {steps} steps...")
    
    history = []
    phi_history = []
    
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
            'beliefs': agent.qs.copy(),
            'action': action,
            'obs': obs,
            'phi': phi
        })
        
        phi_history.append(phi)
        
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
        'metrics': metrics,
        'agent': agent
    }
    
    print("\\nExperiment completed!")
    print(f"Final recursion depth: {metrics['max_recursion_depth']:.1f}")
    print(f"Average Φ: {metrics['avg_phi']:.3f}")
    print(f"Φ increase: {metrics['phi_increase']:.3f}")
    
    return results


def simple_visualization(history, phi_history, save_path=None):
    """
    Create simple visualizations of the strange loop dynamics.
    
    Args:
        history: List of history dictionaries
        phi_history: List of phi values
        save_path: Optional path to save the figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Recursion depth over time
    recursion_depths = [h['recursion_depth'] for h in history]
    ax1.plot(range(len(recursion_depths)), recursion_depths, 'b-', linewidth=2, alpha=0.8)
    ax1.fill_between(range(len(recursion_depths)), recursion_depths, alpha=0.3)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Recursion Depth')
    ax1.set_title('Strange Loop Formation')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phi evolution
    ax2.plot(range(len(phi_history)), phi_history, 'g-', linewidth=2, alpha=0.8)
    ax2.fill_between(range(len(phi_history)), phi_history, alpha=0.3, color='green')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Φ (Integration)')
    ax2.set_title('Consciousness Measure Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Belief evolution
    beliefs = np.array([h['beliefs'] for h in history])
    for i in range(beliefs.shape[1]):
        ax3.plot(beliefs[:, i], label=f'State {i}', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Belief Probability')
    ax3.set_title('Belief Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Phi vs Recursion correlation
    ax4.scatter(recursion_depths, phi_history, alpha=0.6, c=range(len(phi_history)), cmap='viridis')
    ax4.set_xlabel('Recursion Depth')
    ax4.set_ylabel('Φ (Integration)')
    ax4.set_title('Φ vs Recursion Depth')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    if len(phi_history) > 1:
        z = np.polyfit(recursion_depths, phi_history, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(recursion_depths), p(sorted(recursion_depths)), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def demo_simple_strange_loop():
    """
    Main demo function showing simple strange loop formation.
    """
    print("=" * 60)
    print("SIMPLE STRANGE LOOPS ACTIVE INFERENCE DEMO")
    print("=" * 60)
    print()
    
    # Create agent and environment
    agent = SimpleStrangeLoopAgent(num_states=3, num_obs=3, num_controls=3)
    environment = create_simple_environment()
    
    print(f"Agent initialized with {agent.num_states} states, {agent.num_obs} observations")
    print(f"Initial Phi: {calculate_simple_phi(agent):.3f}")
    print()
    
    # Run experiment
    results = run_simple_experiment(agent, environment, steps=100, verbose=True)
    
    # Create visualizations
    print("\\nGenerating visualizations...")
    
    # Create output directory
    output_dir = "/workspace/strange_loops_agent/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Main visualization
    fig = simple_visualization(
        results['history'], 
        results['phi_history'],
        save_path=f"{output_dir}/simple_strange_loops_demo.png"
    )
    plt.show()
    
    # Analyze results
    metrics = results['metrics']
    recursion_depths = [h['recursion_depth'] for h in results['history']]
    phi_values = results['phi_history']
    
    print("\\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Max Recursion Depth: {metrics['max_recursion_depth']:.2f}")
    print(f"Average Recursion Depth: {metrics['avg_recursion_depth']:.2f}")
    print(f"Total Loops Detected: {metrics['total_loops_detected']}")
    print()
    print(f"Final Φ (Integration): {metrics['final_phi']:.3f}")
    print(f"Average Φ: {metrics['avg_phi']:.3f}")
    print(f"Max Φ: {metrics['max_phi']:.3f}")
    print(f"Φ Increase: {metrics['phi_increase']:+.3f}")
    print()
    
    # Check for correlation between recursion and consciousness
    if len(recursion_depths) > 1 and len(phi_values) > 1:
        correlation = np.corrcoef(recursion_depths, phi_values)[0, 1]
        print(f"Correlation between recursion and Φ: {correlation:.3f}")
        
        if correlation > 0.5:
            print("Strong positive correlation: Self-reference increases integration!")
        elif correlation > 0.3:
            print("Moderate correlation detected.")
        else:
            print("Weak correlation - other factors may be influencing consciousness.")
    
    # Check when strange loops first form
    loop_start = None
    for i, depth in enumerate(recursion_depths):
        if depth > 0.5:
            loop_start = i
            break
    
    if loop_start is not None:
        print(f"\\nStrange loops first detected at step: {loop_start}")
        print(f"Initial Φ: {phi_values[0]:.3f}, Φ at loop start: {phi_values[loop_start]:.3f}")
    else:
        print("\\nNo significant strange loop formation detected")
    
    return results


if __name__ == "__main__":
    # Run the demo
    results = demo_simple_strange_loop()
    print("\\nDemo completed! Check the 'results' directory for visualizations.")