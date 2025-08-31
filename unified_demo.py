import numpy as np
import matplotlib.pyplot as plt
from visualization import visualize_strange_loop, plot_phi_evolution
import os


def create_unified_environment():
    """
    Create an environment that works with both agent types.
    
    Returns:
        function: Environment function that takes agent state and returns observations
    """
    def environment_step(agent, step_count):
        """
        Environment that gives different observations based on agent's recursion depth.
        """
        recursion_depth = agent.recursion_depth
        
        # Determine observation format based on agent type
        if hasattr(agent, 'num_obs') and isinstance(agent.num_obs, list):
            # Full PyMDP agent - return list of observations
            world_obs = 2 if recursion_depth > 2 else (1 if recursion_depth > 1 else 0)
            self_obs = 2 if recursion_depth > 2 else (1 if recursion_depth > 1 else 0)
            
            # Add noise
            if np.random.random() < 0.1:
                world_obs = np.random.randint(3)
                self_obs = np.random.randint(3)
                
            return [world_obs, self_obs]
        else:
            # Simple agent - return single observation
            obs = 2 if recursion_depth > 2 else (1 if recursion_depth > 1 else 0)
            if np.random.random() < 0.2:
                obs = np.random.randint(3)
            return obs
    
    return environment_step


def run_unified_experiment(agent, environment_fn, steps=100, verbose=True):
    """
    Run experiment that works with both agent types.
    
    Args:
        agent: Either StrangeLoopAgent or SimpleStrangeLoopAgent
        environment_fn: Function that returns observations
        steps: Number of steps to run
        verbose: Whether to print progress
        
    Returns:
        dict: Experiment results and metrics
    """
    print(f"Starting experiment with {type(agent).__name__} for {steps} steps...")
    
    # Import the appropriate phi calculation
    if 'SimpleStrangeLoopAgent' in str(type(agent)):
        from simple_strange_loop_agent import calculate_simple_phi
    else:
        from strange_loops_agent import calculate_simple_phi
    
    history = []
    phi_history = []
    
    for step in range(steps):
        # Get observation from environment
        obs = environment_fn(agent, step)
        
        # Agent takes a step with strange loop logic
        action = agent.quine_step(obs)
        
        # Calculate consciousness measure
        phi = calculate_simple_phi(agent)
        
        # Record data (adapt format for different agent types)
        if hasattr(agent, 'qs') and isinstance(agent.qs, list):
            # Full PyMDP agent
            beliefs = [q.copy() for q in agent.qs]
        else:
            # Simple agent
            beliefs = agent.qs.copy() if hasattr(agent, 'qs') else None
        
        history.append({
            'step': step,
            'recursion_depth': agent.recursion_depth,
            'beliefs': beliefs,
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
    print(f"Agent type: {type(agent).__name__}")
    print(f"Final recursion depth: {metrics['max_recursion_depth']:.1f}")
    print(f"Average Φ: {metrics['avg_phi']:.3f}")
    print(f"Φ increase: {metrics['phi_increase']:.3f}")
    
    return results


def demo_both_versions():
    """
    Demo both the simple and full PyMDP versions.
    """
    print("=" * 80)
    print("UNIFIED STRANGE LOOPS DEMO - COMPARING BOTH IMPLEMENTATIONS")
    print("=" * 80)
    
    results_comparison = {}
    
    # Test Simple Version
    print("\\n" + "="*50)
    print("1. TESTING SIMPLE VERSION")
    print("="*50)
    
    try:
        from simple_strange_loop_agent import SimpleStrangeLoopAgent
        simple_agent = SimpleStrangeLoopAgent(num_states=3, num_obs=3, num_controls=3)
        simple_env = create_unified_environment()
        
        simple_results = run_unified_experiment(
            agent=simple_agent,
            environment_fn=simple_env,
            steps=50,
            verbose=True
        )
        results_comparison['simple'] = simple_results
        
    except Exception as e:
        print(f"Simple version failed: {e}")
        results_comparison['simple'] = None
    
    # Test Full PyMDP Version
    print("\\n" + "="*50)
    print("2. TESTING FULL PyMDP VERSION")
    print("="*50)
    
    try:
        from strange_loops_agent import StrangeLoopAgent
        full_agent = StrangeLoopAgent(num_states=[3, 3], num_obs=[3, 3], num_controls=[3, 3])
        full_env = create_unified_environment()
        
        full_results = run_unified_experiment(
            agent=full_agent,
            environment_fn=full_env,
            steps=50,
            verbose=True
        )
        results_comparison['full'] = full_results
        
    except Exception as e:
        print(f"Full PyMDP version failed: {e}")
        results_comparison['full'] = None
    
    # Compare Results
    print("\\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    if results_comparison['simple'] and results_comparison['full']:
        simple_metrics = results_comparison['simple']['metrics']
        full_metrics = results_comparison['full']['metrics']
        
        print("\\nPerformance Comparison:")
        print(f"{'Metric':<25} {'Simple':<15} {'Full PyMDP':<15} {'Difference':<15}")
        print("-" * 70)
        
        metrics_to_compare = ['max_recursion_depth', 'avg_phi', 'phi_increase']
        
        for metric in metrics_to_compare:
            simple_val = simple_metrics[metric]
            full_val = full_metrics[metric]
            diff = full_val - simple_val
            print(f"{metric:<25} {simple_val:<15.3f} {full_val:<15.3f} {diff:+.3f}")
        
        # Statistical comparison
        simple_phi = results_comparison['simple']['phi_history']
        full_phi = results_comparison['full']['phi_history']
        
        simple_corr = np.corrcoef(
            [h['recursion_depth'] for h in results_comparison['simple']['history']],
            simple_phi
        )[0, 1] if len(simple_phi) > 1 else 0
        
        full_corr = np.corrcoef(
            [h['recursion_depth'] for h in results_comparison['full']['history']],
            full_phi
        )[0, 1] if len(full_phi) > 1 else 0
        
        print(f"\\nCorrelation Analysis:")
        print(f"Simple version (Recursion ↔ Φ): {simple_corr:.3f}")
        print(f"Full PyMDP version (Recursion ↔ Φ): {full_corr:.3f}")
        
        # Determine winner
        print(f"\\n🏆 RESULTS:")
        if full_metrics['phi_increase'] > simple_metrics['phi_increase']:
            print("Full PyMDP version shows stronger consciousness emergence!")
        elif simple_metrics['phi_increase'] > full_metrics['phi_increase']:
            print("Simple version shows stronger consciousness emergence!")
        else:
            print("Both versions show similar consciousness emergence.")
            
        print(f"✅ Both implementations successfully demonstrate strange loops!")
        
    elif results_comparison['simple']:
        print("\\n✅ Simple version working, Full PyMDP version had issues")
        
    elif results_comparison['full']:
        print("\\n✅ Full PyMDP version working, Simple version had issues")
        
    else:
        print("\\n❌ Both versions had issues")
    
    # Create visualizations if we have results
    output_dir = "/workspace/strange_loops_agent/results"
    os.makedirs(output_dir, exist_ok=True)
    
    if results_comparison['simple']:
        print("\\n📊 Creating Simple Version Visualization...")
        try:
            fig = visualize_strange_loop(results_comparison['simple']['history'],
                                       save_path=f"{output_dir}/simple_version_results.png")
            plt.title("Simple Strange Loop Agent Results")
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    if results_comparison['full']:
        print("\\n📊 Creating Full PyMDP Version Visualization...")
        try:
            fig = visualize_strange_loop(results_comparison['full']['history'],
                                       save_path=f"{output_dir}/full_pymdp_results.png")
            plt.title("Full PyMDP Strange Loop Agent Results")  
            plt.show()
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    return results_comparison


if __name__ == "__main__":
    # Run the unified demo
    results = demo_both_versions()
    print("\\nUnified demo completed! Check the 'results' directory for visualizations.")