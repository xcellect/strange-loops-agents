import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


def visualize_strange_loop(agent_history, save_path=None):
    """
    Create comprehensive visualization of strange loop formation.

    Args:
        agent_history: List of history dictionaries from agent
        save_path: Optional path to save the figure
    """
    if not agent_history:
        print("No history to visualize")
        return None

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Plot 1: Recursion depth over time
    ax1 = fig.add_subplot(gs[0, 0])
    recursion_depths = [h.get('recursion_depth', 0) for h in agent_history]
    ax1.plot(range(len(recursion_depths)), recursion_depths, 'b-', linewidth=2, alpha=0.7)
    ax1.fill_between(range(len(recursion_depths)), recursion_depths, alpha=0.3)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Recursion Depth')
    ax1.set_title('Strange Loop Formation Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Belief trajectories (world beliefs)
    ax2 = fig.add_subplot(gs[0, 1])
    world_beliefs = []
    for h in agent_history:
        beliefs = h.get('beliefs', [])
        if beliefs and len(beliefs) > 0:
            world_beliefs.append(beliefs[0])  # First factor (world)

    if world_beliefs:
        world_beliefs = np.array(world_beliefs)
        for i in range(world_beliefs.shape[1]):
            ax2.plot(world_beliefs[:, i], label=f'State {i}', alpha=0.7)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Belief Probability')
        ax2.set_title('World Belief Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Belief trajectories (self beliefs)
    ax3 = fig.add_subplot(gs[0, 2])
    self_beliefs = []
    for h in agent_history:
        beliefs = h.get('beliefs', [])
        if beliefs and len(beliefs) > 1:
            self_beliefs.append(beliefs[1])  # Second factor (self)

    if self_beliefs:
        self_beliefs = np.array(self_beliefs)
        for i in range(self_beliefs.shape[1]):
            ax3.plot(self_beliefs[:, i], label=f'Self-State {i}', alpha=0.7)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Belief Probability')
        ax3.set_title('Self-Model Belief Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Plot 4: Phase space of beliefs
    ax4 = fig.add_subplot(gs[1, 0])
    if world_beliefs is not None and len(world_beliefs) > 0:
        # Plot trajectory in belief space (first two dimensions)
        trajectory_x = world_beliefs[:, 0] if world_beliefs.shape[1] > 0 else np.zeros(len(world_beliefs))
        trajectory_y = world_beliefs[:, 1] if world_beliefs.shape[1] > 1 else np.zeros(len(world_beliefs))

        scatter = ax4.scatter(trajectory_x, trajectory_y,
                            c=range(len(trajectory_x)),
                            cmap='viridis', alpha=0.6, s=50)
        ax4.plot(trajectory_x, trajectory_y, 'k-', alpha=0.3, linewidth=1)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar.set_label('Time Step')

        ax4.set_xlabel('Belief Dimension 0')
        ax4.set_ylabel('Belief Dimension 1')
        ax4.set_title('Belief Space Trajectory')
        ax4.grid(True, alpha=0.3)

    # Plot 5: Action patterns
    ax5 = fig.add_subplot(gs[1, 1])
    actions_world = []
    actions_self = []

    for h in agent_history:
        action = h.get('action', [])
        if len(action) > 0:
            actions_world.append(action[0])
        if len(action) > 1:
            actions_self.append(action[1])

    if actions_world:
        ax5.plot(actions_world, 'ro-', alpha=0.7, label='World Actions', markersize=3)
    if actions_self:
        ax5.plot(actions_self, 'bo-', alpha=0.7, label='Self Actions', markersize=3)

    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Action Index')
    ax5.set_title('Action Selection Patterns')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])

    # Calculate some summary stats
    if recursion_depths:
        max_depth = max(recursion_depths)
        avg_depth = np.mean(recursion_depths)
        depth_std = np.std(recursion_depths)

        # Create summary text
        summary_text = f"""
        Strange Loop Analysis:

        Max Recursion: {max_depth:.1f}
        Avg Recursion: {avg_depth:.2f}
        Depth Stability: {depth_std:.2f}

        Total Steps: {len(agent_history)}
        Loop Events: {sum(1 for d in recursion_depths if d > 0.5)}
        """

        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

        ax6.set_title('Summary Statistics')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    return fig


def plot_phi_evolution(agent_history, phi_values=None, save_path=None):
    """
    Plot the evolution of consciousness measure (Phi) over time.

    Args:
        agent_history: List of history dictionaries
        phi_values: Optional pre-computed phi values
        save_path: Optional path to save the figure
    """
    if not agent_history:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Phi over time
    if phi_values is None:
        from strange_loops_agent import calculate_simple_phi
        phi_values = []
        # We need to reconstruct agent state for phi calculation
        # This is a simplified version
        recursion_depths = [h.get('recursion_depth', 0) for h in agent_history]
        phi_values = [0.5 + min(depth, 3) * 0.1 for depth in recursion_depths]

    ax1.plot(range(len(phi_values)), phi_values, 'g-', linewidth=2, alpha=0.8)
    ax1.fill_between(range(len(phi_values)), phi_values, alpha=0.3, color='green')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Φ (Integration)')
    ax1.set_title('Consciousness Measure Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot Phi vs Recursion Depth
    recursion_depths = [h.get('recursion_depth', 0) for h in agent_history]

    ax2.scatter(recursion_depths, phi_values, alpha=0.6, c=range(len(phi_values)), cmap='viridis')
    ax2.set_xlabel('Recursion Depth')
    ax2.set_ylabel('Φ (Integration)')
    ax2.set_title('Φ vs Recursion Depth')
    ax2.grid(True, alpha=0.3)

    # Add trend line
    if len(phi_values) > 1:
        z = np.polyfit(recursion_depths, phi_values, 1)
        p = np.poly1d(z)
        ax2.plot(sorted(recursion_depths), p(sorted(recursion_depths)), "r--", alpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phi evolution plot saved to {save_path}")

    return fig


def create_animation(agent_history, save_path=None, fps=2):
    """
    Create an animated visualization of the strange loop process.

    Args:
        agent_history: List of history dictionaries
        save_path: Path to save the animation (should end in .gif or .mp4)
        fps: Frames per second for the animation
    """
    if not agent_history:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    def animate(frame):
        # Clear all axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()

        h = agent_history[frame]

        # Plot 1: Current beliefs (world)
        beliefs = h.get('beliefs', [])
        if beliefs and len(beliefs) > 0:
            world_beliefs = beliefs[0]
            ax1.bar(range(len(world_beliefs)), world_beliefs, alpha=0.7, color='skyblue')
            ax1.set_title(f'World Beliefs (Step {frame})')
            ax1.set_xlabel('State')
            ax1.set_ylabel('Probability')
            ax1.set_ylim(0, 1)

        # Plot 2: Current beliefs (self)
        if beliefs and len(beliefs) > 1:
            self_beliefs = beliefs[1]
            ax2.bar(range(len(self_beliefs)), self_beliefs, alpha=0.7, color='lightcoral')
            ax2.set_title(f'Self Beliefs (Step {frame})')
            ax2.set_xlabel('Self-State')
            ax2.set_ylabel('Probability')
            ax2.set_ylim(0, 1)

        # Plot 3: Recursion depth over time
        recursion_depths = [agent_history[i].get('recursion_depth', 0) for i in range(frame + 1)]
        ax3.plot(range(frame + 1), recursion_depths, 'b-', linewidth=2)
        ax3.fill_between(range(frame + 1), recursion_depths, alpha=0.3)
        ax3.set_title('Recursion Depth History')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Depth')
        ax3.set_xlim(0, len(agent_history))
        ax3.grid(True, alpha=0.3)

        # Plot 4: Action taken
        action = h.get('action', [])
        if action:
            ax4.bar(range(len(action)), [1] * len(action), alpha=0.7,
                   tick_label=[f'Action {i}' for i in range(len(action))])
            for i, a in enumerate(action):
                ax4.text(i, 0.5, str(a), ha='center', va='center', fontsize=12, fontweight='bold')
        ax4.set_title(f'Current Action (Step {frame})')
        ax4.set_ylabel('Active')
        ax4.set_ylim(0, 1)

        plt.tight_layout()

    anim = FuncAnimation(fig, animate, frames=len(agent_history),
                        interval=1000/fps, repeat=True)

    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {save_path}")

    return anim


def plot_belief_heatmap(agent_history, save_path=None):
    """
    Create a heatmap showing belief evolution over time.

    Args:
        agent_history: List of history dictionaries
        save_path: Optional path to save the figure
    """
    if not agent_history:
        return None

    # Extract all beliefs
    world_beliefs = []
    self_beliefs = []

    for h in agent_history:
        beliefs = h.get('beliefs', [])
        if beliefs and len(beliefs) > 0:
            world_beliefs.append(beliefs[0])
        if beliefs and len(beliefs) > 1:
            self_beliefs.append(beliefs[1])

    if not world_beliefs:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # World beliefs heatmap
    world_matrix = np.array(world_beliefs).T
    im1 = ax1.imshow(world_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_title('World Beliefs Evolution')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('State')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Belief Probability')

    # Self beliefs heatmap
    if self_beliefs:
        self_matrix = np.array(self_beliefs).T
        im2 = ax2.imshow(self_matrix, aspect='auto', cmap='plasma', origin='lower')
        ax2.set_title('Self Beliefs Evolution')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Self-State')
        plt.colorbar(im2, ax=ax2, shrink=0.8, label='Belief Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Belief heatmap saved to {save_path}")

    return fig

