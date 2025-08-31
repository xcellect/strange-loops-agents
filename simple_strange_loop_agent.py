import numpy as np
from pymdp import utils, maths


class SimpleStrangeLoopAgent:
    def __init__(self, num_states=3, num_obs=3, num_controls=3):
        """
        Simplified Strange Loop Agent without full PyMDP integration.
        Focuses on the core strange loop mechanism.
        
        Args:
            num_states: Number of states
            num_obs: Number of observations  
            num_controls: Number of actions
        """
        self.num_states = num_states
        self.num_obs = num_obs
        self.num_controls = num_controls
        
        self.self_model_history = []
        self.recursion_depth = 0
        self.max_recursion_depth = 10
        
        # Simple belief state (uniform initially)
        self.qs = utils.norm_dist(np.ones(num_states))
        
        # Initialize simple matrices
        self._initialize_simple_matrices()
        
    def _initialize_simple_matrices(self):
        """Initialize simple matrices for basic active inference"""
        # A matrix: observation likelihood given states
        self.A = utils.norm_dist(np.random.rand(self.num_obs, self.num_states) + np.eye(min(self.num_obs, self.num_states)))
        
        # B matrix: state transitions given actions
        self.B = np.zeros((self.num_states, self.num_states, self.num_controls))
        for a in range(self.num_controls):
            # Mostly stay in same state with some transitions
            self.B[:, :, a] = np.eye(self.num_states) * 0.8 + np.random.rand(self.num_states, self.num_states) * 0.2
            self.B[:, :, a] = utils.norm_dist(self.B[:, :, a])
            
        # C vector: observation preferences
        self.C = utils.norm_dist(np.ones(self.num_obs) + np.random.rand(self.num_obs) * 0.5)
        
    def infer_states(self, obs):
        """Simple state inference given observation"""
        if isinstance(obs, list):
            obs = obs[0]  # Take first element if list
        obs = int(obs) % self.num_obs  # Ensure valid observation
        
        # Bayes rule: P(s|o) ∝ P(o|s) * P(s)
        likelihood = self.A[obs, :]  # P(o|s)
        prior = self.qs  # P(s) - current belief
        
        posterior = likelihood * prior
        return utils.norm_dist(posterior)
    
    def quine_step(self, obs):
        """
        Self-referential update - agent models itself modeling the world.
        
        Args:
            obs: Current observation
            
        Returns:
            action: Chosen action
        """
        if self.recursion_depth >= self.max_recursion_depth:
            # Emergency break to prevent infinite recursion
            return np.random.randint(self.num_controls)
        
        # Step 1: Normal state inference
        qs = self.infer_states(obs)
        
        # Step 2: Model myself modeling (strange loop)
        # Convert beliefs to pseudo-observations
        meta_obs = self.encode_beliefs_as_obs(qs)
        meta_qs = self.infer_states(meta_obs)
        
        # Step 3: Detect fixed point (loop closure)
        if self.detect_fixed_point(qs, meta_qs):
            self.recursion_depth += 1
            print(f"Strange loop detected at depth {self.recursion_depth}")
        else:
            # Gradually decrease depth if no loop detected
            self.recursion_depth = max(0, self.recursion_depth - 0.1)
            
        # Step 4: Update based on meta-beliefs
        self.qs = self.merge_beliefs(qs, meta_qs)
        
        # Step 5: Sample action based on expected free energy
        action = self.sample_action()
        
        # Record for analysis
        self.self_model_history.append({
            'recursion_depth': self.recursion_depth,
            'beliefs': qs.copy(),
            'meta_beliefs': meta_qs.copy(),
            'action': action,
            'obs': obs
        })
        
        return action
    
    def encode_beliefs_as_obs(self, beliefs):
        """
        Convert internal beliefs into observations of self.
        
        Args:
            beliefs: Current belief distribution over states
            
        Returns:
            obs: Encoded observation of self-state
        """
        # Find the most likely state and add some noise
        max_idx = np.argmax(beliefs)
        
        # Map state index to observation with some noise
        noise = np.random.normal(0, 0.1)
        obs_continuous = max_idx + noise
        obs_discrete = int(np.clip(obs_continuous, 0, self.num_obs - 1))
        
        return obs_discrete
    
    def detect_fixed_point(self, qs1, qs2, threshold=0.1):
        """
        Check if beliefs converge (loop closes).
        
        Args:
            qs1: First set of beliefs
            qs2: Second set of beliefs (meta-beliefs)
            threshold: Convergence threshold
            
        Returns:
            bool: Whether fixed point is detected
        """
        if qs2 is None or len(qs1) != len(qs2):
            return False
            
        # Calculate KL divergence between belief distributions
        diff = np.linalg.norm(qs1 - qs2)
        return diff < threshold
    
    def merge_beliefs(self, qs, meta_qs, alpha=0.3):
        """
        Weighted combination of object and meta-level beliefs.
        
        Args:
            qs: Object-level beliefs
            meta_qs: Meta-level beliefs
            alpha: Weight for meta-beliefs (0-1)
            
        Returns:
            merged_qs: Combined beliefs
        """
        if meta_qs is None:
            return qs
            
        # Weighted average
        merged = alpha * meta_qs + (1 - alpha) * qs
        return utils.norm_dist(merged)
    
    def sample_action(self):
        """
        Sample an action based on expected free energy.
        Simplified version that just samples based on entropy of beliefs.
        
        Returns:
            action: Chosen action index
        """
        # Higher uncertainty -> more exploration
        entropy = -np.sum(self.qs * np.log(self.qs + 1e-16))  # Manual entropy calculation
        
        # If very certain, exploit; if uncertain, explore
        if entropy < 0.1:
            # Exploitation: choose action that maintains current state
            current_state = np.argmax(self.qs)
            action = current_state % self.num_controls
        else:
            # Exploration: random action
            action = np.random.randint(self.num_controls)
        
        return action
    
    def get_recursion_metrics(self):
        """
        Get metrics about strange loop formation.
        
        Returns:
            dict: Metrics including max recursion depth, stability measures, etc.
        """
        if not self.self_model_history:
            return {}
            
        recursion_depths = [h['recursion_depth'] for h in self.self_model_history]
        
        return {
            'max_recursion_depth': max(recursion_depths),
            'avg_recursion_depth': np.mean(recursion_depths),
            'recursion_stability': np.std(recursion_depths),
            'total_loops_detected': sum(1 for d in recursion_depths if d > 0),
            'history_length': len(self.self_model_history)
        }


def calculate_simple_phi(agent):
    """
    Simplified IIT Phi calculation for consciousness measure.
    
    Args:
        agent: SimpleStrangeLoopAgent instance
        
    Returns:
        phi: Integration measure (higher = more conscious-like)
    """
    if not hasattr(agent, 'qs') or agent.qs is None:
        return 0.0
        
    # Higher phi when beliefs are more integrated/certain
    entropy = -np.sum(agent.qs * np.log(agent.qs + 1e-16))  # Manual entropy calculation
    
    # Factor in recursion depth
    recursion_bonus = min(agent.recursion_depth, 5) * 0.1
    
    # Normalize and return
    phi = 1.0 / (1.0 + entropy) + recursion_bonus
    return min(phi, 1.0)  # Cap at 1.0