import numpy as np
from pymdp import utils, maths
from pymdp.agent import Agent


class StrangeLoopAgent(Agent):
    def __init__(self, num_states=[3, 3], num_obs=[3, 3], num_controls=[3, 3]):
        """
        Initialize the Strange Loop Agent with self-referential capabilities.

        Args:
            num_states: List of state dimensions [world_states, self_model_states]
            num_obs: List of observation dimensions [world_obs, self_obs]
            num_controls: List of control dimensions [world_actions, self_actions]
        """
        # Store dimensions
        self.num_states = num_states
        self.num_obs = num_obs  
        self.num_controls = num_controls
        self.self_model_history = []
        self.recursion_depth = 0
        self.max_recursion_depth = 10  # Prevent infinite loops

        # Initialize matrices with some structure
        self._initialize_matrices()
        
        # Initialize the parent Agent class with our matrices
        super().__init__(
            A=self.A,
            B=self.B,
            C=self.C,
            num_controls=self.num_controls
        )

    def _initialize_matrices(self):
        """Initialize A, B, and C matrices with proper PyMDP structure"""
        # A matrices (observation likelihoods) - PyMDP format
        # For PyMDP, A matrices should map from factorized states to observations
        self.A = utils.obj_array(len(self.num_obs))
        
        # A[0]: World observations given world and self states  
        A0 = np.random.rand(self.num_obs[0], self.num_states[0], self.num_states[1]) + 0.1
        # Add diagonal structure for better observability
        for i in range(min(self.num_obs[0], self.num_states[0])):
            A0[i, i, :] += 1.0
        # Normalize across observation dimension
        for s0 in range(self.num_states[0]):
            for s1 in range(self.num_states[1]):
                A0[:, s0, s1] = utils.norm_dist(A0[:, s0, s1])
        self.A[0] = A0
        
        # A[1]: Self observations given world and self states
        A1 = np.random.rand(self.num_obs[1], self.num_states[0], self.num_states[1]) + 0.1  
        # Self observations more dependent on self-model states
        for i in range(min(self.num_obs[1], self.num_states[1])):
            A1[i, :, i] += 1.0
        # Normalize across observation dimension
        for s0 in range(self.num_states[0]):
            for s1 in range(self.num_states[1]):
                A1[:, s0, s1] = utils.norm_dist(A1[:, s0, s1])
        self.A[1] = A1

        # B matrices (state transitions) - separate for each factor  
        self.B = utils.obj_array(len(self.num_states))
        
        # B[0]: World state transitions
        B0 = np.zeros((self.num_states[0], self.num_states[0], self.num_controls[0]))
        for a in range(self.num_controls[0]):
            # Mostly diagonal with some transitions
            B0[:, :, a] = np.eye(self.num_states[0]) * 0.8 + np.random.rand(self.num_states[0], self.num_states[0]) * 0.2
            B0[:, :, a] = utils.norm_dist(B0[:, :, a])
        self.B[0] = B0
        
        # B[1]: Self-model state transitions (more dynamic)
        B1 = np.zeros((self.num_states[1], self.num_states[1], self.num_controls[1]))
        for a in range(self.num_controls[1]):
            B1[:, :, a] = np.eye(self.num_states[1]) * 0.6 + np.random.rand(self.num_states[1], self.num_states[1]) * 0.4
            B1[:, :, a] = utils.norm_dist(B1[:, :, a])
        self.B[1] = B1

        # C vectors (preferences) - separate for each modality
        self.C = utils.obj_array(len(self.num_obs))
        
        # Preferences for world observations (neutral)
        self.C[0] = utils.norm_dist(np.ones(self.num_obs[0]))
        
        # Slight preference for self-awareness observations
        pref_array = np.ones(self.num_obs[1])
        if len(pref_array) > 0:
            pref_array[0] += 0.0
        if len(pref_array) > 1:
            pref_array[1] += 0.2
        if len(pref_array) > 2:
            pref_array[2] += 0.5
        self.C[1] = utils.norm_dist(pref_array)

    def quine_step(self, obs):
        """
        Self-referential update - agent models itself modeling the world.

        Args:
            obs: Current observation tuple [world_obs, self_obs]

        Returns:
            action: Chosen action tuple [world_action, self_action]
        """
        if self.recursion_depth >= self.max_recursion_depth:
            # Emergency break to prevent infinite recursion
            return self.sample_action()

        # Step 1: Normal active inference
        qs = self.infer_states(obs)
        
        # Initialize qs if needed
        if qs is None:
            self.qs = [utils.norm_dist(np.ones(n)) for n in self.num_states]
            qs = self.qs

        # Step 2: Model myself modeling (strange loop)
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

        # Step 5: Infer policies and sample action
        try:
            # Use PyMDP's infer_policies method if available
            if hasattr(super(), 'infer_policies'):
                q_pi = super().infer_policies()
                self.q_pi = q_pi
            action = self.sample_action()
        except Exception as e:
            # Fallback to random action if policy inference fails
            action = [np.random.randint(n) for n in self.num_controls]

        # Record for analysis
        self.self_model_history.append({
            'recursion_depth': self.recursion_depth,
            'beliefs': qs,
            'meta_beliefs': meta_qs if meta_qs is not None else None,
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
        # Discretize beliefs into observable states
        encoded_obs = []
        for belief in beliefs:
            # Find the most likely state
            max_idx = np.argmax(belief)
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.1, len(belief))
            noisy_belief = belief + noise
            noisy_belief = np.clip(noisy_belief, 0, 1)
            noisy_belief = noisy_belief / np.sum(noisy_belief)  # Renormalize
            encoded_obs.append(np.argmax(noisy_belief))

        return encoded_obs

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

        # Calculate difference between belief distributions
        total_diff = 0
        for q1, q2 in zip(qs1, qs2):
            diff = np.linalg.norm(q1 - q2)
            total_diff += diff

        avg_diff = total_diff / len(qs1)
        return avg_diff < threshold

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

        merged = []
        for q, meta_q in zip(qs, meta_qs):
            # Weighted average
            merged_q = alpha * meta_q + (1 - alpha) * q
            # Renormalize
            merged_q = merged_q / np.sum(merged_q)
            merged.append(merged_q)

        return merged

    def sample_action(self):
        """
        Sample an action based on current policies.

        Returns:
            action: Tuple of actions for each control factor
        """
        if not hasattr(self, 'q_pi') or self.q_pi is None:
            # Random action if no policies computed
            return [np.random.randint(n) for n in self.num_controls]

        # Get the best action from the inferred policies
        try:
            # Use PyMDP's sample_action if available
            if hasattr(super(), 'sample_action'):
                return super().sample_action()
            else:
                # Simplified action sampling based on policy probabilities
                best_policy_idx = np.argmax(self.q_pi)
                # Extract first action from best policy
                action = []
                for i, n_controls in enumerate(self.num_controls):
                    # Sample action for each control factor (simplified)
                    if hasattr(self, 'policies') and self.policies is not None and len(self.policies) > best_policy_idx:
                        # Use action from policy if available
                        action.append(self.policies[best_policy_idx][0][i])
                    else:
                        # Random action as fallback
                        action.append(np.random.randint(n_controls))
                
                return action
        except Exception as e:
            # Fallback to random action
            return [np.random.randint(n) for n in self.num_controls]

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
        agent: StrangeLoopAgent instance

    Returns:
        phi: Integration measure (higher = more conscious-like)
    """
    if not hasattr(agent, 'qs') or agent.qs is None:
        return 0.0

    # Higher phi when beliefs are more integrated/certain
    entropy = sum([-np.sum(q * np.log(q + 1e-16)) for q in agent.qs])

    # Factor in recursion depth
    recursion_bonus = min(agent.recursion_depth, 5) * 0.1

    # Normalize and return
    phi = 1.0 / (1.0 + entropy) + recursion_bonus
    return min(phi, 1.0)  # Cap at 1.0

