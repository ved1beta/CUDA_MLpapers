"""
RPG Gradient Selection - Core Algorithm
Based on: https://github.com/complex-reasoning/RPG

RPG selectively computes gradients only for high-reward rollouts to improve
training efficiency and avoid learning from incorrect reasoning.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Rollout:
    """Represents a single reasoning rollout"""
    tokens: List[int]  # Token sequence
    log_probs: List[float]  # Log probabilities for each token
    reward: float  # Final reward (0.0 to 1.0)
    reasoning_text: str  # Human-readable reasoning
    answer: str  # Final answer

class RPGGradientSelector:

    
    def __init__(self, 
                 reward_threshold: float = 0.5,
                 top_k: int = None,
                 use_advantage: bool = True):
        """
        Args:
            reward_threshold: Minimum reward to compute gradients (default: 0.5)
            top_k: If set, only use top-k highest reward rollouts
            use_advantage: Whether to use advantage weighting (reward - baseline)
        """
        self.reward_threshold = reward_threshold
        self.top_k = top_k
        self.use_advantage = use_advantage
    
    def select_rollouts_for_training(self, 
                                    rollouts: List[Rollout]) -> Tuple[List[Rollout], List[float]]:
        """
        Select which rollouts to compute gradients for.
        
        Args:
            rollouts: List of reasoning rollouts with rewards
            
        Returns:
            selected_rollouts: Rollouts to train on
            weights: Weight for each selected rollout's gradient
        """
        if not rollouts:
            return [], []
        
        # Step 1: Filter by reward threshold
        high_reward_rollouts = [
            r for r in rollouts if r.reward >= self.reward_threshold
        ]
        
        if not high_reward_rollouts:
            print(f"Warning: No rollouts above threshold {self.reward_threshold}")
            return [], []
        
        # Step 2: Select top-k if specified
        if self.top_k is not None:
            high_reward_rollouts = sorted(
                high_reward_rollouts, 
                key=lambda r: r.reward, 
                reverse=True
            )[:self.top_k]
        
        # Step 3: Compute weights for gradient scaling
        rewards = [r.reward for r in high_reward_rollouts]
        
        if self.use_advantage:
            # Use advantage: reward - baseline
            baseline = np.mean([r.reward for r in rollouts])
            weights = [r - baseline for r in rewards]
            # Normalize to prevent very large gradients
            weights = [max(0, w) for w in weights]  # Only positive advantages
        else:
            # Use raw rewards as weights
            weights = rewards
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return high_reward_rollouts, weights
    
    def compute_policy_gradient(self, 
                               rollouts: List[Rollout]) -> Dict:
        """
        Compute policy gradient using RPG's selective approach.
        
        Standard REINFORCE: ∇L = E[R * ∇log π(a|s)]
        RPG: Only compute gradients for high-reward rollouts
        
        Args:
            rollouts: List of reasoning rollouts
            
        Returns:
            Dictionary with gradient computation info
        """
        # Select rollouts and get weights
        selected_rollouts, weights = self.select_rollouts_for_training(rollouts)
        
        if not selected_rollouts:
            return {
                "num_selected": 0,
                "num_total": len(rollouts),
                "gradients": None,
                "selection_ratio": 0.0
            }
        
        print(f"\n{'='*60}")
        print(f"RPG Gradient Selection")
        print(f"{'='*60}")
        print(f"Total rollouts: {len(rollouts)}")
        print(f"Selected rollouts: {len(selected_rollouts)}")
        print(f"Selection ratio: {len(selected_rollouts)/len(rollouts):.1%}")
        print(f"\nSelected rollouts:")
        
        # Compute gradients (pseudo-code, actual implementation would use PyTorch)
        gradients = []
        for i, (rollout, weight) in enumerate(zip(selected_rollouts, weights)):
            print(f"\n  Rollout {i+1}:")
            print(f"    Reward: {rollout.reward:.3f}")
            print(f"    Weight: {weight:.3f}")
            print(f"    Answer: {rollout.answer}")
            
            # Pseudo-gradient computation
            # In real implementation: grad = weight * sum(log_probs * advantage)
            grad_magnitude = weight * sum(rollout.log_probs)
            gradients.append({
                "rollout_idx": i,
                "weight": weight,
                "grad_magnitude": grad_magnitude
            })
        
        print(f"\n{'='*60}\n")
        
        return {
            "num_selected": len(selected_rollouts),
            "num_total": len(rollouts),
            "selected_rollouts": selected_rollouts,
            "weights": weights,
            "gradients": gradients,
            "selection_ratio": len(selected_rollouts) / len(rollouts),
            "avg_selected_reward": np.mean([r.reward for r in selected_rollouts])
        }


def example_usage():
    """Example demonstrating RPG gradient selection"""
    
    # Simulate 5 rollouts with different rewards
    rollouts = [
        Rollout(
            tokens=[1, 2, 3],
            log_probs=[-0.5, -0.3, -0.2],
            reward=1.0,
            reasoning_text="Correct reasoning path 1",
            answer="360"
        ),
        Rollout(
            tokens=[1, 4, 5],
            log_probs=[-0.6, -0.4, -0.3],
            reward=0.0,
            reasoning_text="Incorrect reasoning",
            answer="350"
        ),
        Rollout(
            tokens=[1, 2, 6],
            log_probs=[-0.4, -0.3, -0.4],
            reward=1.0,
            reasoning_text="Correct reasoning path 2",
            answer="360"
        ),
        Rollout(
            tokens=[1, 7, 8],
            log_probs=[-0.7, -0.5, -0.3],
            reward=0.0,
            reasoning_text="Another incorrect reasoning",
            answer="340"
        ),
        Rollout(
            tokens=[1, 2, 9],
            log_probs=[-0.5, -0.4, -0.2],
            reward=1.0,
            reasoning_text="Correct reasoning path 3",
            answer="360"
        ),
    ]
    
    print("="*60)
    print("Example: RPG Gradient Selection")
    print("="*60)
    print(f"\nProblem: What is 15 × 24?")
    print(f"Ground truth: 360\n")
    
    # Create RPG selector
    selector = RPGGradientSelector(
        reward_threshold=0.5,  # Only train on correct answers
        top_k=None,  # Use all high-reward rollouts
        use_advantage=True  # Use advantage weighting
    )
    
    # Compute gradients
    result = selector.compute_policy_gradient(rollouts)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Rollouts used for training: {result['num_selected']}/{result['num_total']}")
    print(f"Selection ratio: {result['selection_ratio']:.1%}")
    print(f"Average reward (selected): {result['avg_selected_reward']:.3f}")
    print("\nKey insight: RPG only computes gradients for correct answers,")
    print("avoiding the policy learning from incorrect reasoning paths.")
    print("="*60)


if __name__ == "__main__":
    example_usage()