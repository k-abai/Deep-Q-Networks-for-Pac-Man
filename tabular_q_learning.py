#!/usr/bin/env python
"""
tabular_q_learning.py

Implements tabular Q-learning for Pac-Man on spiral layouts.
State space: (pac_x, pac_y, ghost_x, ghost_y, pellets_remaining_bitmask)

Usage:
    python tabular_q_learning.py --layout spiral --episodes 5000 --visualize
    python tabular_q_learning.py --layout spiral_harder --episodes 3000 --play
    python tabular_q_learning.py --layout spiral --play --headless  # No display
"""

from __future__ import annotations
import argparse
import pickle
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Try to import cv2, but handle gracefully if display is not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - running in headless mode")

from pacman_env import PacmanEnv


class TabularQLearning:
    """Tabular Q-learning agent for Pac-Man."""
    
    def __init__(
        self,
        layout: str,
        alpha: float = 0.1,      # learning rate
        gamma: float = 0.99,     # discount factor
        epsilon: float = 1.0,    # initial exploration rate
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.layout = layout
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q[state][action] = value
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Create environment to understand the layout
        self.env = PacmanEnv(layout)
        self.n_actions = self.env.action_space.n
        
        # Find valid positions (non-wall cells)
        self.valid_positions = self._get_valid_positions()
        
        # Initial pellet positions for encoding
        self.env.reset()
        self.initial_pellets = set(self.env.pellets)
        self.pellet_to_bit = {pellet: i for i, pellet in enumerate(sorted(self.initial_pellets))}
        
        print(f"Initialized tabular Q-learning for {layout}")
        print(f"Valid positions: {len(self.valid_positions)}")
        print(f"Initial pellets: {self.initial_pellets}")
        print(f"Estimated state space size: {len(self.valid_positions)**2 * (2**len(self.initial_pellets))}")
    
    def _get_valid_positions(self) -> Set[Tuple[int, int]]:
        """Find all valid (non-wall) positions in the environment."""
        valid = set()
        h, w = self.env.floor.shape
        for i in range(h):
            for j in range(w):
                if self.env.floor[i, j] == 0:  # 0 = empty space
                    valid.add((i, j))
        return valid
    
    def _encode_state(self, pac_pos: Tuple[int, int], ghost_pos: Tuple[int, int], 
                     current_pellets: Set[Tuple[int, int]]) -> Tuple:
        """Encode game state as a hashable tuple."""
        # Encode which pellets remain as a bitmask
        pellet_mask = 0
        for pellet in current_pellets:
            if pellet in self.pellet_to_bit:
                pellet_mask |= (1 << self.pellet_to_bit[pellet])
        
        return (pac_pos[0], pac_pos[1], ghost_pos[0], ghost_pos[1], pellet_mask)
    
    def _get_state(self, env=None) -> Tuple:
        """Get current state from environment."""
        # Use provided environment or fall back to self.env
        env = env or self.env
        
        # Handle multiple ghosts by using the first one
        ghost_pos = env.ghost_pos[0] if env.ghost_pos else (0, 0)
        current_pellets = set(env.pellets)
        state = self._encode_state(env.pac_pos, ghost_pos, current_pellets)
        
        # Validate state components are reasonable
        if not (0 <= state[0] < env.h and 0 <= state[1] < env.w):
            print(f"Warning: Invalid Pac-Man position in state: {state}")
        if not (0 <= state[2] < env.h and 0 <= state[3] < env.w):
            print(f"Warning: Invalid ghost position in state: {state}")
            
        return state
    
    def select_action(self, state: Tuple, debug: bool = False) -> int:
        """
        TODO: IMPLEMENT THIS FUNCTION
        
        Select an action using epsilon-greedy policy.
        This balances exploration (trying random actions) with exploitation (using learned knowledge).
        
        PARAMETERS:
        - state: Tuple - The current state (from _encode_state)
        - debug: bool - Whether to print debug information (you can ignore this)
        
        
        VARIABLES AND TYPES YOU'LL WORK WITH:
        - self.epsilon: float (current exploration probability, e.g., 0.1 means 10% chance of random action)
        - self.n_actions: int (number of possible actions, equals 4 for up/down/left/right)
        - self.q_table: Dict[Tuple, Dict[int, float]] (the Q-table storing learned values)
        - self.q_table[state]: Dict[int, float] (Q-values for all actions in given state)
        - self.q_table[state][action]: float (Q-value for specific state-action pair)
        
        NUMPY FUNCTIONS YOU'LL NEED:
        - np.random.random(): returns random float between 0.0 and 1.0
        - np.random.randint(n): returns random integer from 0 to n-1
        - np.argmax(list_or_array): returns index of maximum value
        
        
        EDGE CASE HANDLING:
        - If all Q-values are 0.0 (unseen state), choose random action
        - Check using: if all(q == 0.0 for q in q_values)
        
        RETURN TYPE: int
        - Returns action number (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        """
        
        return 0  # Replace with actual implementation
    
    def update_q_value(self, state: Tuple, action: int, reward: float, 
                      next_state: Tuple, done: bool):
        """
        TODO: IMPLEMENT THIS FUNCTION
        
        Update Q-value using the Q-learning update rule.
        This is the CORE of the Q-learning algorithm!
        
        PARAMETERS:
        - state: Tuple - Current state before taking action
        - action: int - Action that was taken (0,1,2,3)
        - reward: float - Immediate reward received after taking action
        - next_state: Tuple - State reached after taking action  
        - done: bool - Whether episode ended (True if game over, False otherwise)
        
        
        VARIABLES AND TYPES:
        - self.alpha: float (learning rate, typically 0.1)
        - self.gamma: float (discount factor, typically 0.99)
        - self.q_table[state][action]: float (current Q-value to update)
        - current_q: float (store current Q-value before updating)
        - target: float (the target value we want Q-value to move towards)
        
        
        RETURN TYPE: None (this function modifies self.q_table in-place)
        """
        
        pass  # Replace with actual implementation
    
    def decay_epsilon(self):
        """
        TODO: IMPLEMENT THIS FUNCTION
        
        Decay the exploration rate (epsilon) over time.
        This implements the exploration-exploitation tradeoff strategy.
        
        CONCEPT:
        - Start with high epsilon (lots of exploration)
        - Gradually decrease epsilon (more exploitation as we learn)
        - Never go below epsilon_min (always keep some exploration)
        
        VARIABLES AND TYPES:
        - self.epsilon: float (current exploration rate)
        - self.epsilon_min: float (minimum allowed exploration rate, e.g., 0.01)
        - self.epsilon_decay: float (decay factor, e.g., 0.995 means multiply by 0.995 each episode)
        
        

        

        
        RETURN TYPE: None (this function modifies self.epsilon in-place)
        """

        pass  # Replace with actual implementation
    
    def train(self, episodes: int, verbose: bool = True) -> Dict:
        """Train the Q-learning agent."""
        rewards_per_episode = []
        wins_per_episode = []
        steps_per_episode = []
        epsilon_history = []
        
        # Track recent performance
        recent_window = 100
        recent_rewards = deque(maxlen=recent_window)
        recent_wins = deque(maxlen=recent_window)
        
        for episode in range(1, episodes + 1):
            state, _ = self.env.reset()
            current_state = self._get_state()
            
            total_reward = 0
            steps = 0
            won = False
            max_steps = 1000
            
            while steps < max_steps:
                action = self.select_action(current_state)
                next_obs, reward, done, _, _ = self.env.step(action)
                next_state = self._get_state()
                
                self.update_q_value(current_state, action, reward, next_state, done)
                
                current_state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    if len(self.env.pellets) == 0:  # Won by collecting all pellets
                        won = True
                    break
            
            # Track metrics
            rewards_per_episode.append(total_reward)
            wins_per_episode.append(1 if won else 0)
            steps_per_episode.append(steps)
            epsilon_history.append(self.epsilon)
            
            recent_rewards.append(total_reward)
            recent_wins.append(1 if won else 0)
            
            self.decay_epsilon()
            
            # Progress reporting
            if verbose and (episode % 500 == 0 or episode == episodes):
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                win_rate = np.mean(recent_wins) if recent_wins else 0
                avg_steps = np.mean(list(recent_rewards)[-recent_window:]) if len(rewards_per_episode) >= recent_window else np.mean(steps_per_episode)
                
                print(f"Episode {episode:4d} | "
                      f"Avg Reward: {avg_reward:6.1f} | "
                      f"Win Rate: {win_rate:.2f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Q-table size: {len(self.q_table)}")
        
        return {
            'rewards': rewards_per_episode,
            'wins': wins_per_episode,
            'steps': steps_per_episode,
            'epsilon': epsilon_history,
        }
    
    def save_policy(self, filepath: Path):
        """Save the Q-table to disk."""
        # Convert defaultdict to regular dict for pickling
        q_dict = {state: dict(actions) for state, actions in self.q_table.items()}
        
        data = {
            'q_table': q_dict,
            'layout': self.layout,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'pellet_to_bit': self.pellet_to_bit,
            'initial_pellets': self.initial_pellets,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved Q-table to {filepath}")
    
    def load_policy(self, filepath: Path):
        """Load Q-table from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in data['q_table'].items():
            for action, value in actions.items():
                self.q_table[state][action] = value
        
        self.layout = data['layout']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.pellet_to_bit = data['pellet_to_bit']
        self.initial_pellets = data['initial_pellets']
        
    def analyze_q_table(self):
        """Print Q-table statistics for debugging."""
        print(f"Q-table contains {len(self.q_table)} states")
        
        if len(self.q_table) > 0:
            # Find state with highest Q-values
            best_state = None
            best_max_q = float('-inf')
            
            for state, actions in self.q_table.items():
                max_q = max(actions.values()) if actions else 0
                if max_q > best_max_q:
                    best_max_q = max_q
                    best_state = state
            
            print(f"Best state: {best_state}")
            print(f"Best Q-values: {dict(self.q_table[best_state])}")
            print(f"Max Q-value found: {best_max_q}")
            
            # Check a few random states
            sample_states = list(self.q_table.keys())[:5]
            print(f"\nSample states and Q-values:")
            for state in sample_states:
                q_vals = [self.q_table[state][a] for a in range(4)]
                print(f"  {state}: {q_vals}")
    
    def get_initial_state_for_debug(self) -> Tuple:
        """Get the initial state to check if it's in Q-table."""
        temp_env = PacmanEnv(self.layout)
        temp_env.reset()
        ghost_pos = temp_env.ghost_pos[0] if temp_env.ghost_pos else (0, 0)
        current_pellets = set(temp_env.pellets)
        initial_state = self._encode_state(temp_env.pac_pos, ghost_pos, current_pellets)
        temp_env.close()
        return initial_state


def is_display_available():
    """Check if display is available for OpenCV."""
    if not CV2_AVAILABLE:
        return False
    
    # Only check DISPLAY for Linux (not macOS)
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    
    # Try to create a test window
    try:
        cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("test")
        return True
    except Exception:
        return False


def visualize_training(history: Dict, layout: str):
    """Plot training metrics."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    episodes = range(1, len(history['rewards']) + 1)
    
    # Rewards
    ax1.plot(episodes, history['rewards'], alpha=0.3, color='blue', linewidth=0.5)
    # Moving average
    window = 100
    if len(history['rewards']) >= window:
        moving_avg = np.convolve(history['rewards'], np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(history['rewards']) + 1), moving_avg, 'red', linewidth=2)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Win rate
    window = 50
    if len(history['wins']) >= window:
        win_rate = np.convolve(history['wins'], np.ones(window)/window, mode='valid')
        ax2.plot(range(window, len(history['wins']) + 1), win_rate, 'green', linewidth=2)
    ax2.set_title(f'Win Rate (sliding window = {window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    
    # Steps per episode
    ax3.plot(episodes, history['steps'], alpha=0.3, color='orange', linewidth=0.5)
    if len(history['steps']) >= window:
        moving_avg_steps = np.convolve(history['steps'], np.ones(window)/window, mode='valid')
        ax3.plot(range(window, len(history['steps']) + 1), moving_avg_steps, 'red', linewidth=2)
    ax3.set_title('Steps per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.grid(True)
    
    # Epsilon decay
    ax4.plot(episodes, history['epsilon'], 'purple', linewidth=2)
    ax4.set_title('Exploration Rate (ε)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon')
    ax4.grid(True)
    
    plt.suptitle(f'Tabular Q-Learning Training Progress - {layout.title()}')
    plt.tight_layout()
    plt.show()


def play_policy_visual(agent: TabularQLearning, episodes: int = 10, delay_ms: int = 200):
    """Visualize the trained policy with OpenCV."""
    env = PacmanEnv(agent.layout)
    wins = 0
    
    # Debug the Q-table before playing
    print("\n" + "="*50)
    print("DEBUGGING Q-TABLE")
    print("="*50)
    agent.analyze_q_table()
    
    initial_state = agent.get_initial_state_for_debug()
    print(f"\nInitial state: {initial_state}")
    print(f"Initial state in Q-table: {initial_state in agent.q_table}")
    if initial_state in agent.q_table:
        q_vals = [agent.q_table[initial_state][a] for a in range(4)]
        print(f"Initial state Q-values: {q_vals}")
    print("="*50)
    
    cv2.namedWindow("Tabular Q-Learning Pac-Man", cv2.WINDOW_AUTOSIZE)
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    try:
        for ep in range(1, episodes + 1):
            state, _ = env.reset()
            current_state = agent._get_state(env)
            
            done = False
            steps = 0
            won = False
            max_steps = 500
            
            while not done and steps < max_steps:
                # Enable debugging for first few steps of first episode
                debug_mode = (ep == 1 and steps < 5)
                action = agent.select_action(current_state, debug=debug_mode)
                next_obs, reward, done, _, _ = env.step(action)
                current_state = agent._get_state(env)
                steps += 1
                
                if done and len(env.pellets) == 0:
                    won = True
                
                # Render
                frame_rgb = env.render(mode="rgb_array")
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Scale up for better visibility
                scale = 4
                h, w = frame_bgr.shape[:2]
                frame_bgr = cv2.resize(frame_bgr, (w * scale, h * scale), 
                                     interpolation=cv2.INTER_NEAREST)
                
                # Add text
                text = f"Tabular Q-Learning | {agent.layout} | Ep {ep}/{episodes} | Step {steps}"
                cv2.putText(frame_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 255), 1)
                
                # Show Q-values for current state
                q_vals = [agent.q_table[current_state][a] for a in range(4)]
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                for i, (action_name, q_val) in enumerate(zip(actions, q_vals)):
                    y_pos = 50 + i * 20
                    color = (0, 255, 0) if i == action else (255, 255, 255)
                    cv2.putText(frame_bgr, f"{action_name}: {q_val:.2f}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1)
                
                cv2.imshow("Tabular Q-Learning Pac-Man", frame_bgr)
                
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord('q'):
                    print("Quit requested.")
                    env.close()
                    cv2.destroyAllWindows()
                    return
            
            wins += won
            print(f"Episode {ep}: {'WIN' if won else 'LOSE'} in {steps} steps")
        
        print(f"\nFinished: {wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
        
    finally:
        agent.epsilon = original_epsilon
        env.close()
        cv2.destroyAllWindows()


def play_policy_headless(agent: TabularQLearning, episodes: int = 10):
    """Test the trained policy without visual display."""
    env = PacmanEnv(agent.layout)
    wins = 0
    
    # Debug the Q-table before playing
    print("\n" + "="*50)
    print("DEBUGGING Q-TABLE")
    print("="*50)
    agent.analyze_q_table()
    
    initial_state = agent.get_initial_state_for_debug()
    print(f"\nInitial state: {initial_state}")
    print(f"Initial state in Q-table: {initial_state in agent.q_table}")
    if initial_state in agent.q_table:
        q_vals = [agent.q_table[initial_state][a] for a in range(4)]
        print(f"Initial state Q-values: {q_vals}")
    print("="*50)
    
    # Set epsilon to 0 for pure exploitation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    try:
        for ep in range(1, episodes + 1):
            state, _ = env.reset()
            current_state = agent._get_state(env)
            
            done = False
            steps = 0
            won = False
            max_steps = 500
            
            print(f"\nEpisode {ep}: Starting game")
            
            while not done and steps < max_steps:
                # Enable debugging for first few steps of first episode
                debug_mode = (ep == 1 and steps < 3)
                action = agent.select_action(current_state, debug=debug_mode)
                next_obs, reward, done, _, _ = env.step(action)
                current_state = agent._get_state(env)
                steps += 1
                
                if done and len(env.pellets) == 0:
                    won = True
                
                # Print some progress info
                if steps % 50 == 0:
                    pellets_left = len(env.pellets)
                    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                    q_vals = [agent.q_table[current_state][a] for a in range(4)]
                    best_action = actions[np.argmax(q_vals)]
                    print(f"  Step {steps}: Pellets left: {pellets_left}, Best action: {best_action}")
            
            wins += won
            result = "WIN" if won else "LOSE"
            print(f"Episode {ep}: {result} in {steps} steps (Pellets left: {len(env.pellets)})")
        
        print(f"\nFinished: {wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
        
    finally:
        agent.epsilon = original_epsilon
        env.close()


def play_policy(agent: TabularQLearning, episodes: int = 10, delay_ms: int = 200, headless: bool = False):
    """Play policy with or without visual display."""
    if headless or not is_display_available():
        if not headless:
            print("No display available - running in headless mode")
        play_policy_headless(agent, episodes)
    else:
        play_policy_visual(agent, episodes, delay_ms)


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-learning for Pac-Man")
    parser.add_argument("--layout", choices=["spiral", "spiral_harder"], 
                       default="spiral", help="Which spiral layout to use")
    parser.add_argument("--episodes", type=int, default=3000, 
                       help="Number of training episodes")
    parser.add_argument("--alpha", type=float, default=0.1, 
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, 
                       help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, 
                       help="Initial exploration rate")
    parser.add_argument("--save", type=Path, 
                       help="Save trained policy to file")
    parser.add_argument("--load", type=Path, 
                       help="Load policy from file")
    parser.add_argument("--visualize", action="store_true", 
                       help="Show training plots")
    parser.add_argument("--play", action="store_true", 
                       help="Play the policy with visualization")
    parser.add_argument("--headless", action="store_true", 
                       help="Run without display (text output only)")
    parser.add_argument("--play-episodes", type=int, default=10, 
                       help="Number of episodes to play")
    parser.add_argument("--speed", type=int, default=200, 
                       help="Delay between frames (ms) when playing")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = TabularQLearning(
        layout=args.layout,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )
    
    # Load existing policy if specified
    if args.load:
        if not args.load.exists():
            print(f"Error: Policy file {args.load} not found")
            sys.exit(1)
        agent.load_policy(args.load)
    
    # Train or play
    if args.play:
        print(f"Playing trained policy on {args.layout}...")
        play_policy(agent, args.play_episodes, args.speed, args.headless)
    else:
        print(f"Training tabular Q-learning on {args.layout} for {args.episodes} episodes...")
        history = agent.train(args.episodes)
        
        if args.visualize and not args.headless:
            try:
                visualize_training(history, args.layout)
            except Exception as e:
                print(f"Could not show training plots: {e}")
                print("Consider using --headless flag")
        
        # Save policy
        save_path = args.save or Path(f"tabular_q_{args.layout}.pkl")
        agent.save_policy(save_path)
        
        # Debug Q-table
        print(f"\nQ-table Analysis:")
        agent.analyze_q_table()
        
        # Quick test
        print(f"\nTesting trained policy...")
        play_policy(agent, episodes=5, delay_ms=100, headless=args.headless)


if __name__ == "__main__":
    main()