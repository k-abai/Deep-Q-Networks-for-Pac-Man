#!/usr/bin/env python
"""
play_cv.py – visualise a trained Pac‑Man agent with OpenCV animation.
Usage example:
    python play_cv.py --layout classic --scale 3 --speed 70
    python play_cv.py --layout classic --headless  # No display
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import torch
from pacman_env import PacmanEnv
from dqn_agent import DQN, DEVICE

# Try to import cv2, but handle gracefully if display is not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - running in headless mode")

# ───────────────────────── helper ─────────────────────────
def load_net(weight_file: Path, n_actions: int, obs_shape) -> DQN:
    net = DQN(obs_shape, n_actions).to(DEVICE)
    net.load_state_dict(torch.load(weight_file, map_location=DEVICE))
    net.eval()
    return net

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

# ───────────────────────── visual play loop ──────────────────────
def play_visual(layout: str, net: DQN, episodes: int, delay_ms: int, scale: int):
    """Play with OpenCV visualization."""
    env = PacmanEnv(layout)
    wins = 0
    cv2.namedWindow("Pac-Man", cv2.WINDOW_AUTOSIZE)  # Also changed the dash
    cv2.waitKey(1)  # Let macOS initialize the window

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, step, win = False, 0, False

        while not done and step < 1000:
            with torch.no_grad():
                action = int(net(torch.as_tensor(state, device=DEVICE)
                                 .unsqueeze(0)).argmax())
            state, _, done, _, _ = env.step(action)
            step += 1
            if done and not env.pellets:
                win = True

            frame = cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR)
            if scale > 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
            cv2.putText(frame, f"{layout}  Ep {ep}/{episodes}  step {step}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.imshow("Pac‑Man", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                print("Quit requested.")
                env.close(); cv2.destroyAllWindows(); sys.exit(0)

        wins += win
        print(f"Episode {ep}: {'WIN' if win else 'LOSE'} in {step} steps")

    print(f"\n{wins}/{episodes} wins ({wins/episodes*100:.1f}%)")
    env.close(); cv2.destroyAllWindows()

# ───────────────────────── headless play loop ──────────────────────
def play_headless(layout: str, net: DQN, episodes: int):
    """Play without display - text output only."""
    env = PacmanEnv(layout)
    wins = 0
    
    print(f"Running DQN agent on {layout} layout (headless mode)")
    print(f"Device: {DEVICE}")
    print("-" * 50)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, step, win = False, 0, False
        
        print(f"\nEpisode {ep}/{episodes}: Starting game")
        initial_pellets = len(env.pellets)
        print(f"  Initial pellets: {initial_pellets}")
        print(f"  Starting position: Pac-Man at {env.pac_pos}, Ghost(s) at {env.ghost_pos}")

        while not done and step < 1000:
            with torch.no_grad():
                q_values = net(torch.as_tensor(state, device=DEVICE).unsqueeze(0))
                action = int(q_values.argmax())
            
            # Print action details for first few steps or periodically
            if step < 5 or step % 100 == 0:
                actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
                q_vals = q_values.squeeze().cpu().numpy()
                print(f"  Step {step}: Action={actions[action]}, "
                      f"Q-values={[f'{q:.2f}' for q in q_vals]}, "
                      f"Pellets left: {len(env.pellets)}")
            
            state, reward, done, _, _ = env.step(action)
            step += 1
            
            if done and not env.pellets:
                win = True

        wins += win
        pellets_collected = initial_pellets - len(env.pellets)
        result = "WIN" if win else "LOSE"
        print(f"Episode {ep}: {result} in {step} steps")
        print(f"  Pellets collected: {pellets_collected}/{initial_pellets}")
        print(f"  Final position: Pac-Man at {env.pac_pos}, Ghost(s) at {env.ghost_pos}")

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"Wins: {wins}/{episodes} ({wins/episodes*100:.1f}%)")
    print(f"Layout: {layout}")
    print(f"Model device: {DEVICE}")
    print(f"{'='*50}")
    
    env.close()

# ───────────────────────── main play function ──────────────────────
def play(layout: str, net: DQN, episodes: int, delay_ms: int = 100, scale: int = 3, headless: bool = False):
    """Play with automatic display detection or explicit headless mode."""
    if headless or not is_display_available():
        if not headless:
            print("No display available - running in headless mode")
        play_headless(layout, net, episodes)
    else:
        play_visual(layout, net, episodes, delay_ms, scale)

# ───────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout",
                        choices=["classic", "empty", "spiral", "spiral_harder"],
                        required=True)
    parser.add_argument("--model", type=Path,
                        help="explicit path to .pt weight file "
                             "(default pacman_dqn_<layout>.pt)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--speed", type=int, default=100,
                        help="delay between frames in ms (visual mode only)")
    parser.add_argument("--scale", type=int, default=3,
                        help="integer up‑scale factor for window size (visual mode only)")
    parser.add_argument("--headless", action="store_true",
                        help="run without display (text output only)")
    args = parser.parse_args()

    weight_path = args.model or Path(f"pacman_dqn_{args.layout}.pt")
    if not weight_path.exists():
        sys.exit(f"weight file {weight_path} not found")

    # Build a temp env to discover observation shape & action count
    tmp_env = PacmanEnv(args.layout)
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    net = load_net(weight_path, n_actions, obs_shape)
    play(args.layout, net, args.episodes, args.speed, args.scale, args.headless)