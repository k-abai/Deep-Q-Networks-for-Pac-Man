#!/usr/bin/env python
"""
train_dqn.py

Train a visual DQN agent on Pac-Man across multiple layouts using:
  - Dueling Double DQN
  - Prioritized Experience Replay
  - Optional n-step returns
  - NoisyNet exploration

Usage examples:

    # Default curriculum: empty → spiral → spiral_harder → classic
    python train_dqn.py

    # More episodes per layout and slightly longer max steps
    python train_dqn.py --episodes-per-layout 1500 --max-steps 800

    # Train only on classic
    python train_dqn.py --layouts classic --episodes-per-layout 3000

After training, this script writes:
    pacman_dqn_empty.pt
    pacman_dqn_spiral.pt
    pacman_dqn_spiral_harder.pt
    pacman_dqn_classic.pt

All four files contain the same trained weights; this just matches the
naming convention expected by play_cv.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pacman_env import PacmanEnv
from dqn_agent import DQNAgent, DQN, DEVICE

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - training will run headless (no live rendering)")

# Re-export for play_pacman.py compatibility:
# play_pacman.py does: from train_dqn import PacmanEnv, DQN, DEVICE
PacmanEnv = PacmanEnv
DQN = DQN
DEVICE = DEVICE

# Canonical frame size we train on
FRAME_SIZE = 84


# ───────────────────────── preprocessing ─────────────────────────
def preprocess_obs(obs: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """
    Convert env observation (H, W, C) to a fixed size (size x size x C).
    Uses bilinear interpolation via PyTorch.

    We keep it RGB uint8; the network normalizes to [0,1] internally.
    """
    h, w, c = obs.shape
    if h == size and w == size:
        return obs.astype(np.uint8)

    # (H, W, C) -> (1, C, H, W)
    img = torch.as_tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False)
    # Back to (H, W, C), clamp and cast to uint8
    img = img.squeeze(0).permute(1, 2, 0)
    img = img.clamp(0, 255).byte().cpu().numpy()
    return img


# ───────────────────────── epsilon schedule ──────────────────────
def linear_epsilon_schedule(
    step: int,
    eps_start: float,
    eps_final: float,
    eps_decay_frames: int,
) -> float:
    """
    Linearly decay epsilon from eps_start to eps_final over eps_decay_frames steps.
    """
    if eps_decay_frames <= 0:
        return eps_final
    fraction = min(1.0, step / float(eps_decay_frames))
    return eps_start + fraction * (eps_final - eps_start)


# ───────────────────────── training loop ─────────────────────────
def train_on_layout(
    layout: str,
    agent: DQNAgent,
    episodes: int,
    max_steps: int,
    global_step_start: int,
    eps_start: float,
    eps_final: float,
    eps_decay_frames: int,
    print_every: int = 20,
    render: bool = False,
    render_every: int = 50,
    render_speed_ms: int = 50,
):
    """
    Train agent for a given number of episodes on a single layout.

    Returns:
        global_step (int): updated total environment steps
    """
    env = PacmanEnv(layout)
    global_step = global_step_start
    local_step = 0  # epsilon schedule resets per layout

    episode_rewards = []
    episode_wins = []

    print(f"\n=== Training on layout '{layout}' for {episodes} episodes ===")

    render_enabled = render and CV2_AVAILABLE
    if render_enabled:
        window_name = f"Pac-Man Training ({layout})"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        done = False
        total_reward = 0.0
        steps = 0
        won = False

        while not done and steps < max_steps:
            # Epsilon based on local_step (per-layout), not global_step
            epsilon = linear_epsilon_schedule(
                local_step,
                eps_start=eps_start,
                eps_final=eps_final,
                eps_decay_frames=eps_decay_frames,
            )

            # ε-greedy action (plus NoisyNet inside the agent)
            action = agent.act(obs, epsilon=epsilon)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            # ───── optional live rendering ─────
            if render_enabled and (ep % render_every == 0):
                frame = env.render("rgb_array")          # HWC, RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # scale up for visibility
                scale = 4
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_NEAREST,
                )

                # HUD text
                hud = (
                    f"Layout: {layout} | Ep {ep}/{episodes} | "
                    f"Step {steps} | Global step {global_step} "
                    f"| epsilon ~ {epsilon:.3f}"
                )
                cv2.putText(
                    frame,
                    hud,
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(render_speed_ms) & 0xFF
                if key == ord('q'):
                    print("Render: 'q' pressed – turning off rendering for this run.")
                    render_enabled = False
                    cv2.destroyWindow(window_name)

            next_obs = preprocess_obs(next_obs)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)

            # One training update step (if replay buffer is warm)
            _metrics = agent.update()

            obs = next_obs
            total_reward += reward
            steps += 1
            global_step += 1
            local_step += 1

        if done and len(env.pellets) == 0:
            won = True

        episode_rewards.append(total_reward)
        episode_wins.append(1 if won else 0)

        if ep % print_every == 0 or ep == 1 or ep == episodes:
            recent = 100
            avg_reward = (
                np.mean(episode_rewards[-recent:])
                if len(episode_rewards) >= recent
                else np.mean(episode_rewards)
            )
            win_rate = (
                np.mean(episode_wins[-recent:])
                if len(episode_wins) >= recent
                else np.mean(episode_wins)
            )
            print(
                f"[{layout}] Ep {ep:4d}/{episodes}  "
                f"steps: {steps:3d}  "
                f"reward: {total_reward:7.2f}  "
                f"win: {'Y' if won else 'N'}  "
                f"avg_R(last {min(recent, len(episode_rewards))}): {avg_reward:7.2f}  "
                f"win_rate: {win_rate*100:5.1f}%"
            )

    env.close()
    if render and CV2_AVAILABLE:
        cv2.destroyAllWindows()

    return global_step


# ───────────────────────── main ─────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Pac-Man")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render some training episodes with OpenCV.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=50,
        help="Render every Nth episode (per layout) when --render is enabled.",
    )
    parser.add_argument(
        "--render-speed",
        type=int,
        default=50,
        help="Delay between rendered frames in ms (lower = faster).",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["empty", "spiral", "spiral_harder", "classic"],
        choices=["classic", "empty", "spiral", "spiral_harder"],
        help="Layouts to train on (in order).",
    )
    parser.add_argument(
        "--episodes-per-layout",
        type=int,
        default=1000,
        help="Number of training episodes per layout.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="Replay buffer capacity.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for updates.",
    )
    parser.add_argument(
        "--learn-start",
        type=int,
        default=10_000,
        help="Steps before starting gradient updates.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=3,
        help="Number of steps for n-step returns (1 = standard DQN).",
    )
    parser.add_argument(
        "--eps-start",
        type=float,
        default=1.0,
        help="Initial epsilon for ε-greedy.",
    )
    parser.add_argument(
        "--eps-final",
        type=float,
        default=0.05,
        help="Final epsilon for ε-greedy.",
    )
    parser.add_argument(
        "--eps-decay-frames",
        type=int,
        default=200_000,
        help="Number of steps over which to linearly decay epsilon.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("."),
        help="Directory to save final .pt files.",
    )
    parser.add_argument(
        "--per-alpha",
        type=float,
        default=0.6,
        help="Prioritization exponent alpha.",
    )
    parser.add_argument(
        "--per-beta-start",
        type=float,
        default=0.4,
        help="Initial importance-sampling exponent beta.",
    )
    parser.add_argument(
        "--per-beta-frames",
        type=int,
        default=200_000,
        help="Frames over which to anneal beta to 1.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()
    if args.render and not CV2_AVAILABLE:
        print("WARNING: --render was requested but OpenCV is not available; ignoring.")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build a temp env just to get action space size
    temp_env = PacmanEnv(args.layouts[0])
    n_actions = temp_env.action_space.n
    temp_env.close()

    # Canonical observation shape for training (we always resize to 84x84x3)
    obs_shape = (FRAME_SIZE, FRAME_SIZE, 3)

    agent = DQNAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learn_start=args.learn_start,
        target_sync_interval=1000,
        dueling=True,
        resize_to=(FRAME_SIZE, FRAME_SIZE),
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_frames=args.per_beta_frames,
        n_step=args.n_step,
        max_grad_norm=10.0,
        noisy=True,  # enable NoisyNet by default
    )

    print("Using device:", DEVICE)
    print("Training layouts:", args.layouts)
    print("Obs shape (training):", obs_shape)
    print("Actions:", n_actions)

    global_step = 0

    layouts = args.layouts
    episodes_total = args.episodes_per_layout
    rounds = 4  # e.g., 4 cycles
    episodes_per_round = episodes_total // rounds

    # Curriculum over layouts
    for r in range(rounds):
        print(f"\n=== Curriculum Round {r+1}/{rounds} ===")
        for layout in layouts:
            # Train on each layout for a fraction of episodes, then switch
            train_on_layout(layout, agent,
                            episodes=episodes_per_round,
                            max_steps=args.max_steps,
                            global_step_start=global_step,
                            eps_start=args.eps_start,
                            eps_final=args.eps_final,
                            eps_decay_frames=args.eps_decay_frames,
                            print_every=50)

    # Save the final model under all expected filenames
    args.save_dir.mkdir(parents=True, exist_ok=True)
    for layout in ["classic", "empty", "spiral", "spiral_harder"]:
        out_path = args.save_dir / f"pacman_dqn_{layout}.pt"
        agent.save(str(out_path))
        print(f"Saved weights to {out_path.resolve()}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
