#!/usr/bin/env python
"""
play_pacman.py
-----------------

Animate a trained DQN Pac‑Man agent with OpenCV.

Example:
    python play_pacman.py --layout spiral --episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Re‑use env, network and DEVICE from your training module
from train_dqn import PacmanEnv, DQN, DEVICE


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_policy(model_path: Path, num_actions: int) -> DQN:
    """Instantiate a DQN and load weights."""
    policy = DQN(num_actions).to(DEVICE)
    policy.load_state_dict(torch.load(model_path, map_location=DEVICE))
    policy.eval()
    return policy


def play(
    layout: str,
    policy: DQN,
    episodes: int = 10,
    frame_delay_ms: int = 100,
) -> None:
    """Run `episodes` games on the chosen layout, animating with OpenCV."""
    env = PacmanEnv(layout=layout)
    wins = 0

    cv2.namedWindow("Pac‑Man  (press q to quit)", cv2.WINDOW_AUTOSIZE)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        steps = 0
        won = False

        while not done and steps < 500:
            # Greedy action
            with torch.no_grad():
                st = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                action = int(policy(st).argmax(1).item())

            # Environment step
            state, _, done, _, _ = env.step(action)
            steps += 1
            if done and not env.pellets:      # no pellets left → win
                won = True

            # Render RGB frame → BGR for OpenCV
            frame_rgb = env.render(mode="rgb_array")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # HUD text
            hud = f"Layout: {layout}   Ep {ep}/{episodes}   Step {steps}"
            cv2.putText(
                frame_bgr,
                hud,
                org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow("Pac‑Man  (press q to quit)", frame_bgr)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == ord("q"):
                print("▶ Quit requested — exiting.")
                env.close()
                cv2.destroyAllWindows()
                sys.exit(0)

        wins += won
        print(f"Episode {ep:2d}: {'WIN' if won else 'LOSE'} in {steps} steps")

    print(f"\nFinished {episodes} episodes on '{layout}': "
          f"{wins} win(s)  ({wins/episodes*100:.1f} %)")

    env.close()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Play trained Pac‑Man with animation")
    parser.add_argument(
        "--layout",
        choices=["classic","empty", "spiral", "spiral_harder"],
        default="empty",
        help="which board layout to use",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="path to .pt weight file "
             "(default: pacman_dqn_<layout>.pt in CWD)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="games to play")
    parser.add_argument(
        "--speed",
        type=int,
        default=100,
        metavar="MS",
        help="delay between frames in milliseconds (lower = faster)",
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model or Path(f"pacman_dqn_{args.layout}.pt")
    if not model_path.is_file():
        print(f"Error: weights not found at {model_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # Build a temp env to get action‑space size
    tmp_env = PacmanEnv(layout=args.layout)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    policy = load_policy(model_path, n_actions)
    play(layout=args.layout,
         policy=policy,
         episodes=args.episodes,
         frame_delay_ms=args.speed)


if __name__ == "__main__":
    main()
