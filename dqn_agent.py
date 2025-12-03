#!/usr/bin/env python
"""
dqn_agent.py

DQN network + agent for visual Pac-Man.

Exposes:
    - DEVICE: torch.device (cuda / cpu)
    - DQN:    convolutional dueling Q-network (optionally with NoisyNet layers)
    - DQNAgent: training wrapper with:
        * Double DQN
        * Prioritized Experience Replay
        * Optional n-step returns
        * Optional NoisyNet exploration
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ───────────────────────── device ─────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────── Noisy layer ────────────────────
class NoisyLinear(nn.Module):
    """
    Factorised Gaussian NoisyNet linear layer (Fortunato et al.).
    Noise is sampled via explicit reset_noise() calls, NOT inside forward.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 * mu_range)

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 * mu_range)

    def _scale_noise(self, size: int) -> torch.Tensor:
        # No grad, so in-place is safe here
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: do NOT call reset_noise() here.
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)



# ───────────────────────── network ────────────────────────
class DQN(nn.Module):
    """
    Convolutional Dueling DQN network that works directly on image observations.

    * Accepts inputs in HWC (gym-style) or CHW format, with optional batch dimension.
    * Internally resizes all inputs to `resize_to` (default 84×84) so a single
      set of weights can generalise across different board sizes.
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        dueling: bool = True,
        resize_to: Tuple[int, int] = (84, 84),
        noisy: bool = True,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.dueling = dueling
        self.resize_to = resize_to
        self.noisy = noisy
        # Determine input channels from obs_shape
        if obs_shape is None:
            in_channels = 3
        elif len(obs_shape) == 3:
            # (H, W, C) – common for Gym image observations
            in_channels = obs_shape[2]
        elif len(obs_shape) == 1:
            # already channels-only
            in_channels = obs_shape[0]
        elif len(obs_shape) == 2:
            # (H, W) – assume grayscale
            in_channels = 1
        else:
            # fallback: last dimension
            in_channels = obs_shape[-1]

        self.in_channels = in_channels

        # Convolutional feature extractor (Atari-style + one residual block)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Extra conv with residual connection (keeps H,W due to padding)
        self.extra_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Compute conv output size using a dummy forward
        dummy_input = torch.zeros(1, in_channels, resize_to[0], resize_to[1])
        with torch.no_grad():
            conv_out = self._forward_conv(dummy_input)
        conv_out_size = conv_out.reshape(1, -1).size(1)
        self.conv_out_size = conv_out_size

        Linear = NoisyLinear if noisy else nn.Linear

        if dueling:
            self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            Linear(256, 1),
            )
            self.fc_advantage = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.ReLU(),
                Linear(256, n_actions),
            )

        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(inplace=True),
                Linear(512, n_actions),
            )
        
    def reset_noise(self):
        """
        Reset noise for all NoisyLinear layers in this network.
        Call this once per action selection / update.
        """
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
    
    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv backbone + residual conv.
        """
        features = self.conv(x)
        features = F.relu(self.extra_conv(features) + features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Accepts:
            * (H, W, C)
            * (B, H, W, C)
            * (B, C, H, W)

        Returns:
            Q-values of shape (B, n_actions)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"expected input dim 3 or 4, got {x.shape}")

        # Move to float and normalise from [0,255] to [0,1] if needed
        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        B = x.shape[0]

        # Detect channel position.
        # If second dim matches in_channels (1,3,4) assume (B, C, H, W),
        # otherwise assume (B, H, W, C).
        if x.shape[1] in (1, 3, 4) and x.shape[1] == self.in_channels:
            x_chw = x
        else:
            x_chw = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        # Resize to canonical size if needed
        if self.resize_to is not None:
            h, w = x_chw.shape[2], x_chw.shape[3]
            if (h, w) != self.resize_to:
                x_chw = F.interpolate(
                    x_chw,
                    size=self.resize_to,
                    mode="bilinear",
                    align_corners=False,
                )

        features = self._forward_conv(x_chw)
        flat = features.reshape(B, -1)

        if self.dueling:
            value = self.fc_value(flat)  # (B, 1)
            advantage = self.fc_advantage(flat)  # (B, n_actions)
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            q_values = value + (advantage - advantage_mean)
        else:
            q_values = self.fc(flat)

        return q_values


# ───────────────────────── replay buffer ────────────────────────
class PrioritizedReplayBuffer:
    """
    Simple proportional prioritized replay buffer.

    Sampling uses probabilities p_i^alpha / sum_j p_j^alpha.
    Importance-sampling (IS) weights are returned to correct the bias.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.alpha = alpha

        self.pos = 0
        self.size = 0

        self.obs_shape = tuple(obs_shape)

        # Store raw uint8 frames to save memory; network normalises internally.
        self.states = np.zeros((capacity, *self.obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *self.obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done):
        idx = self.pos

        # Convert to uint8 numpy arrays if needed
        self.states[idx] = np.asarray(state, dtype=np.uint8)
        self.next_states[idx] = np.asarray(next_state, dtype=np.uint8)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = bool(done)

        # New experience gets max priority so it will be sampled at least once
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[idx] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        assert self.size > 0, "Cannot sample from an empty buffer"
        assert beta >= 0.0

        prios = self.priorities[: self.size]
        # Small epsilon to avoid zero probability
        prios = np.where(prios > 0, prios, 1e-6)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)

        states = self.states[indices]
        next_states = self.next_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        # Importance-sampling weights
        N = self.size
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        )

    def update_priorities(self, indices, new_priorities):
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = max(prio, 1e-6)


# ───────────────────────── agent ────────────────────────
class DQNAgent:
    """
    DQN agent with:
      * Dueling network
      * Double DQN target computation
      * Prioritized experience replay
      * Optional n-step returns
      * Optional NoisyNet exploration (applied to fully-connected layers)

    This class *doesn't* own the environment – training code should:
      * interact with the env
      * call `store_transition` each step
      * call `update` periodically
    """

    def __init__(
        self,
        obs_shape,
        n_actions: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        learn_start: int = 10_000,
        target_sync_interval: int = 1000,
        dueling: bool = True,
        resize_to: Tuple[int, int] = (84, 84),
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 200_000,
        n_step: int = 5,
        max_grad_norm: float = 10.0,
        noisy: bool = True,
        distributional=False
    ):
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_sync_interval = target_sync_interval
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        self.max_grad_norm = max_grad_norm
        self.n_step = max(1, int(n_step))
        self.gamma_n = gamma ** self.n_step
        self.noisy = noisy

        # Online & target networks
        self.online_net = DQN(obs_shape, n_actions, dueling=dueling, 
                               noisy=noisy, **({"distributional": self.atoms} if distributional else {}))
        self.target_net = DQN(obs_shape, n_actions, dueling=dueling, 
                               noisy=noisy, **({"distributional": self.atoms} if distributional else {}))
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # Prioritized replay buffer
        self.replay = PrioritizedReplayBuffer(
            capacity=buffer_size,
            obs_shape=self.obs_shape,
            alpha=per_alpha,
        )

        # For n-step returns we keep a short FIFO buffer of recent transitions
        self._n_step_buffer = deque(maxlen=self.n_step)

        # Training counters
        self.num_updates = 0
        self.num_steps = 0
        self.beta = per_beta_start

    # ─────────────── action selection ───────────────
    def act(self, state, epsilon: float = 0.0) -> int:
        """
        Epsilon-greedy action selection.
        For exploration we use BOTH ε-greedy and NoisyNet noise.
        """
        # sample fresh noise for this decision (if noisy is enabled)
        if self.noisy:
            self.online_net.reset_noise()

        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, device=DEVICE)
            q_values = self.online_net(state_t).squeeze(0)
            action = int(q_values.argmax().item())
        return action

    # ─────────────── n-step helper ───────────────
    def _get_n_step_info(self):
        """
        Aggregate rewards over n steps from the internal n-step buffer.
        Returns (state, action, n_step_return, next_state, done_flag).
        """
        # First transition in the buffer defines (state, action)
        first = self._n_step_buffer[0]
        state, action = first["state"], first["action"]

        R = 0.0
        next_state = self._n_step_buffer[-1]["next_state"]
        done = self._n_step_buffer[-1]["done"]

        for idx, transition in enumerate(self._n_step_buffer):
            R += (self.gamma ** idx) * transition["reward"]
            if transition["done"]:
                # Once done, subsequent rewards don't matter and next_state is terminal
                done = True
                next_state = transition["next_state"]
                break

        return state, action, R, next_state, done

    def store_transition(self, state, action, reward, next_state, done: bool):
        """
        Add a transition to the replay buffer, applying n-step processing if enabled.
        """
        self.num_steps += 1

        transition = {
            "state": np.asarray(state, dtype=np.uint8),
            "action": int(action),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.uint8),
            "done": bool(done),
        }

        if self.n_step == 1:
            # Directly store 1-step transition
            self.replay.add(
                transition["state"],
                transition["action"],
                transition["reward"],
                transition["next_state"],
                transition["done"],
            )
        else:
            # n-step: push to local buffer first
            self._n_step_buffer.append(transition)
            if len(self._n_step_buffer) == self.n_step:
                s, a, R, ns, d = self._get_n_step_info()
                self.replay.add(s, a, R, ns, d)
                # remove the oldest transition and continue
                self._n_step_buffer.popleft()

            # If episode ended, flush the remaining transitions
            if done:
                while len(self._n_step_buffer) > 0:
                    s, a, R, ns, d = self._get_n_step_info()
                    self.replay.add(s, a, R, ns, d)
                    self._n_step_buffer.popleft()

    # ─────────────── training update ───────────────
    def _update_beta(self):
        """
        Linearly anneal importance-sampling exponent beta towards 1.0
        over `per_beta_frames` update steps.
        """
        self.beta = min(
            1.0,
            self.per_beta_start
            + (1.0 - self.per_beta_start)
            * (self.num_updates / max(1, self.per_beta_frames)),
        )

    def can_update(self) -> bool:
        return len(self.replay) >= self.learn_start

    def update(self) -> Optional[dict]:
        """
        Perform a single gradient update from a prioritized replay batch.

        Returns a small metrics dict (loss, mean_Q, beta, etc.) or None if
        the replay buffer is not yet warmed up (`learn_start` not reached).
        """
        if len(self.replay) < self.learn_start:
            return None

        self.online_net.train()
        self._update_beta()

        batch_size = self.batch_size
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.replay.sample(batch_size, beta=self.beta)

        # Convert to torch tensors
        states_t = torch.as_tensor(states, device=DEVICE)  # uint8 -> handled in network
        next_states_t = torch.as_tensor(next_states, device=DEVICE)
        actions_t = torch.as_tensor(actions, device=DEVICE, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards, device=DEVICE, dtype=torch.float32)
        dones_t = torch.as_tensor(dones, device=DEVICE, dtype=torch.float32)
        weights_t = torch.as_tensor(weights, device=DEVICE, dtype=torch.float32)

        if len(self.replay) < self.learn_start:
            return None

        self.online_net.train()
        self._update_beta()
        
        # reset noisy layers for this gradient step
        if self.noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        # Current Q-values for taken actions
        q_values = self.online_net(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online_net(next_states_t)
            next_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target_net(next_states_t)
            next_q_target = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            targets = rewards_t + (1.0 - dones_t) * self.gamma_n * next_q_target

        # Huber loss with per-sample weights
        td_errors = q_values - targets
        loss_unreduced = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (loss_unreduced * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Update priorities based on absolute TD error
        new_prios = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.replay.update_priorities(indices, new_prios)

        self.num_updates += 1

        # Periodically sync target network
        if self.num_updates % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        metrics = {
            "loss": float(loss.item()),
            "mean_q": float(q_values.mean().item()),
            "beta": float(self.beta),
            "buffer_size": len(self.replay),
        }
        return metrics

    # ─────────────── persistence helpers ───────────────
    def save(self, path: str):
        """
        Save online network weights to a .pt file.
        """
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        """
        Load weights into the online network. The target network is also synced.
        """
        state_dict = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(state_dict, strict=strict)
        self.target_net.load_state_dict(self.online_net.state_dict())
