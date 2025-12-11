# dqn_agent.py  –  network, replay buffer, ε‑greedy, optimiser
from __future__ import annotations
import math, random
from collections import deque
from typing import Tuple, List
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from typing import Tuple    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# Distributional RL (C51-style) hyperparameters
V_MIN     = -1000.0   # lower support bound (tune as needed)
V_MAX     =  1000.0   # upper support bound (tune as needed)
N_ATOMS   = 51      # number of atoms
DELTA_Z   = (V_MAX - V_MIN) / (N_ATOMS - 1)
"""


# ------------- Residual Block (optional) -------------
class ResidualBlock(nn.Module):
    """
    A standard Residual Block as used in ResNet architectures.
    It consists of two convolutional layers with a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        skip = self.skip_connection(x)
        out += skip
        return F.relu(out)
    
# --------------- noise network block ---------------
class NoisyLinear(nn.Module):
    """
    Factorized Noisy Linear layer (Fortunato et al., 2017).
    Used to replace standard Linear layers for exploration.
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learnable parameters: mu and sigma for weights and biases
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Non-learnable buffers for noise
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # Initialization
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init: float):
        # Recommended initialization from the paper
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()  # f(x) = sign(x) * sqrt(|x|)

    def reset_noise(self):
        # Factorized: eps_w = f(eps_in) ⊗ f(eps_out)
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.unsqueeze(1) * eps_in.unsqueeze(0))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # At eval time, use deterministic weights (no noise)
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)

# ──────────────── network ────────────────
class DQN(nn.Module):
    """
    This class defines the Deep Q-Network (DQN) architecture.
    It processes input observations (images) and outputs Q-values for each action.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        # TODO: Implement network architecture
        self.n_actions = n_actions
        C = obs_shape[2]  # extract channels
        H = obs_shape[0]  # extract height
        W = obs_shape[1]  # extract width
        # Define convolutional layers
        # DQN with Residual Blocks
        self.net = nn.Sequential(
            # Initial Conv Layer
            nn.Conv2d(C, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Residual Blocks
        self.Res1 = ResidualBlock(64, 64, stride=2)
        #self.Res2 = ResidualBlock(64, 64, stride=1)
        #self.Res3 = ResidualBlock(128, 256, stride=1)
        #self.Res4 = ResidualBlock(256, 512, stride=1)
        #self.Res5 = ResidualBlock(512, 512, stride=1)
        #self.Res6 = ResidualBlock(512, 512, stride=1)
        # Adaptive Pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Flatten layer will be applied after avgpool in forward pass
        self.flatten = nn.Flatten()

        # Default DQN 
        """self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        """
        # Calculate dimension after conv layers using a dummy input
        dummy_input = torch.zeros(1, C, H, W)
        conv_output = self.net(dummy_input)            
        dim = conv_output.shape[1]  # Adjust based on final conv output size
        

        # Noisy head instead of plain Linear
        self.fc1 = NoisyLinear(64, 512)
        self.fc2 = NoisyLinear(512, n_actions)

    def reset_noise(self):
        """
        Resample noise for all noisy layers.
        Call this once per step (for policy net) and occasionally for target net.
        """
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        Input: x - tensor of shape (B, H, W, C) with pixel values [0, 255]
        Output: tensor of shape (B, n_actions) with Q-values for each action.       
        """        
        # Forward pass through the network
        x = x.float() / 255.0  # Convert to float + normalize to [0, 1]
        x = x.permute(0, 3, 1, 2)  # Rearrange to (B, C, H, W)
        x = self.net(x)
        x = self.Res1(x)
        #x = self.Res2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

# ──────────────── target network ────────────────
class tDQN(nn.Module):
    """
    This class defines the target Deep Q-Network (tDQN) architecture. NO NOISY LAYERS.
    It processes input observations (images) and outputs Q-values for each action.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], n_actions: int):
        super().__init__()
        # TODO: Implement network architecture
        self.n_actions = n_actions
        C = obs_shape[2]  # extract channels
        H = obs_shape[0]  # extract height
        W = obs_shape[1]  # extract width
        # Define convolutional layers
        # DQN with Residual Blocks
        self.net = nn.Sequential(
            # Initial Conv Layer
            nn.Conv2d(C, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Residual Blocks
        self.Res1 = ResidualBlock(64, 64, stride=2)
        #self.Res2 = ResidualBlock(64, 64, stride=1)
        #self.Res3 = ResidualBlock(128, 256, stride=1)
        #self.Res4 = ResidualBlock(256, 512, stride=1)
        #self.Res5 = ResidualBlock(512, 512, stride=1)
        #self.Res6 = ResidualBlock(512, 512, stride=1)
        # Adaptive Pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Flatten layer will be applied after avgpool in forward pass
        self.flatten = nn.Flatten()

        # Default DQN 
        """self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        """
        # Calculate dimension after conv layers using a dummy input
        dummy_input = torch.zeros(1, C, H, W)
        conv_output = self.net(dummy_input)            
        dim = conv_output.shape[1]  # Adjust based on final conv output size
        

        # Linear head instead of Noisy Linear
        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, n_actions)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.
        Input: x - tensor of shape (B, H, W, C) with pixel values [0, 255]
        Output: tensor of shape (B, n_actions) with Q-values for each action.       
        """        
        # Forward pass through the network
        x = x.float() / 255.0  # Convert to float + normalize to [0, 1]
        x = x.permute(0, 3, 1, 2)  # Rearrange to (B, C, H, W)
        x = self.net(x)
        x = self.Res1(x)
        #x = self.Res2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x
# ───────────── replay buffer ─────────────
class PrioritizedReplayMemory:
    """
    Proportional prioritized replay buffer.
    Each transition has a priority p_i and is sampled with probability:
        P(i) = p_i^alpha / sum_j p_j^alpha
    Importance sampling weights are returned to correct the bias.
    """
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def push(self, *transition):
        """
        transition = (state, action, reward, next_state, done)
        New transitions get max priority so they are sampled at least once.
        """
        max_prio = max(self.priorities) if self.priorities else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a minibatch of transitions with priorities.
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities[:len(self.buffer)])

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, new_priorities):
        for idx, prio in zip(indices, new_priorities):
            self.priorities[idx] = float(prio)


    # ───────────── ε‑greedy & optimise ───────


def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start : float = 0.1, eps_end: float = 0.01, eps_decay: int = 8000) -> int:
    """
    This function selects an action using an ε-greedy policy. 
    with probability ε, a random action is chosen (exploration),
    and with probability 1-ε, the action with the highest Q-value is chosen (exploitation) -
    also supports NoisyNet exploration by resampling noise for each decision.
    """
    
    # Calculate current epsilon using decay formula
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    if random.random() < eps:
        return random.randrange(net.n_actions)
    
    # NoisyNet exploration: resample noise for this decision
    net.reset_noise()

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q_values = net(state_tensor)
        return int(q_values.argmax().item())

    
 
def optimise(memory: PrioritizedReplayMemory,
             policy: DQN,
             target: tDQN,
             optimiser: optim.Optimizer,
             batch_size: int,
             gamma: float,
             beta: float = 0.4,
             prio_eps: float = 1e-5):
    """
    One optimisation step of DQN with prioritized replay.

    Double DQN target:
    y = r + (1 - d) * gamma * Q_target(s', argmax_a Q_policy(s', a))

    - Sample transitions from PrioritizedReplayMemory with probabilities P(i).
    - Compute importance sampling weights w_i.
    - Compute standard DQN TD targets with the target network.
    - Update priorities p_i <- |δ_i|.
    """
    # Reset noise for NoisyNet layers
    #policy.reset_noise()    


    if len(memory) < batch_size:
        return

    # Sample batch with priorities and IS weights
    (states, actions, rewards, next_states, dones,
     indices, weights) = memory.sample(batch_size, beta=beta)

    states      = torch.as_tensor(states, device=DEVICE, dtype=torch.float32)
    actions     = torch.as_tensor(actions, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards     = torch.as_tensor(rewards, device=DEVICE, dtype=torch.float32)
    next_states = torch.as_tensor(next_states, device=DEVICE, dtype=torch.float32)
    dones       = torch.as_tensor(dones, device=DEVICE, dtype=torch.float32)
    weights_t   = torch.as_tensor(weights, device=DEVICE, dtype=torch.float32).unsqueeze(1)

    # Q(s,a) from current policy
   
    q_values = policy(states).gather(1, actions)

    # ---------- Double DQN target ----------
    with torch.no_grad():
   
        # 1) policy net chooses next actions
        next_q_policy = policy(next_states)                    # [B, |A|]
        next_actions  = next_q_policy.argmax(dim=1, keepdim=True)  # [B, 1]

        # 2) target net evaluates those actions
        next_q_target = target(next_states)                    # [B, |A|]
        max_next_q_values = next_q_target.gather(1, next_actions).squeeze(1)  # [B]

        # 3) TD target
        target_values = rewards + (1.0 - dones) * gamma * max_next_q_values  # [B]
    
    target_values = target_values.unsqueeze(1)  # [B,1]

    # TD error
    td_errors = target_values - q_values  # [B,1]

    # Update priorities: |δ| + eps
    new_priorities = td_errors.detach().abs().cpu().numpy().flatten() + prio_eps
    memory.update_priorities(indices, new_priorities)

    # Weighted MSE loss
    loss = (weights_t * (td_errors ** 2)).mean()

    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()

