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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Residual Blocks
        self.Res1 = ResidualBlock(64, 64, stride=2),
        self.Res2 = ResidualBlock(64, 128, stride=1),
        self.Res3 = ResidualBlock(128, 256, stride=1),
        self.Res4 = ResidualBlock(256, 512, stride=1),
        self.Res5 = ResidualBlock(512, 512, stride=1),
        self.Res6 = ResidualBlock(512, 512, stride=1),

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
        dim = conv_output.shape[1]
        
        # Connected layers from conv output to action Q-values
        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    
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
        x = self.Res2(x)
        x = self.Res3(x)
        x = self.Res4(x)
        x = self.Res5(x)
        x = self.Res6(x)
        x = self.avgpool(x) 
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ───────────── replay buffer ─────────────
class ReplayMemory:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(tuple(transition))

    def sample(self, batch_size: int):
        s = random.sample(self.buf, batch_size)
        return map(np.array, zip(*s))

    def __len__(self):
        return len(self.buf)


    # ───────────── ε‑greedy & optimise ───────
def select_action(state: np.ndarray, net: DQN, step: int,
                  eps_start: float, eps_end: float, eps_decay: int) -> int:
    """
    This function selects an action using an ε-greedy policy. 
    with probability ε, a random action is chosen (exploration),
    and with probability 1-ε, the action with the highest Q-value is chosen (exploitation).
    """
    
    # Calculate current epsilon using decay formula
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)
    if random.random() < eps:
        return random.randrange(net.n_actions)
    
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q_values = net(state_tensor)
        return int(q_values.argmax().item())
 

def optimise(memory: ReplayMemory, policy: DQN, target: DQN,
             optimiser: optim.Optimizer, batch_size: int, gamma: float):
    """
    Perform one step of DQN optimization using experience replay and target network.
    This is the CORE of the DQN algorithm!
    
    CONCEPT:
    DQN uses two key innovations:
    1. Experience Replay: Store transitions and learn from random batches
    2. Target Network: Use separate network for computing targets (more stable)
    """
    
    # Check if memory has enough experiences (return early if not)
    if len(memory) < batch_size:
        return
    # Sample batch of experiences
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    # Convert numpy arrays to torch tensors on correct device
    states = torch.as_tensor(states, device=DEVICE, dtype=torch.float32)
    actions = torch.as_tensor(actions, device=DEVICE, dtype=torch.int64).unsqueeze(1)
    rewards = torch.as_tensor(rewards, device=DEVICE, dtype=torch.float32)
    next_states = torch.as_tensor(next_states, device=DEVICE, dtype=torch.float32)
    dones = torch.as_tensor(dones, device=DEVICE, dtype=torch.float32)

    # Compute current Q-values
    q_values = policy(states).gather(1, actions)

    # Compute target Q-values
    with torch.no_grad():
        max_next_q_values = target(next_states).max(1)[0]
    target_values = rewards + (1 - dones) * gamma * max_next_q_values

    # Compute loss
    loss = nn.functional.mse_loss(q_values, target_values.unsqueeze(1))

    # Optimize policy network
    optimiser.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimiser.step()
    
