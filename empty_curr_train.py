# train_dqn_with_tabular.py
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tabular_q_learning import TabularQLearning  # tabular agent file
from pacman_env import PacmanEnv                 # environment file
from dqn_agent import DQN            # ResNet DQN class
import argparse
"""
Pretrain a DQN using targets from a Tabular Q-learning agent.
The tabular agent collects Q-values for states in the environment,
which are then used to train the DQN in a supervised manner.
To run: 
python train_dqn_with_tabular_s1.py --layout classic --episodes 25 --epochs 10 --batch_size 64
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TabularQDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obs, q_vals = self.data[idx]
        obs = torch.tensor(obs, dtype=torch.uint8)  # Shape (H, W, C)
        q_vals = torch.tensor(q_vals, dtype=torch.float32)
        return obs, q_vals

def collect_tabular_dataset(layout: str = "empty", episodes: int = 500, max_steps=1000):
    print(f"Collecting tabular dataset for layout '{layout}' with {episodes} episodes...")
    tab_agent = TabularQLearning(layout)
    env = PacmanEnv(layout)
    dataset = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            tab_state = tab_agent._get_state(env)
            q_vals = [tab_agent.q_table[tab_state][a] for a in range(tab_agent.n_actions)]
            dataset.append((obs, q_vals))
            action = tab_agent.select_action(tab_state)
            obs, reward, done, _, _ = env.step(action)
            steps += 1
    env.close()
    print(f"Collected {len(dataset)} samples.")
    return dataset

def train_dqn_with_tabular(dataset, obs_shape, n_actions, epochs=10, batch_size=64):
    print("Starting DQN pretraining with tabular Q dataset...")
    dqn_model = DQN(obs_shape, n_actions).to(DEVICE)
    loader = DataLoader(TabularQDataset(dataset), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(dqn_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    dqn_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for obs_batch, q_batch in loader:
            obs_batch = obs_batch.to(DEVICE)
            q_batch = q_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = dqn_model(obs_batch)
            loss = criterion(outputs, q_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * obs_batch.size(0)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(dataset):.6f}")

    saved_path = f"pretrained_dqn_{args.layout}.pth"
    torch.save(dqn_model.state_dict(), saved_path)
    print(f"Pretrained model saved to {saved_path}")

    return saved_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain DQN with Tabular Q-learning Targets")
    parser.add_argument("--layout", type=str, default="classic", help="Pacman layout to train on")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes to collect tabular data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for training")
    args = parser.parse_args()

    tabular_dataset = collect_tabular_dataset(args.layout, args.episodes)

    # Assuming obs_shape and n_actions from env (you may adjust for your env)
    env_init = PacmanEnv(args.layout)
    obs_shape = env_init.observation_space.shape
    n_actions = env_init.action_space.n
    env_init.close()

    train_dqn_with_tabular(tabular_dataset, obs_shape, n_actions, args.epochs, args.batch_size)