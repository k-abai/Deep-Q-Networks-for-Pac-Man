# Deep-Q-Networks-for-Pac-Man
EC418 Final Project: DQN for Pac-Man using tabular Q learning target method along with a ResNet

# Default: 
- Simple DQN utilizing a single CNN

FINAL RESULTS:
2/10 wins (20.0%)
Layout: spiral_harder
Model device: cpu

# ResNet: 
- Allows us to have a deeper model that trains more efficiently; Essentially having less parameters. This architecture skip happens when number of channels or resolution changes. This promotes better feature reuse, allows learning of "deeper" sentimental features in gameplay thus improving decision making.

FINAL RESULTS:
Wins: 1/10 (10.0%) #simplified resnet
Layout: spiral_harder
Model device: cuda

# Pre-trained ResNet: 
- Utilizes tabular q learning as a target to initialize weights in ResNet. Allowing for better convergence. Additional, learning rate adapted with a cosine scheduler.
python train_dqn_with_tabular.py --layout classic --episodes 25 --epochs 10 --batch_size 64

FINAL RESULTS:
Wins: 3/10 (30.0%)
Layout: spiral_harder
Model device: cuda

# Multi-layout pre-trained ResNet:
- Original version used trained only on a classic layout. Training over a randomized selection of layouts will increase robustness in model initialization and prevent overfitting, thus improving generalized ResNet DQN performance.
python train_dqn_with_tabular.py --episodes 25 --epochs 10 --batch_size 64

