# Deep-Q-Networks-for-Pac-Man
EC418 Final Project: DQN for Pac-Man using tabular Q learning target method along with a ResNet

## Why This Project Matters

This project goes beyond achieving performance improvements in a reinforcement
learning benchmark. Its primary contribution is demonstrating how **reproducible
experimentation, systematic ablation, and curriculum design** can dramatically
affect stability and learning outcomes in deep RL systems. Rather than taking a results oriented approach
this proves a strong process oriented understanding.

Rather than treating training instability as a hyperparameter issue, this work
investigates why and how different architectural and algorithmic choices
succeed or fail under sparse rewards and complex dynamics. 

The methods used in this project generalize to real-world ML systems
where instability, overfitting, and misleading evaluation metrics can derail
otherwise promising approaches.


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

FINAL RESULTS:
Wins: 0/10 (0.0%)
Layout: spiral_harder
Model device: cuda

# Staged Multi-layout:
- Pretraining with classic gave better results but I belive increasing robustness by staging a brief multi-layout training staged after will give better results. Non-staged multi-layout training seemed to be too broad as on average pacman would live for many more steps but not win.
python train_dqn_tabular_staged.py --fast

- To make sure we have a secure baseline I ran 

python train_dqn_with_tabular_s1.py --layout classic --episodes 25 --epochs 10 --batch_size 64

FINAL RESULTS:
Wins: 66/100 (66.0%)
Layout: spiral_harder
Model device: cuda

- Now for staged

python train_dqn_tabular_staged.py --fast

FINAL RESULTS:
Wins: 58/100 (58.0%)
Layout: spiral_harder
Model device: cuda

## Best overall performance (win%)

layout: classic, spiral, spiral_harder, empty, 
staged: ,,58,
classic:
spriral:
spiral_h:
empty: 

# Prioritzed Replay Memory
For fast paced progress we test on spiral_harder as its quick to train. 
Once better results are reached we fine tune on classic
!python spiral_curr_train.py --episodes 50 --epochs 10 --batch_size 32

FINAL RESULTS:
Wins: 20/50 (40.0%)
Layout: spiral_harder
Model device: cuda

# Double DQN 
To solve for overestimation of Q values and increase wins by not making model overconfident

FINAL RESULTS:
Wins: 27/50 (54.0%)
Layout: spiral_harder
Model device: cuda

# Distributional Q learning
Update theta to match distribution of actual play

FINAL RESULTS:
Wins: 0/50 (0.0%)
Layout: spiral_harder
Model device: cuda
Expand distributional hyperparameters

# Noisy Net Small e-Greedy
FINAL RESULTS:
Wins: 59/100 (59.0%)
Layout: spiral_harder
Model device: cuda

# Remove e-Greedy
FINAL RESULTS:
Wins: 9/100 (9.0%)
Layout: spiral_harder
Model device: cuda

# Noisey net small e-Greedy (patched)
FINAL RESULTS:
Wins: 32/100 (32.0%)
Layout: spiral_harder
Model device: cuda

I assume model must overfit and we still have small memory_cap

# Increase memory cap drop 1 Resnet block
drop a resnet and boost memory cap from 20000 to 200000. 

FINAL RESULTS:
Wins: 17/100 (17.0%)
Layout: spiral_harder
Model device: cuda


