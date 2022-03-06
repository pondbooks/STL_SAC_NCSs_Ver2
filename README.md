# STL_SAC_NCSs_Ver2

This repository includes source codes for https://arxiv.org/abs/2108.01317 (Ver. 2). 

## Configuration
1. OS: ubuntu 18.04
2. Python: 3.6.9
3. Cuda: 10.2
4. PyTorch: 1.7.1
5. Torch Audio: 0.7.2
6. Torch Vision: 0.8.2

## Difference from Ver. 1
- We consider a discrete-time system.
- We consider a stochastic control system (Adding a noise term).
- We change the environment of the example.

## typo
- Eq. (9) k_s^i -> t_s^i

## Result Data
- tau_preprocess: tau-MDP (without previous actions) with preprocessing
- tau_delta_nopreprocess: tau d-MDP (with previous actions) without preprocessing
- tau_delta_preprocess (proposed method): tau d-MDP (with previous actions) without preprocessing

![returns_tau_d](https://user-images.githubusercontent.com/68591842/156919395-1cb3df9c-d8d5-4188-a1f4-85da6dab6f6e.png)
