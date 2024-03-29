# Soft actor critic for satisfying STL specifications with network delays (Ver. 2 and 3)

This repository includes source codes for https://arxiv.org/abs/2108.01317 (Ver. 2 and 3). 

![env](https://user-images.githubusercontent.com/68591842/156919636-ff35054a-57af-4478-b623-e5e2e5fa78f6.gif)

## Python Version
```
Python 3.6.9
Cuda 10.2
Pytorch 1.7.1
Torch Audio 0.7.2
Torch Vision 0.8.2
gym 0.19.0

```

## PC
```
CPU: AMD(R) Ryzen 9 3950X
Main Memory: DDR4-2666 16GB 2
Motherboard: ASUS PRIME X570-PRO
GPU: NVIDIA(R) GeForce RTX 2070 SUPER
OS: ubuntu 18.04
```

## Difference from Ver. 1
- We consider a discrete-time system.
- We consider a stochastic control system (Adding a noise term).
- We change the environment of the example.

## typo
- Eq. (9) k_s^i -> t_s^i (ver. 2 -> ver. 3)
- In section III, we revised as follows: p_f(x')=|\Delta_{w}^{-1}|p_{w}(\Delta_{w}^{-1}(x'-f(x,u))) -> p_f(x')=|\Delta_{w}^{-1}|p_{w}(\Delta_{w}^{-1}(x'-f(x,u))) (ver. 2 -> ver. 3)
- z[i] -> x^{\tau}[i] in preproccessing.

## Result Data
- tau_preprocess: tau-MDP (without previous actions) with preprocessing
- tau_delta_nopreprocess: tau d-MDP (with previous actions) without preprocessing
- tau_delta_preprocess (proposed method): tau d-MDP (with previous actions) without preprocessing

![returns_tau_d](https://user-images.githubusercontent.com/68591842/156919395-1cb3df9c-d8d5-4188-a1f4-85da6dab6f6e.png)
![success_rates_tau_d](https://user-images.githubusercontent.com/68591842/156919433-1f7e21ed-6ebe-4a2b-a684-f6e653ca256e.png)
![returns_preprocess](https://user-images.githubusercontent.com/68591842/156919461-416a68d2-fcfe-487c-a84a-9f1971092382.png)
![success_rates_preprocess](https://user-images.githubusercontent.com/68591842/156919486-1d0b83ba-aa0c-45bd-bb4d-5ec1bb7c4352.png)

## Note
In this study, we use the SAC algorithm owing to good sample efficiency and asymptotic performance. In general, we can apply any off-policy DRL algorithm with Experience Replay such as DDPG and TD3.  
