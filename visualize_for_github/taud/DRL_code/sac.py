import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

import replay_memory
import network
import trainer

class SAC(trainer.Algorithm):

    def __init__(self, state_shape, action_shape, device=torch.device('cuda'), seed=0,
                batch_size=256, gamma=0.99, lr_actor=3e-4, lr_critic=3e-4, lr_entropy=3e-4,
                replay_size=10**6, start_steps=10**4, tau=0.01, alpha=1.0, reward_scale=1.0, auto_coef=False):
        super().__init__()


        # Initialize Replay Buffer
        self.buffer = replay_memory.ReplayBuffer(
            buffer_size=replay_size,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
        )

        # Construct DNNs for SAC．
        self.actor = network.SACActor(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.critic = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device)
        self.critic_target = network.SACCritic(
            state_shape=state_shape,
            action_shape=action_shape
        ).to(device).eval()

        # auto entropy coef adjusting ?
        self.auto_coef = auto_coef

        self.alpha = alpha # entropy coef
        if self.auto_coef: # adjust coef is True
            self.target_entropy = -torch.prod(torch.Tensor(action_shape).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        
        # Initialize Target Network
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Define Optimizer
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        if self.auto_coef:
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr_entropy) # 3e-4
        
        # other parameters
        self.learning_steps = 0
        self.batch_size = batch_size
        self.device = device
        self.gamma = gamma
        self.start_steps = start_steps
        self.tau = tau
        self.reward_scale = reward_scale

    def explore(self, state):
        """ Returns the logarithm log(pi(a|s)) of the stochastic action and the probability density of that action. """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()
    
    def exploit(self, state):
        """ Returns deterministic action. """
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def is_update(self, steps):
        # Learning is not performed for a certain period (start_steps) at the beginning of learning.
        return steps >= max(self.start_steps, self.batch_size)

    def step(self, env, state, t, steps): 
        t += 1

        if steps <= self.start_steps:
            action = env.action_space.sample()
        else:
            action, _ = self.explore(state)
        next_state, reward, done, _ = env.step(action)

        if t == env._max_episode_steps:
            done_masked = False
        else:
            done_masked = done

        # Add the experience to the replay buffer.
        self.buffer.append(state, action, reward, done_masked, next_state)

        # At the end of an episode, reset the environment.
        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def update(self):
        self.learning_steps += 1
        states, actions, rewards, dones, next_states = self.buffer.sample(self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        if self.auto_coef:
            self.update_entropy_coef(states)
        self.update_target()    
    
    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + (1.0 - dones) * self.gamma * next_qs

        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()

        self.optim_critic.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.optim_critic.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

    def update_entropy_coef(self, states):
        _, log_pis = self.actor.sample(states)

        alpha_loss = -(self.log_alpha * (log_pis + self.target_entropy).detach()).mean()

        self.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.optim_alpha.step()

        self.alpha = self.log_alpha.exp()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
    
    def backup_model(self, steps):
        torch.save(self.actor.state_dict(), 'SAC_STL_actor_' + str(steps) + '.pth')
        torch.save(self.critic.state_dict(), 'SAC_STL_critic_' + str(steps) + '.pth')

        






