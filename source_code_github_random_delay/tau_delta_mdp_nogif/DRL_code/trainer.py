from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from datetime import timedelta
import numpy as np
import torch
import pandas as pd

class Trainer:

    def __init__(self, env, env_test, algo, seed=0, num_steps=10**6, eval_interval=10**4, num_eval_episodes=1):

        self.env = env
        self.env_test = env_test
        self.algo = algo

        self.returns = {'step': [], 'return': [], 'success_rate':[]}

        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
       
        self.start_time = time()

        t = 0

        state = self.env.reset()

        for steps in range(1, self.num_steps + 1): # num_steps = 6 * 10 ** (5)
            
            state, t = self.algo.step(self.env, state, t, steps)

            if self.algo.is_update(steps):
                self.algo.update()

            # Evaluate the learned policy each eval_interval
            if steps % self.eval_interval == 0:
                self.evaluate(steps)
        #self.save_gif() # save gif for final policy
    
    # def save_gif(self):
    #     images = []
    #     state = self.env_test.reset()
    #     done = False

    #     while(not done):
    #         images.append(self.env_test.render(mode='rgb_array'))
    #         action = self.algo.exploit(state)
    #         state, reward, done, _ = self.env_test.step(action)
    #     self.display_video(images)

    # def display_video(self, frames):
    #     plt.figure(figsize=(8, 8), dpi=50)
    #     patch = plt.imshow(frames[0])
    #     plt.axis('off')

    #     def animate(i):
    #         patch.set_data(frames[i])

    #     anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    #     anim.save('env.gif', writer='PillowWriter')  

    def evaluate(self, steps):

        returns = []
        evaluates = []
        GAMMA = 0.99

        for _ in range(self.num_eval_episodes): # 100 episodes for policy evaluation
            evaluate_val = 1.0 
            state = self.env_test.reset()
            eval_temp = self.env_test.evaluate_stl_formula() # Return 1.0.
            evaluate_val = min(evaluate_val, eval_temp) # \Phi = G\phi
            done = False
            episode_return = 0.0
            gamma_count = 0

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                eval_temp = self.env_test.evaluate_stl_formula() # Return 0.0, if the past state sequence does not satisfy the STL specification at the time after k=tau-1.
                evaluate_val = min(evaluate_val, eval_temp) # \Phi = G\phi
                episode_return += (GAMMA**(gamma_count)) * reward
                gamma_count += 1

            evaluates.append(evaluate_val)
            returns.append(episode_return)

        mean_return = np.mean(returns)
        success_rate = np.mean(evaluates)
        self.returns['step'].append(steps)
        self.returns['return'].append(mean_return)
        self.returns['success_rate'].append(success_rate)

        print(f'Num steps: {steps:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Success Rate: {success_rate:<5.2f}   '
              f'Time: {self.time}')
        if steps % 100000 == 0:    
            self.algo.backup_model(steps)

    def plot(self):

        datasets = pd.DataFrame(self.returns['return'])
        datasets.to_csv('returns.csv', mode='w')
        datasets = pd.DataFrame(self.returns['success_rate'])
        datasets.to_csv('success.csv', mode='w')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))

class Algorithm(ABC):

    @abstractmethod
    def explore(self, state):
        pass

    @abstractmethod
    def exploit(self, state):
        pass

    @abstractmethod
    def is_update(self, steps):
        pass

    @abstractmethod
    def step(self, env, state, t, steps):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def backup_model(self, steps):
        pass