import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import copy
from collections import namedtuple
from torch.autograd import Variable
import cv2
import argparse
from tensorboardX import SummaryWriter
import shutil
import os

NUM_EPISODES = 500


HYPERPARAMS = {
        'replay_size':      50000,
        'replay_initial':   50000,
        'target_net_sync':  200,
        'epsilon_frames':   10**4,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
}


class RewardTracker:
    def __init__(self, length=20):
        self.length = length
        self.rewards = []
        self.position = 0
        self.mean_score = 0

    def add(self, reward):
        if len(self.rewards) < self.length:
            self.rewards.append(reward)
        else:
            self.rewards[self.position] = reward
            self.position = (self.position + 1) % self.length
        if len(self.rewards) < self.length:
            self.mean_score = 0
        else:
            self.mean_score = np.mean(self.rewards)

    def meanScore(self):
        return self.mean_score


class EpsilonTracker:
    def __init__(self, params):
        self.epsilon_params = params
        self._epsilon = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_delta = 1.0 * (params['epsilon_start'] - params['epsilon_final']) / params['epsilon_frames']

    def epsilon(self):
        old_epsilon = self._epsilon
        self._epsilon -= self.epsilon_delta
        return max(old_epsilon, self.epsilon_final)

    def reset_epsilon(self):
        self._epsilon = self.epsilon_params['epsilon_start']

    def percievedEpsilon(self):
        return max(self._epsilon, self.epsilon_final)



class DQN(nn.Module):
    def __init__(self, size):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(size*2, size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class ReplayMemory(object):
    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Bit flipping environment
class Env(object):
    def __init__(self, size):
        self.size = size
        self.state = np.random.randint(2, size=size)
        self.target = np.random.randint(2, size=size)
        while np.sum(self.state == self.target) == size:
            self.target = np.random.randint(2, size=size)

    # Take in a n index
    # Update the environment, return the reward
    def step(self, action):
        old_score = np.sum(self.state == self.target)
        self.state[action] = 1 - self.state[action]
        new_score = np.sum(self.state == self.target)
        done = False
        if new_score == self.size:
            done = True
        delta = new_score - old_score
        if delta == 1:
            return 0., done
        elif delta == 2:
            return -1., done
        else:
            assert delta in (0, -1)

    def reset(self):
        self.state = np.random.randint(2, size=self.size)
        self.target = np.random.randint(2, size=self.size)

class Trainer(object):
    def __init__(self, size, params):
        self.size = size
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = Env(size)

        self.policy_net = DQN(self.size).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.reward_tracker = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.tb_writer = SummaryWriter('results')

    def prepareState(self):
        state = np.copy(self.env.state)
        target = np.copy(self.env.target)
        import pdb; pdb.set_trace()
        # check the dimensions work here
        return torch.tensor(np.concatenate([state, target], axis=0), device=self.device)

    def addExperience(self):
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.size)], device=self.device)
        else:
            action = torch.argmax(self.policy_net(self.env.state), dim=1).to(self.device)
        previous_state = self.prepareState()
        reward, done = self.env.step(action.item())
        reward_t = torch.tensor(reward, device=self.device)
        current_state = self.prepareState()
        self.memory.push(previous_state, action, reward_t, current_state)
        return reward, done

    def optimizeModel(self):
        import pdb; pdb.set_trace()
        transitions = self.memory.sample(self.params['batch_size'])
        batch = self.transition(*zip(*transitions))
        # tensor of 1's and 0's for all next states which are not None
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)),
        # device=self.device, dtype=torch.uint8)
        # a tensor of all next states which are not None
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        next_state_batch = torch.cat(list(batch.next_state))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # next_state_values = torch.zeros(self.params['batch_size'], device=self.device)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        frame_idx = 0
        for episode in range(NUM_EPISODES):
            self.env.reset()
            max_iterations = self.size + 10
            for iteration in range(max_iterations):
                frame_idx += 1
                reward, done = self.addExperience()

                if not len(self.memory) > self.memory.capacity:
                    if done:
                        break
                    continue

                if len(self.memory) == self.memory.capacity:
                    print('Done Prefetching.')
                    break
                if iteration == max_iterations - 1:
                    done = True

                self.optimizeModel()
                if frame_idx % self.params['target_net_sync'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done:
                    print('Episode Completed:', episode)
                    print('Score for Epoch %s' % reward)
                    self.reward_tracker.add(reward)
                    self.tb_writer.add_scalar('Score', reward, episode)
                    self.tb_writer.add_scalar('Average Score', self.reward_tracker.meanScore(), episode)
                    self.tb_writer.add_scalar('Steps in Episode', iteration, episode)

def cleanup():
    if os.path.isdir('results_continuous'):
        shutil.rmtree('results_continuous')
    csv_txt_files = [x for x in os.listdir('.') if '.TXT' in x or '.csv' in x]
    for csv_txt_file in csv_txt_files:
        os.remove(csv_txt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    gpu_num = parser.parse_args().gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    cleanup()
    print('Creating Trainer')
    trainer = Trainer(HYPERPARAMS, 5)
    print('Trainer Initialized')
    trainer.train()



