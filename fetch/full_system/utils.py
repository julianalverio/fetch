import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple
import torch.optim as optim

np.random.seed(5)
random.seed(5)
torch.backends.cudnn.deterministic = True
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)


TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class EpsilonTracker:
    def __init__(self, epsilon_start=1., epsilon_final=0.02, epsilon_frames=10**5):
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self._epsilon = self.epsilon_start
        self.epsilon_delta = 1.0 * (self.epsilon_start - self.epsilon_final) / self.epsilon_frames

    def epsilon(self):
        old_epsilon = self._epsilon
        self._epsilon -= self.epsilon_delta
        return max(old_epsilon, self.epsilon_final)

    def reset_epsilon(self):
        self._epsilon = self.epsilon_start

    def percievedEpsilon(self):
        return max(self._epsilon, self.epsilon_final)

class ReplayMemory(object):
    def __init__(self, transition, capacity=8000):
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

    def showCapacity(self):
        print('Buffer Capacity:', len(self.memory) * 1. / self.capacity)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, device, hyperparams):
        super(DQN, self).__init__()
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out = self.conv(Variable(torch.zeros(1, *input_shape)))
        conv_out_size = int(np.prod(conv_out.size()))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.memory = ReplayMemory(TRANSITION, hyperparams['memory_size'])
        self.epsilon_tracker = EpsilonTracker()
        self.optimizer = optim.Adam(self.parameters(), lr=hyperparams['learning_rate'])
        self.counter = 0
        self.hyperparams = hyperparams


    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)

    def getAction(self, state, task):
        if random.random() < self.epsilon_tracker.epsilon():
            return torch.tensor([random.randrange(self.num_actions)], device=self.device)
        else:
            return torch.argmax(self(state), task, dim=1).to(self.device)

    def optimizeModel(self, target_net):
        self.counter += 1
        transitions = self.memory.sample(self.hyperparams['batch_size'])
        batch = TRANSITION(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        state_action_values = self(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.hyperparams['batch_size'], device=self.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.hyperparams['gamma']) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.counter % self.hyperparams['target_net_sync'] == 0:
            target_net.load_state_dict(self.state_dict())




