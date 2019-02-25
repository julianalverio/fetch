#!/usr/bin/env python3
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
import time
import sys
import copy
sys.path.insert(0, '..')
sys.path.insert(0, '.')
from gym.envs.robotics import fetch_env
from gym.wrappers.time_limit import TimeLimit
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv
from PIL import Image
import time
import math

# Actions:
# 0 -- increment X
# 1 -- decrement X
# 2 -- increment Y
# 3 -- decrement Y
# 4 -- increment Z
# 5 -- decrement Z
# 6 -- continuously try to open gripper
# 7 -- continuously try to close gripper


class ReplayMemory(object):
    def __init__(self, capacity, transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = transition

    def add(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(self.transition(*args))
        else:
            self.memory[self.position] = self.transition(*args)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def showCapacity(self):
        print('Buffer Capacity:', len(self.memory) * 1. / self.capacity)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()
        self.num_actions = num_actions

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

        self.fc_adv = nn.Sequential(
            nn.Linear(in_features=conv_out_size + 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(in_features=conv_out_size + 3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def forward(self, state_and_goal):
        state = state_and_goal[:, 0:3, :, :]
        goal = state_and_goal[:, -1, 0, :3]
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        x = torch.cat([state, goal], dim=1)

        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.num_actions)
        return val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

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
            nn.Linear(conv_out_size + 3, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state_and_goal):
        state = state_and_goal[:, 0:3, :, :]
        goal = state_and_goal[:, -1, 0, :3]
        state = self.conv(state)
        state = state.view(state.size(0), -1)
        x = torch.cat([state, goal], dim=1)
        return self.fc(x)


class ValueTracker(object):
    def __init__(self, length=20):
        self.length = length
        self.values = []
        self.position = 0
        self.mean = 0

    def add(self, value):
        if len(self.values) < self.length:
            self.values.append(value)
        else:
            self.values[self.position] = value
            self.position = (self.position + 1) % self.length
        self.mean = np.mean(self.values)

    def getMean(self):
        return self.mean


class LinearScheduler(object):
    def __init__(self, start, stop, delta=None, timespan=None):
        assert delta or timespan
        self.value = start
        self.stop = stop
        if delta:
            self.delta = float(delta)
        elif timespan:
            self.delta = (stop - start) / float(timespan)

    def updateAndGetValue(self):
        self.value += self.delta
        return self.observeValue()

    def observeValue(self):
        if self.delta > 0:
            return min(self.value, self.stop)
        else:
            return max(self.value, self.stop)

class Trainer(object):
    def __init__(self, hyperparams, dueling=False, HER=False, reach=False, pick=False, push=False, slide=False, place=False, resilience=False):
        self.params = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_space = 8
        # self.observation_space = [3, 127, 102]
        self.place_env_idx = None

        if HER:
            self.episode_buffer = []
        self.HER = HER

        self.makeEnvs(reach, pick, push, slide, place)
        self.env = None
        self.resilience = resilience
        initial_obs = self.preprocess(self.envs[0].render(mode='rgb_array')).shape
        self.observation_space = [initial_obs[1], initial_obs[2], initial_obs[3]]
        self.dueling = dueling
        if dueling:
            self.policy_net = DuelingDQN(self.observation_space, self.action_space).to(self.device)
        else:
            self.policy_net = DQN(self.observation_space, self.action_space).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        self.epsilon_scheduler = LinearScheduler(self.params['epsilon_start'], self.params['epsilon_final'], timespan=self.params['total_frames'] * 0.15)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_trackers = [ValueTracker() for _ in range(len(self.envs))]
        self.episode_counters = [0] * len(self.envs)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.directory = self.getDirectory()
        self.cleanup()
        self.tb_writer = SummaryWriter(self.directory)

        self.gripper_states = [0] * len(self.envs)  # 0 for opening, 1 for closing
        self.task = 0
        self.state = None

    def getDirectory(self):
        directory = 'results/'
        for task in self.env_names:
            directory += task
        if self.HER:
            directory += 'her'
        if self.dueling:
            directory += 'dueling'
        return directory

    def makeEnvs(self, reach, pick, push, slide, place):
        envs = list()
        env_names = list()
        if pick:
            envs.append(FetchPickAndPlaceEnv())
            env_names.append('pick and place')
        if slide:
            envs.append(FetchSlideEnv())
            env_names.append('slide')
        if push:
            envs.append(FetchPushEnv())
            env_names.append('push')
        if reach:
            envs.append(FetchReachEnv())
            env_names.append('reach')
        if place:
            envs.append(FetchPickAndPlaceEnv(target_in_the_air=False))
            env_names.append('place')
            self.place_env_idx = len(envs) - 1
        self.envs = envs
        self.env_names = env_names

    def reset(self):
        self.task = random.randrange(0, len(self.envs))
        # print('Task:', self.task)
        self.env = self.envs[self.task]
        self.env.reset()
        self.env.sim.nsubsteps = 2
        if self.task == self.place_env_idx:
            self.resetforPlacing()
        self.gripper_states[self.task] = 0
        self.state = self.prepareState()
        # self.env.render()  # for debugging

    # there are some additional movements here to compensate for momentum
    # WARNING: THIS METHOD HAS YET TO BE TUNED
    def resetforPlacing(self):
        object_position = self.env.sim.data.get_site_xpos('object0')
        gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        starting_position = copy.deepcopy(gripper_position)

        self.renderalot()
        import pdb; pdb.set_trace()
        # Get x just right
        if gripper_position[0] > object_position[0]:
            while gripper_position[0] > object_position[0]:
                self.env.step([-1., 0., 0., 0.])
                self.renderalot()
            self.move([1., 0., 0., 0.], 2)
        elif gripper_position[0] < object_position[0]:
            while gripper_position[0] < object_position[0]:
                self.env.step([1., 0., 0., 0.])
                self.renderalot()
            self.move([-1., 0., 0., 0.], 2)

        # Get y just right
        if gripper_position[1] > object_position[1]:
            while gripper_position[1] > object_position[1]:
                self.env.step([0., -1., 0., 0.])
                self.renderalot()
            self.env.step([0., 1., 0., 0.])

        elif gripper_position[1] < object_position[1]:
            while gripper_position[1] < object_position[1]:
                self.env.step([0., 1., 0., 0.])
                self.renderalot()
            self.env.step([0., -1., 0., 0.])

        self.env.move([0., 0., 0., 1.], count=10)  # open
        self.renderalot()
        self.env.move([0., 0., -1., 1.], count=20)  # drop
        self.renderalot()
        import pdb; pdb.set_trace()
        self.env.move([0., 0., 0., -1.], count=15)  # close
        self.renderalot()

        # get z right
        while gripper_position[2] < starting_position[2]:
            self.env.step([0., 0., 1., -1.])
            self.renderalot()
        self.env.step([0., 0., -1., -1.])
        self.renderalot()
        # get y right
        while gripper_position[1] < starting_position[1]:
            self.env.step([0., 1., 0., -1.])
            self.renderalot()
        self.env.step([0., -1., 0., -1.])
        self.renderalot()
        while gripper_position[1] > starting_position[1]:
            self.env.step([0., -1., 0., -1.])
            self.renderalot()
        self.env.step([0., 1., 0., -1.])
        self.renderalot()
        # get x right
        while gripper_position[0] > starting_position[0]:
            self.env.step([-1., 0., 0., -1.])
            self.renderalot()
        self.env.step([1., 0., 0., -1.])
        self.renderalot()
        while gripper_position[0] < starting_position[0]:
            self.env.step([1., 0., 0, -1.])
            self.renderalot()
        self.env.step([-1., 0., 0., -1])
        self.renderalot()
        import pdb; pdb.set_trace()

    def preprocess(self, state):
        state = state[180:435, 50:460]
        state = cv2.resize(state, (state.shape[1]//4, state.shape[0]//4), interpolation=cv2.INTER_AREA).astype(np.float32)/256
        state = np.swapaxes(state, 0, 2)
        return torch.tensor(state, device=self.device).unsqueeze(0)

    def renderalot(self, count=6):
        for _ in range(count):
            self.env.render()

    # indices are x, y, z, gripper
    def convertAction(self, action):
        movement = np.zeros(4)
        if action.item() % 2 == 0:
            movement[action.item() // 2] += 1
        else:
            movement[action.item() // 2] -= 1
        if action.item() == 6:
            self.gripper_states[self.task] = 0
        elif action.item() == 7:
            self.gripper_states[self.task] = 1
        if self.gripper_states[self.task] == 0:
            movement[-1] = 1
        elif self.gripper_states[self.task] == 1:
            movement[-1] = -1
        return movement

    def prepareState(self, state_prime=None, goal_prime=None):
        if goal_prime is not None and state_prime is not None:
            goal = goal_prime
            state = state_prime
        else:
            state, goal = self.env.getStateAndGoal()
            state = self.preprocess(state)
        goal_zeros = np.zeros([1, 1] + self.observation_space[1:], dtype=np.float32)
        goal_zeros[0, 0, 0, 0:3] = goal
        goal = torch.tensor(goal_zeros, device=self.device)
        return torch.cat([state, goal], dim=1)

    def addExperience(self):
        if random.random() < self.epsilon_scheduler.updateAndGetValue():
            action = torch.tensor([random.randrange(self.action_space)], device=self.device)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_net(self.state), dim=1).to(self.device)
        action_converted = self.convertAction(action)
        self.env.step(action_converted)
        next_state = self.prepareState()
        reward = torch.tensor(self.env.getReward(), device=self.device).unsqueeze(0)  # 0 or -1
        self.memory.add(self.state, action, reward, next_state)
        if self.HER:
            self.episode_buffer.append((self.state[:, 0:3, :, :], action, next_state[:, 0:3, :, :], self.env.getGoalAchieved()))
        self.state = next_state
        return reward, reward == 0

    def optimizeModel(self):
        batch = self.transition(*zip(*self.memory.sample(self.params['batch_size'])))
        states = torch.cat(list(batch.state))
        actions = torch.cat(list(batch.action)).unsqueeze(1)
        rewards = torch.cat(list(batch.reward))
        next_states = torch.cat(list(batch.next_state))
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + rewards
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def logEpisode(self, iteration, reward):
        self.reward_trackers[self.task].add(reward.item())
        self.tb_writer.add_scalar('Steps per episode | task=%s' % self.env_names[self.task], iteration, self.episode_counters[self.task])
        self.tb_writer.add_scalar('Score | task=%s' % self.env_names[self.task], reward, self.episode_counters[self.task])
        self.tb_writer.add_scalar('Average Score | task=%s' % self.env_names[self.task], self.reward_trackers[self.task].getMean(), self.episode_counters[self.task])
        self.tb_writer.add_scalar('Epsilon', self.epsilon_scheduler.observeValue(), sum(self.episode_counters))
        self.episode_counters[self.task] += 1

    def prefetch(self, max_iterations):
        while len(self.memory) < self.params['replay_initial']:
            self.reset()
            for iteration in range(max_iterations):
                print(len(self.memory))
                print(torch.cuda.memory_allocated())
                import pdb; pdb.set_trace()
                reward = self.addExperience()
                done = reward == 0
                if done:
                    break
        print('Done Prefetching.')

    def move(self, action, count=10):
        for _ in range(count):
            self.env.step(action)

    def randomMove(self):
        rand = random.random()
        if rand > 0.01:
            action = np.zeros(4)
            idx = random.randrange(4)
            if rand > 0.5:
                action[idx] = 1
            else:
                action[idx] = -1
            self.move(action, count=random.randrange(5))

    # 'FINAL' implementation
    def HERFinal(self):
        final_goal = self.episode_buffer[-1][-1]
        for state, action, next_state, goal_achieved in self.episode_buffer:
            state = self.prepareState(state_prime=state, goal_prime=final_goal)
            next_state = self.prepareState(state_prime=next_state, goal_prime=final_goal)
            distance = np.linalg.norm(goal_achieved - final_goal)
            reward = -(distance < self.env.distance_threshold).astype(np.float32)
            reward = torch.tensor(np.array(reward), device=self.device)
            self.memory.add(state, action, reward, next_state, 0)
        self.episode_buffer = list()

    def train(self, num_episodes, max_iterations):
        frame_idx = 0
        for episode in range(num_episodes):
            self.reset()
            for iteration in range(max_iterations):
                reward, done = self.addExperience()
                self.optimizeModel()
                frame_idx += 1
                if frame_idx % self.params['target_net_sync'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done or iteration == max_iterations - 1:
                    print('Episode Completed:', episode)
                    self.logEpisode(iteration, reward)
                    if self.HER:
                        self.HERFinal()
                    break
                # if self.resilience:
                #     self.randomMove()

    def cleanup(self):
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
        csv_txt_files = [x for x in os.listdir('.') if '.TXT' in x or '.csv' in x]
        for csv_txt_file in csv_txt_files:
            os.remove(csv_txt_file)

# def seed():
#     seed = random.randrange(0, 100)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


def timeStuff(hyperparams, args):
    import time
    times = []
    print('Now timing stuff')
    for _ in range(5):
        trainer = Trainer(hyperparams, dueling=args.dueling, HER=args.her, reach=args.reach, pick=args.pick,
                          push=args.push, slide=args.slide, place=args.place)
        trainer.prefetch(32)
        start = time.time()
        trainer.train(5, 200)
        times.append(time.time() - start)
    print('Mean time:', np.mean(times))
    print('Standard Dev:', np.std(times))


if __name__ == "__main__":
    hyperparams = {
        'replay_size': 50 * 10**3,
        'replay_initial': 1000,
        'target_net_sync': 500,
        'epsilon_frames': 10**5 * 2,
        'total_frames': 10**6 * 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 5e-4,
        'gamma': 0.99,
        'batch_size': 32
    }

    MAX_ITERATIONS = 300
    NUM_EPISODES = hyperparams['total_frames'] // MAX_ITERATIONS

    parser = argparse.ArgumentParser()
    parser.add_argument("--her", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--reach", action="store_true")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--slide", action="store_true")
    parser.add_argument("--place", action="store_true")
    parser.add_argument("--resilience", action="store_true")
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()
    assert args.reach or args.pick or args.slide or args.place or args.push
    gpu_num = args.gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    print('Creating Trainer')
    trainer = Trainer(hyperparams, dueling=args.dueling, HER=args.her, reach=args.reach, pick=args.pick, push=args.push, slide=args.slide, place=args.place)
    print('Trainer Initialized')
    trainer.prefetch(hyperparams['replay_initial'])
    trainer.prefetch(hyperparams['replay_size'])
    trainer.train(NUM_EPISODES, MAX_ITERATIONS)
