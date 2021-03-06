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
from gym.envs.robotics import fetch_env
from gym.wrappers.time_limit import TimeLimit
from replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv
import time


# Actions:
# 0 -- increment X
# 1 -- decrement X
# 2 -- increment Y
# 3 -- decrement Y
# 4 -- increment Z
# 5 -- decrement Z
# 6 -- continuously try to open gripper
# 7 -- continuously try to close gripper

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


# TODO: add support for extensions: Dueling, PER, HER
# TODO: add a pick and place environment where the starting position can also change
# TODO: finish dealing with substeps, then run everything through to the end to make sure it works (most likely the optimizeModel will break)
class Trainer(object):
    def __init__(self, hyperparams, dueling=False, HER=False):
        self.params = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_space = 8
        self.observation_space = [3, 102, 205]

        if HER:
            self.episode_buffer = []
        self.HER = HER

        self.envs, self.env_names = self.makeEnvs()
        self.env = None

        if dueling:
            self.policy_net = DuelingDQN(self.observation_space, self.action_space).to(self.device)
        else:
            self.policy_net = DQN(self.observation_space, self.action_space).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        self.epsilon_scheduler = LinearScheduler(self.params['epsilon_start'], self.params['epsilon_final'], timespan=self.params['epsilon_frames'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_trackers = [ValueTracker() for _ in range(len(self.envs))]
        self.episode_counters = [0] * len(self.envs)
        self.memory = ReplayBuffer(self.params['replay_size'])
        self.tb_writer = SummaryWriter('results')

        self.gripper_states = [0] * len(self.envs)  # 0 for opening, 1 for closing
        self.task = 0
        self.place_env_idx = None

    def makeEnvs(self):
        envs = list()
        env_names = list()
        # envs.append(FetchPickAndPlaceEnv())
        # env_names.append('pick and place')
        # envs.append(FetchSlideEnv)
        # env_names.append('slide')
        # envs.append(FetchPushEnv())
        # env_names.append('push')
        envs.append(FetchReachEnv())
        env_names.append('reach')
        # envs.append(FetchPickAndPlaceEnv(target_in_the_air=False))
        # env_names.append('place')
        # self.place_env_idx = len(envs) - 1
        return envs, env_names

    def reset(self):
        self.task = random.randrange(0, len(self.envs))
        print('Task:', self.task)
        env = self.envs[self.task]
        env.reset()
        if self.task == self.place_env_idx:
            self.resetforPlacing(env)
        else:
            env.move([0, 0, 0, 0], 40)
        self.gripper_states[self.task] = 0
        self.env = env
        self.env.render()

    # there are some additional movements here to compensate for momentum
    def resetforPlacing(self, env):
        object_position = env.sim.data.get_site_xpos('object0')
        gripper_position = env.sim.data.get_site_xpos('robot0:grip')
        starting_position = copy.deepcopy(gripper_position)

        # Get x just right
        while gripper_position[0] > object_position[0]:
            env.step([-1, 0, 0, 0])
        while gripper_position[0] < object_position[0]:
            env.step([1, 0, 0, 0])

        # Get y just right
        while gripper_position[1] > object_position[1]:
            env.step([0, -1, 0, 0])
        while gripper_position[1] < object_position[1]:
            env.step([0, 1, 0, 0])

        env.move([0, 0, 0, 1], count=10)  # open
        env.move([0, 0, -1, 1], count=20)  # drop
        env.move([0, 0, 0, -1], count=15)  # close

        # get z right
        while gripper_position[2] < starting_position[2]:
            env.step([0, 0, 1, -1])
        env.step([0, 0, -1, -1])
        # get y right
        while gripper_position[1] < starting_position[1]:
            env.step([0, 1, 0, -1])
        env.step([0, -1, 0, -1])
        while gripper_position[1] > starting_position[1]:
            env.step([0, -1, 0, -1])
        env.step([0, 1, 0, -1])
        # get x right
        while gripper_position[0] > starting_position[0]:
            env.step([-1, 0, 0 -1])
        env.step([1, 0, 0 -1])
        while gripper_position[0] < starting_position[0]:
            env.step([1, 0, 0 - 1])
        env.step([-1, 0, 0 - 1])

    def preprocess(self, state):
        state = state[230:435, 50:460]
        state = cv2.resize(state, (state.shape[1]//2, state.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)/256
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
        state, goal = self.env.getStateAndGoal()
        if goal_prime is not None:
            goal = goal_prime
        if state_prime is not None:
            state = state_prime
        else:
            state = self.preprocess(state)
        goal_zeros = np.zeros([1, 1, 205, 102], dtype=np.float32)
        goal_zeros[0, 0, 0, 0:3] = goal
        goal = torch.tensor(goal_zeros, device=self.device)
        return torch.cat([state, goal], dim=1)

    def addExperience(self):
        state = self.prepareState()
        if random.random() < self.epsilon_scheduler.updateAndGetValue():
            action = torch.tensor([random.randrange(self.action_space)], device=self.device)
        else:
            action = torch.argmax(self.policy_net(state), dim=1).to(self.device)
        action_converted = self.convertAction(action)
        self.env.step(action_converted)
        next_state = self.prepareState()
        reward = torch.tensor(self.env.getReward(), device=self.device)  # 0 or -1
        self.memory.add(state, action, reward, next_state, reward)
        if self.HER:
            self.episode_buffer.append((state[:, 0:3, :, :], action, next_state[:, 0:3, :, :], self.env.getGoalAchieved()))
        return reward, reward == 0

    def optimizeModel(self):
        states, actions, rewards, next_states, _ = self.memory.sample(self.params['batch_size'])
        states = torch.tensor(states, device=self.device).squeeze(1)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device).squeeze(1)
        state_action_values = self.policy_net(states).gather(1, actions)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + rewards
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # def optimizeModelPER(self):
    #     # transitions, ISWeights, tree_idx = self.memory.sample(self.batch_size)
    #     ISWeights = torch.tensor(ISWeights, device=self.device)
    #     batch = self.transition(*zip(*transitions))
    #     next_states = batch.next_state
    #     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.device, dtype=torch.uint8)
    #     non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    #     non_final_tasks = torch.cat([batch.task[i] for i in range(self.params['batch_size']) if next_states[i] is not None])
    #     state_batch = torch.cat(list(batch.state))
    #     action_batch = torch.cat(list(batch.action))
    #     reward_batch = torch.cat(list(batch.reward))
    #     task_batch = torch.cat(list(batch.task))
    #     state_action_values = self.policy_net(state_batch, task_batch).gather(1, action_batch.unsqueeze(1))
    #     next_state_values = torch.zeros(self.params['batch_size'], device=self.device)
    #     next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_tasks).max(1)[0].detach()
    #     expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
    #     abs_errors = abs(expected_state_action_values.unsqueeze(1) - state_action_values)
    #     loss = torch.sum((abs_errors ** 2) * ISWeights)
    #     self.memory.batch_update(tree_idx, abs_errors.detach().cpu().numpy() + 1e-6)
    #     # loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    def logEpisode(self, iteration, reward):
        self.reward_trackers[self.task].add(reward)
        self.tb_writer.add_scalar('Steps per episode | task=%s' % self.env_names[self.task], iteration, self.episode_counters[self.task])
        self.tb_writer.add_scalar('Score | task=%s' % self.env_names[self.task], reward, self.episode_counters[self.task])
        self.tb_writer.add_scalar('Average Score | task=%s' % self.env_names[self.task], self.reward_trackers[self.task].getMean(), self.episode_counters[self.task])
        self.tb_writer.add_scalar('Epsilon', self.epsilon_scheduler.observeValue(), sum(self.episode_counters))
        self.episode_counters[self.task] += 1

    def prefetch(self, max_iterations):
        while len(self.memory) < self.params['replay_size']:
            self.reset()
            for iteration in range(max_iterations):
                reward = self.addExperience()
                done = reward == 0
                if done:
                    break
        print('Done Prefetching.')

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
        self.prefetch(max_iterations)
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
                    print('Episode Completed:', episode, 'Task:', self.task, 'Score:', reward)
                    self.logEpisode(iteration, reward)
                    if self.HER:
                        self.HERFinal()
                    break


def cleanup():
    if os.path.isdir('results'):
        shutil.rmtree('results')
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


if __name__ == "__main__":
    hyperparams = {
        'replay_size': 100,  # 8k baseline
        'replay_initial': 8000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 5 * 2,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    }

    NUM_EPISODES = 2000
    MAX_ITERATIONS = 400

    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    parser.add_argument('her', type=bool)
    args = parser.parse_args()
    gpu_num = args.gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    cleanup()
    print('Creating Trainer')
    trainer = Trainer(hyperparams, dueling=False, HER=args.her)
    print('Trainer Initialized')
    trainer.train(NUM_EPISODES, MAX_ITERATIONS)





