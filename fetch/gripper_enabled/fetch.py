#!/usr/bin/env python3
import random
import numpy as np
import torch
import torch.optim as optim

import torch.nn as nn
from PIL import Image
import copy
from collections import namedtuple
from torch.autograd import Variable
import cv2
import time
import argparse
import csv
from tensorboardX import SummaryWriter
import shutil

import os

import sys
# sys.path.pop(0)
# import gym
sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/gripper_enabled')
from gym.envs.robotics import fetch_env
from gym import utils
from gym.wrappers.time_limit import TimeLimit


NUM_EPISODES = 1500


HYPERPARAMS = {
        'replay_size':      8000,
        'replay_initial':   7900,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5 * 2,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
}



class DQN(nn.Module):
    def __init__(self, input_shape, n_actions, device):
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


    def forward(self, x):
        x = self.conv(x).view(x.size()[0], -1)
        return self.fc(x)


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

    def showCapacity(self):
        print('Buffer Capacity:', len(self.memory) * 1. / self.capacity)


class Trainer(object):
    def __init__(self, seed, anneal_count, warm_start_path=''):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = self.makeEnv()
        self.env = self.env.unwrapped

        #Actions:
        # 0 -- increment X
        # 1 -- decrement X
        # 2 -- increment Y
        # 3 -- decrement Y
        # 4 -- increment Z
        # 5 -- decrement Z
        # 6 -- increment gripper
        # 7 -- decrement gripper

        self.initial_gripper_position = copy.deepcopy(self.env.sim.data.get_site_xpos('robot0:grip'))


        self.action_space = 8
        self.observation_space = [3, 102, 205]
        if not warm_start_path:
            self.policy_net = DQN(self.observation_space, self.action_space, self.device).to(self.device)
        else:
            self.policy_net = torch.load(warm_start_path)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_tracker = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.episode = 0
        self.state = self.preprocess(self.reset())
        self.tb_writer = SummaryWriter('results')
        self.tb_writer.add_graph(self.policy_net, (copy.deepcopy(self.state),))
        self.score = 0
        self.batch_size = self.params['batch_size']
        self.task = 5
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        self.movement_count = 0
        self.seed = seed
        self.csv_file = open('seed%s_scores.csv' % self.seed, 'w+')
        self.writer = csv.writer(self.csv_file)

        self.min_radius = 0.038
        self.anneal_count = anneal_count
        self.remaining_anneals = anneal_count + 1

        self.initial_differential_radius = np.linalg.norm(
            self.initial_gripper_position - self.initial_object_position) - self.min_radius
        self.initial_differential_volume = 4. / 3 * np.pi * self.initial_differential_radius ** 3
        self.current_radius = None
        self.updateRewardRadius()

        self.stage_count = 0
        self.target_height = 0.55
        self.xy_threshold = None  # TODO
        self.finger_threshold = None  # TODO


    def updateRewardRadius(self):
        current_volume = self.remaining_anneals * 1. / (self.anneal_count + 1) * self.initial_differential_volume
        current_differential_radius = (0.75 * current_volume / np.pi) ** (1 / 3)
        self.current_radius = current_differential_radius + self.min_radius
        self.remaining_anneals -= 1
        self.reward_tracker.rewards = []
        self.reward_tracker.mean_score = 0
        self.epsilon_tracker.reset_epsilon()
        print('RADIUS DECREASED. Remaining Anneals:', self.remaining_anneals)


    def makeEnv(self):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        env = fetch_env.FetchEnv('fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type='sparse')
        return TimeLimit(env)


    def getGripperWidth(self):
        right_finger = self.env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')[2]
        left_finger = self.env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')[2]
        return abs(right_finger - left_finger)


    def reset(self):
        self.env.reset()
        self.env.render()
        self.env.sim.nsubsteps = 2
        return self.env.render(mode='rgb_array')

    def preprocess(self, state):
        state = state[230:435, 50:460]
        state = cv2.resize(state, (state.shape[1]//2, state.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)/256
        state = np.swapaxes(state, 0, 2)
        return torch.tensor(state, device=self.device).unsqueeze(0)


    # indices are x, y, z, gripper
    def convertAction(self, action):
        movement = np.zeros(4)
        if action.item() % 2 == 0:
            movement[action.item() // 2] += 1
        else:
            movement[action.item() // 2] -= 1
        return movement


    def addExperience(self):
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.action_space)], device=self.device)
        else:
            action = torch.argmax(self.policy_net(self.state), dim=1).to(self.device)
        gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.env.step(self.convertAction(action))
        self.movement_count += 1
        next_state = self.preprocess(self.env.render(mode='rgb_array'))
        reward, done = self.getReward()
        self.reward_tracker.add(self.score)
        done = done or self.movement_count == 1500

        if done:
            self.memory.push(self.state, action, torch.tensor([reward], device=self.device), None)
            self.state = self.preprocess(self.reset())
            self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        else:
            self.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state)
            self.state = next_state
        return done


    def optimizeModel(self):
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # figure out how generous to be on xy constraint
    # figure out how generous to be with finger pinching constraint
    # what is the height threshold? How high should I try to raise this up?
    '''
    Task 1: Touch the block, discrete reward
    Task 2: Touch the block, continuous reward
    Task 3: Annealing Binary Reward. Done if goal sphere entered or success >= 90%
    *** All Tasks Below Are To Pick Up The Block ***
    Task 4: 1 stage continuous reward
    Task 5: 2 stage continuous reward
    Task 6: 3 stage continuous reward
    Task 7: 3 stage continuous reward, penalty for opening fingers while holding block
    Task 8: 3 stage continuous reward, penalty for decreasing in height while holding block
    Task 9: 1 stage binary reward
    Task 10: 2 stage binary reward
    Task 11: 3 stage binary reward
    Task 12: 3 stage binary reward, penalty for opening fingers
    Task 13: 3 stage binary reward, penalty for decreasing in height
    
    For old xy reward function
    https://github.com/julianalverio/fetch/blob/ea076c97ec7e7e15fcd49136ae43188e231ffac6/fetch/old_experiments/gripper_enabled/prepare_to_grasp/fetch.py
    '''
    def getReward(self):
        done = False
        gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        object_position = self.env.sim.data.get_site_xpos('object0')
        if self.task == 1:
            if np.linalg.norm(self.initial_object_position - object_position) > 1e-3:
                self.score += 1.
                return 1., True
            return 0., False

        if self.task == 2:
            distance = np.linalg.norm(gripper_position - object_position)
            reward = -distance
            if np.linalg.norm(self.initial_object_position - object_position) > 1e-3:
                reward += 10.
                self.score += 1.
                done = True
            return reward, done

        if self.task == 3:
            if self.remaining_anneals >= 1:
                if np.linalg.norm(gripper_position - object_position) < self.current_radius:
                    self.score = 1.
                    return 1., True
                else:
                    return -self.movement_count / 300., False
            elif np.linalg.norm(self.initial_object_position - object_position) > 1e-3:
                print('DONE! MEAN SCORES: ', self.reward_tracker.meanScore())
                self.score = 1.
                return 10., True
            else:
                return -self.movement_count / 300., False

        # continuous reward as one task
        if self.task == 4:
            reward = 0.


        if self.task == 4:
            reward = 0.
            if self.task_part_1:
                reward -= self.movement_count / 300.

                # Get directly over the block
                xy_distance = np.linalg.norm(gripper_position[:2] - object_position[:2])
                reward -= xy_distance

                # Don't be too high up. Lower bound: 0.4125. Upper bound: 0.44. Ideal: 0.42
                z_distance = abs(gripper_position[2] - 0.42)
                reward -= z_distance

                if z_distance <= 0.02 and xy_distance < 0.01:
                    self.score = 1.
                self.task_part_1 = False
                return reward, False

            else:
                xy_distance = np.linalg.norm(gripper_position[:2] - object_position[:2])
                if xy_distance > 0.01:
                    reward -= 2
                height_difference = object_position[2] - 0.55
                reward -= height_difference

                if height_difference < 0.02:
                    reward += 5

                return reward, object_position[2] >= 0.54

        if self.task == 5:
            reward = 0.
            delta_z = object_position[2] - self.initial_object_position[2]
            if delta_z > 0.02:
                self.elevated_count += 1
            if self.elevated_count > 20:
                reward += delta_z
            return rewared, (delta_z > 0.02 and self.elevated_count > 20)

    def train(self):
        import pdb; pdb.set_trace()
        frame_idx = 0
        while True:
            frame_idx += 1
            # execute one move
            done = self.addExperience()

            # are we done prefetching?
            if len(self.memory) < self.params['replay_initial']:
                continue
            if len(self.memory) == self.params['replay_initial']:
                self.episode, self.movement_count, self.score = 0, 0, 0
                print("Done Prefetching.")
                self.reset()

            # is this round over?
            if done:
                self.reward_tracker.add(self.score)

                print('Episode Completed:', self.episode)
                print('Score:', self.score)
                print('Perceived Mean Score', self.reward_tracker.meanScore())
                print('Actual Mean Score', np.mean(self.reward_tracker.rewards))
                print('Remaining Anneals:', self.remaining_anneals)
                print('Steps in this episode:', self.movement_count)
                print('Epsilon:', self.epsilon_tracker.percievedEpsilon())

                self.tb_writer.add_scalar('Score for Epoch', self.score, self.episode)
                self.tb_writer.add_scalar('Perceived Mean Score', self.reward_tracker.meanScore(), self.episode)
                self.tb_writer.add_scalar('Actual Mean Score', np.mean(self.reward_tracker.rewards), self.episode)
                self.tb_writer.add_scalar('Remaining Anneals', self.remaining_anneals, self.episode)
                self.tb_writer.add_scalar('Steps in this Episode', self.movement_count, self.episode)
                self.tb_writer.add_scalar('Epsilon', self.epsilon_tracker.percievedEpsilon(), self.episode)

                self.writer.writerow(
                    [self.episode, self.score, self.reward_tracker.meanScore(), np.mean(self.reward_tracker.rewards),
                     self.remaining_anneals, self.epsilon_tracker._epsilon])
                self.csv_file.flush()

                self.score = 0
                self.movement_count = 0
                self.episode += 1

                self.task_part_1 = True
                print('Starting Episode:', self.episode)

            if self.remaining_anneals > 0 and self.reward_tracker.meanScore() > 0.9:
                self.updateRewardRadius()
            if self.remaining_anneals == 0 and self.reward_tracker.meanScore() == 1:
                return

            self.optimizeModel()
            if frame_idx % self.params['target_net_sync'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.episode == NUM_EPISODES:
                print("DONE WITH ALL EPISODES")
                return

    def renderalot(self, count=200):
        for _ in range(count):
            self.env.render()


    def playback(self, path):
        self.target_net = torch.load(path)
        for _ in range(4):
            self.env.render()
        done = False
        while not done:
            self.env.render(mode='human')
            action = self.convertAction(torch.argmax(self.target_net(self.state), dim=1).to(self.device))
            self.env.step(action)
            self.state = self.preprocess(self.env.render(mode='rgb_array'))
            # reward, done = self.getReward()

def cleanup():
    if os.path.isdir('results'):
        shutil.rmtree('results')
    csv_txt_files = [x for x in os.listdir('.') if '.TXT' in x or '.csv' in x]
    for csv_txt_file in csv_txt_files:
        os.remove(csv_txt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    parser.add_argument('anneal_count', type=int, default='10')
    args = parser.parse_args()
    gpu_num = args.gpu
    anneal_count = args.anneal_count
    print('GPU:', gpu_num)
    print('ANNEALS:', anneal_count)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    seed = random.randrange(0, 100)
    print('RANDOM SEED: ', seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('cleaning up...')
    cleanup()
    print('Creating Trainer Object')
    trainer = Trainer(seed, anneal_count)
    print('Trainer Initialized')
    print("Prefetching Now...")
    # print('showing example now')
    trainer.train()
    # trainer.playback('fetch_seed25_8500.pth')





