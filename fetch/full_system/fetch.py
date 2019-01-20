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
# sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/full_system')
from gym.envs.robotics import fetch_env
from gym import utils
from gym.wrappers.time_limit import TimeLimit


NUM_EPISODES = 3000
MAX_ITERATIONS = 1500


HYPERPARAMS = {
        'replay_size':      8000 * 4,
        'replay_initial':   7900 * 4,
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


    def forward(self, x, task):
        # check that everything here works
        import pdb; pdb.set_trace()
        x = self.conv(x).view(x.size()[0], -1)
        x = torch.cat(x, torch.tensor(task))
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
    def __init__(self):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = self.makeEnv()

        #Actions:
        # 0 -- increment X
        # 1 -- decrement X
        # 2 -- increment Y
        # 3 -- decrement Y
        # 4 -- increment Z
        # 5 -- decrement Z
        # 6 -- increment gripper
        # 7 -- decrement gripper
        # 8 -- open gripper until specified otherwise
        # 9 -- close gripper until specified otherwise

        self.initial_gripper_position = copy.deepcopy(self.env.sim.data.get_site_xpos('robot0:grip'))

        self.action_space = 10
        self.observation_space = [3, 102, 205]
        self.policy_net = DQN(self.observation_space, self.action_space, self.device).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_tracker0 = RewardTracker()
        self.reward_tracker1 = RewardTracker()
        self.reward_tracker2 = RewardTracker()
        self.reward_tracker3 = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'task'))
        self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.tb_writer = SummaryWriter('results')
        self.gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.object_position = self.env.sim.data.get_site_xpos('object0')

        self.stage_count = 0
        self.target_height = 0.47  # get the block at least this high
        self.x_threshold = 0.01143004  # to be prepared to grip, gripper x must be no further away than this
        self.y_threshold = 0.01121874  # to be prepared to grip, gripper y must be no further away than this
        self.z_threshold = 0.435  # to be prepared to grip, gripper z must be no higher than this
        self.finger_threshold = 0.046195726  # in order to grip the block your fingers must be at least this wide
        # self.previous_height = self.initial_object_position[2]  # for negative reward when you decrease in height

        self.closing = False
        self.opening = False

        # state variables
        self.task0_episode_counter = 0
        self.task1_episode_counter = 0
        self.task2_episode_counter = 0
        self.task3_episode_counter = 0
        self.task = 0

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
        return TimeLimit(env).unwrapped

    def resetSceneForPickUp(self):
        # Get x just right
        while self.gripper_position[0] > self.object_position[0]:
            self.env.step([-1, 0, 0, 0])
            self.env.render()
        while self.gripper_position[0] < self.object_position[0]:
            self.env.step([1, 0, 0, 0])
            self.env.render()
        # Compensate for momentum
        while self.gripper_position[0] > self.object_position[0]:
            self.env.step([-1, 0, 0, 0])
            self.env.render()
        while self.gripper_position[0] < self.object_position[0]:
            self.env.step([1, 0, 0, 0])
            self.env.render()

        # Get y just right
        while self.gripper_position[1] > self.object_position[1]:
            self.env.step([0, -1, 0, 0])
            self.env.render()
        while self.gripper_position[1] < self.object_position[1]:
            self.env.step([0, 1, 0, 0])
            self.env.render()
        # Compensate for momentum
        while self.gripper_position[1] > self.object_position[1]:
            self.env.step([0, -1, 0, 0])
            self.env.render()
        while self.gripper_position[1] < self.object_position[1]:
            self.env.step([0, 1, 0, 0])
            self.env.render()

        # open the fingers
        self.open()
        self.drop()
        self.close()
        self.closing = True
        self.opening = False
        assert self.validGrip()

    def resetSceneForPutDown(self):
        self.resetSceneForPickUp()
        self.drop()
        self.close()
        # 0.58 is the height threshold for picking something up
        while self.gripper_position[2] < 0.58:
            if random.random() < 0.33:
                self.env.step([0, 1, 1, -1])
            elif 0.33 <= random.random() <= 0.66:
                self.env.step([0, -1, 1, -1])
            elif random.random() > 0.66:
                self.env.step([0, 0, 1, -1])
        self.closing = True
        self.opening = False
        assert self.validGrip()

    def getGripperWidth(self):
        right_finger = self.env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')[2]
        left_finger = self.env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')[2]
        return abs(right_finger - left_finger)


    def reset(self):
        self.env.reset()
        self.env.render()
        self.env.sim.nsubsteps = 2
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        if self.task == 2.:
            self.resetSceneForPickUp()
        if self.task in (3., 4.):
            self.resetSceneForPutDown()
        self.state = self.preprocess(self.env.render(mode='rgb_array'))
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))

    def preprocess(self, state):
        state = state[230:435, 50:460]
        state = cv2.resize(state, (state.shape[1]//2, state.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)/256
        state = np.swapaxes(state, 0, 2)
        return torch.tensor(state, device=self.device).unsqueeze(0)


    # indices are x, y, z, gripper
    def convertAction(self, action):
        movement = np.zeros(4)
        if action.item() == 8:
            self.opening = True
            self.closing = False
            movement[-1] = -1
            return movement
        if action.item() == 9:
            self.opening = False
            self.closing = True
            movement[-1] = 1
            return movement
        if action.item() in (7, 8):
            self.opening = False
            self.closing = False
        if action.item() % 2 == 0:
            movement[action.item() // 2] += 1
        else:
            movement[action.item() // 2] -= 1
        if self.opening:
            movement[-1] = 1
        elif self.closing:
            movement[-1] = -1
        return movement

    def openGripper(self):
        while self.getFingerWidth() < 0.1:
            self.env.step([0, 0, 0, 1])
        self.env.render()
        self.gripper_state = 1

    def closeGripper(self):
        while self.getFingerWidth() > 0.0001:
            self.env.step([0, 0, 0, -1])
        self.env.render()
        self.gripper_state = 0


    # for when stage_count == 0
    def addExperience(self):
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.action_space)], device=self.device)
        else:
            # check if the state and task are of the same type
            action = torch.argmax(self.policy_net(self.state, self.task), dim=1).to(self.device)
        action_converted = self.convertAction(action)
        self.env.step(action_converted)
        next_state = self.preprocess(self.env.render(mode='rgb_array'))
        reward, done = self.getReward()
        if done:
            next_state = None

        self.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state, torch.tensor([self.task], device=self.device))

        # if done:
        #     self.state = self.preprocess(self.reset())
        # else:
        #     self.state = next_state

        # self.tb_writer.add_scalar('Score for Epoch', self.score, self.episode)
        # self.tb_writer.add_scalar('Perceived Mean Score', self.reward_tracker.meanScore(), self.episode)
        # self.tb_writer.add_scalar('Actual Mean Score', np.mean(self.reward_tracker.rewards), self.episode)
        # self.tb_writer.add_scalar('Steps in this Episode', self.movement_count, self.episode)
        # self.tb_writer.add_scalar('Epsilon', self.epsilon_tracker.percievedEpsilon(), self.episode)
        return reward, done

    def optimizeModel(self):
        transitions = self.memory.sample(self.params['batch_size'])
        batch = self.transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.params['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Task 0: Grab
    Task 1: Lift
    Task 2: Put down
    Task 3: Stack
    '''
    def getReward(self):
        # if the block falls off the table
        if self.initial_object_position[2] - self.object_position[2] > 0.1:
            return -1., True
        if self.task == 0.:
            if self.validGrip():
                return 1., True
            return 0., False

        if self.task == 1.:
            if not self.validGrip():
                return -1., True
            if self.gripper_position[2] > self.height_threshold:
                return 1., True
            else:
                return 0., False

        if self.task == 2.:
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] < self.drop_height:
                return 1., True
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] >= self.drop_height:
                return -1., True
            else:
                return 0., False

        if self.task == 3.:
            x_difference = abs(self.object_position[0] - self.object1_position[0])
            y_difference = abs(self.object_position[1]) - self.object1_position[1]
            x_check = x_difference < 0.02
            y_check = y_difference < 0.02
            dropped = self.gripper_position[2] - self.object_position[2] >= 0.025
            if dropped:
                if self.object_position[2] <= 0.5 and x_check and y_check:
                    return 1., True
                else:
                    return -1., False
            return 0., False


    def validGrip(self):
        x_difference = abs(self.object_position[0] - self.gripper_position[0])
        y_difference = abs(self.object_position[1] - self.gripper_position[1])
        return x_difference <= self.x_threshold and y_difference <= self.y_threshold \
               and self.getFingerWidth() > self.finger_threshold and self.gripper_position[2] <= 0.435


    def train(self):
        frame_idx = 0
        for episode in range(NUM_EPISODES):
            self.task = float(random.randrange(0, 4))
            print(self.task)
            import pdb; pdb.set_trace()
            self.reset()
            for iteration in range(MAX_ITERATIONS):
                # execute one move
                frame_idx += 1
                reward, done = self.addExperience()

                # are we done prefetching?
                if len(self.memory) < self.params['replay_initial']:
                    continue
                if len(self.memory) == self.params['replay_initial']:
                    print("Done Prefetching.")
                    break

                # is this round over?
                if done:
                    print('Episode Completed:', episode)
                    print('Task: %s' % self.task)
                    print('Score for Epoch %s' % reward)
                    print('Steps in this episode:', iteration)
                    print('Epsilon:', self.policy_net.epsilon_tracker.percievedEpsilon())
                    if self.task == 0.:
                        self.reward_tracker0.add(reward)
                        average_score = self.reward_tracker0.meanScore()
                        self.tb_writer.add_scalar('Task 0 Score', reward, self.task0_episode_counter)
                        self.tb_writer.add_scalar('Task 0 Average Score', self.reward_tracker0.meanScore(),
                                                  self.task0_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 1', iteration, self.task0_episode_counter)
                        self.task0_episode_counter += 1
                    if self.task == 1.:
                        self.reward_tracker1.add(reward)
                        average_score = self.reward_tracker1.meanScore()
                        self.tb_writer.add_scalar('Task 1 Score', reward, self.task1_episode_counter)
                        self.tb_writer.add_scalar('Task 1 Average Score', self.reward_tracker1.meanScore(),
                                                  self.task1_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 1', iteration, self.task1_episode_counter)
                        self.task1_episode_counter += 1
                    if self.task == 2.:
                        self.reward_tracker2.add(reward)
                        average_score = self.reward_tracker2.meanScore()
                        self.tb_writer.add_scalar('Task 2 Score', reward, self.task2_episode_counter)
                        self.tb_writer.add_scalar('Task 2 Average Score', self.reward_tracker2.meanScore(),
                                                  self.task2_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 2', iteration, self.task2_episode_counter)
                        self.task2_episode_counter += 1
                    if self.task == 3.:
                        self.reward_tracker3.add(reward)
                        average_score = self.reward_tracker3.meanScore()
                        self.tb_writer.add_scalar('Task 3 Score', reward, self.task3_episode_counter)
                        self.tb_writer.add_scalar('Task 3 Average Score', self.reward_tracker3.meanScore(),
                                                  self.task3_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 3', iteration, self.task3_episode_counter)
                        self.task3_episode_counter += 1

                    print('Average score: %s' % average_score)
                    self.tb_writer.add_scalar('Epsilon', self.policy_net.epsilon_tracker.percievedEpsilon(), episode)
                    break

                self.optimizeModel()
                if frame_idx % self.params['target_net_sync'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())


    def getDistances(self):
        object_position = self.env.sim.data.get_site_xpos('object0')
        gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        return object_position[:2] - gripper_position[:2]


    def close(self, count=200):
        for _ in range(count):
            self.env.step([0, 0, 0, -1])
        self.renderalot()

    def open(self, count=200):
        for _ in range(count):
            self.env.step([0, 0, 0, 1])
        self.renderalot()

    def drop(self, count=30):
        for _ in range(count):
            self.env.step([0, 0, -1, 0])
        self.renderalot(5)

    def rise(self, count=15):
        for _ in range(count):
            self.env.step([0, 0, 1, 0])
        self.renderalot(5)

    def getFingerWidth(self):
        right = self.env.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        left = self.env.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')
        return right + left

    def slightOpen(self, amount=1.):
        start = self.getFingerWidth()
        self.env.step([0, 0, 0, amount])
        end = self.getFingerWidth()
        print(end - start)


    def renderalot(self, count=10):
        for _ in range(count):
            self.env.render()


    def wait(self, count=200):
        for _ in range(count):
            self.env.step([0,0,0,0])
            self.env.render()

    def move(self, movement, count=200):
        for _ in range(count):
            self.env.step(movement)
            self.env.render()



    def grabBlock(self):
        object_position = self.env.sim.data.get_site_xpos('object0')
        gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.rise()
        while gripper_position[0] > object_position[0]:
            self.env.step([-1, 0, 0, 0])
            self.env.render()
        while gripper_position[0] < object_position[0]:
            self.env.step([1, 0, 0, 0])
            self.env.render()

        while gripper_position[1] > object_position[1]:
            self.env.step([0, -1, 0, 0])
            self.env.render()
        while gripper_position[1] < object_position[1]:
            self.env.step([0, 1, 0, 0])
            self.env.render()

        self.open()
        self.drop()
        self.drop()
        # self.close()
        self.close(100)
        self.closing = True


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

def seed():
    seed = random.randrange(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    gpu_num = parser.parse_args().gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    cleanup()
    print('Creating Trainer')
    trainer = Trainer()
    print('Trainer Initialized')
    trainer.train()





