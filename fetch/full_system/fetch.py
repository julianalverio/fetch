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
from tensorboardX import SummaryWriter
import shutil
from env_handler import EnvHandler

import os

import sys
# sys.path.pop(0)
# import gym
sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/full_system')
from gym.envs.robotics import fetch_env
from gym import utils
from gym.wrappers.time_limit import TimeLimit
from action_utils import DQN
from PIL import Image

# Actions:
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
# 10 -- no-op


NUM_EPISODES = 3000
MAX_STEPS = 1500   # max steps in an episode


HYPERPARAMS = {
        'memory_size':      8000,
        'replay_initial':   7900,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5 * 2,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'max_steps':        1500

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



class Trainer(object):
    def __init__(self, task_num):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DQN/action-related stuff
        self.action_space = 11
        self.observation_space = [3, 102, 205]
        self.policy_net = DQN(self.observation_space, self.action_space, self.device, self.params).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        # helper classes
        self.env_handler = EnvHandler(self.policy_net)
        self.env = self.env_handler.getEnv()
        self.env_handler.reset()
        self.reward_tracker = RewardTracker()
        self.tb_writer = SummaryWriter('results')

        # keeping track of physical objects
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        self.initial_gripper_position = copy.deepcopy(self.env.sim.data.get_site_xpos('robot0:grip'))
        self.gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.object_position = self.env.sim.data.get_site_xpos('object0')
        self.object1_position = self.env.sim.data.get_site_xpos('object1')

        # state variables
        self.episode = 0
        self.score = 0
        self.task = task_num
        self.movement_count = 0
        self.ready_to_drop = False

        # some useful constants
        self.target_height = 0.47  # get the block at least this high
        self.x_threshold = 0.01143004  # to be prepared to grip, gripper x must be no further away than this
        self.y_threshold = 0.01121874  # to be prepared to grip, gripper y must be no further away than this
        self.z_threshold = 0.435  # to be prepared to grip, gripper z must be no higher than this
        self.finger_threshold = 0.047  # in order to grip the block your fingers must be at least this narrow
        self.previous_height = self.initial_object_position[2]  # for negative reward when you decrease in height
        self.height_threshold = 0.58  # to have lifted the block, you must be higher than this
        self.drop_height = 0.45  # when putting down an object, you can be no higher than this


    def getGripperWidth(self):
        right_finger = self.env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')[2]
        left_finger = self.env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')[2]
        return abs(right_finger - left_finger)

    def openGripper(self):
        while self.getFingerWidth() < 0.1:
            self.env.step([0, 0, 0, 1])
        self.env.render()

    def closeGripper(self):
        while self.getFingerWidth() > 0.0001:
            self.env.step([0, 0, 0, -1])
        self.env.render()


    '''
    Task 1: Touch the block, discrete reward
    Task 2: Touch the block, continuous reward
    Task 3: Annealing Binary Reward. Done if goal sphere entered or success >= 90%
    *** All Tasks Below Are To Pick Up The Block ***
    Task 4: 1 stage continuous reward
    Task 5: 2 stage continuous reward
    Task 6: 3 stage continuous reward
    Task 7: 1 stage binary reward
    Task 8: 2 stage binary reward
    Task 9: 3 stage binary reward
    Task 10: 3 stage binary reward. Additional contrainsts
    
    For old xy reward function
    https://github.com/julianalverio/fetch/blob/ea076c97ec7e7e15fcd49136ae43188e231ffac6/fetch/old_experiments/gripper_enabled/prepare_to_grasp/fetch.py
    '''

    '''
    Task 1: Grab
    Task 2: Lift
    Task 3: Put down
    Task 4: Stack
    '''
    def getReward(self):
        done = False
        if self.task == 1:
            if self.validGrip():
                return 1., True
            return 0., False

        if self.task == 2:
            if not self.validGrip():
                return -1., True
            if self.gripper_position[2] > self.height_threshold:
                return 1., True
            else:
                return 0., False

        if self.task == 3:
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] < self.drop_height:
                return 1., True
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] >= self.drop_height:
                return -1., True
            else:
                return 0., False


        if self.task == 1:
            if np.linalg.norm(self.initial_object_position - self.object_position) > 1e-3:
                self.score += 1.
                return 1., True
            return 0., False

        if self.task == 2:
            distance = np.linalg.norm(self.gripper_position - self.object_position)
            reward = -distance
            if np.linalg.norm(self.initial_object_position - self.object_position) > 1e-3:
                reward += 10.
                self.score += 1.
                done = True
            return reward, done

        if self.task == 3:
            if self.remaining_anneals >= 1:
                if np.linalg.norm(self.gripper_position - self.object_position) < self.current_radius:
                    self.score = 1.
                    return 1., True
                else:
                    return -self.movement_count / 300., False
            elif np.linalg.norm(self.initial_object_position - self.bject_position) > 1e-3:
                print('DONE! MEAN SCORES: ', self.reward_tracker.meanScore())
                self.score = 1.
                return 10., True
            else:
                return -self.movement_count / 300., False

        # continuous reward as one task
        if self.task == 4:
            if not self.validGrip():
                return 0., False
            reward = self.object_position[2] - self.initial_object_position[2]
            if self.object_position[2] > self.target_height:
                return reward + 1, True
            return reward, False

        # 2 stage continuous reward
        if self.task == 5:
            reward = -self.movement_count / 1000.
            if self.stage_count == 0:
                distance = np.linalg.norm(self.gripper_position - self.object_position)
                reward -= distance
                if self.validGrip():
                    self.score = 1
                    self.stage_count = 1
                return reward, False
            elif self.stage_count == 1:
                if not self.validGrip():
                    self.stage_count = 0
                    return reward - 5., False
                reward += self.object_position[2] - self.initial_object_position[2]
                if self.object_position[2] > self.target_height:
                    self.score = 2
                    return reward, True
                return reward, False

        # 3 stage continuous reward
        if self.task == 6:
            reward = -self.movement_count / 1000.
            if self.stage_count == 0:
                distance = np.linalg.norm(self.gripper_position - self.object_position)
                reward -= distance
                if self.validGrip():
                    self.score = 1
                    self.stage_count = 1
                    return 5., False
                return reward, False
            if self.stage_count == 1:
                if not self.validGrip():
                    self.stage_count = 0
                    return reward - 5., False
                if self.getFingerWidth() <= 0.0508578 and self.closing:
                    self.score = 2
                    self.stage_count = 2
                    return 5., False
                else:
                    return reward, False
            if self.stage_count == 2:
                if not self.validGrip():
                    self.stage_count = 0
                    return reward - 5., False
                reward += self.object_position[2] - self.initial_object_position[2]
                if self.object_position[2] > self.target_height:
                    self.score = 3
                    return reward, True
                return reward, False

        # 1 stage binary reward
        if self.task == 7:
            if self.validGrip() and self.object_position[2] >= self.target_height:
                self.score = 1
                return 1., True
            else:
                return 0., False

        # 2-stage binary reward
        if self.task == 8:
            reward = -self.movement_count / 1000.
            if self.stage_count == 0:
                if self.validGrip():
                    self.score = 1
                    self.stage_count = 1
                    return 5., False
                else:
                    return reward, False
            if self.stage_count == 1:
                if not self.validGrip():
                    self.stage_count = 0
                    return -1., False
                if self.object_position[2] >= self.target_height:
                    self.score = 2
                    return 5., True
                return reward, False

        if self.task == 9:
            reward = -self.movement_count / 1000.
            if self.stage_count == 0:
                if self.validGrip():
                    self.score = 1
                    self.stage_count = 1
                    return 5., False
                else:
                    return reward, False
            if self.stage_count == 1:
                if not self.validGrip():
                    self.stage_count = 0
                    return -1., False
                if self.getFingerWidth() <= 0.0508578:
                    self.score = 2
                    self.stage_count = 2
                    return 5., False
                else:
                    return reward, False
            if self.stage_count == 2:
                if not self.validGrip():
                    self.stage_count = 0
                    return -1., False
                if self.object_position[2] >= self.target_height:
                    self.score = 2
                    return 5., True
                else:
                    return reward, False


    def validGrip(self):
        x_difference = abs(self.object_position[0] - self.gripper_position[0])
        y_difference = abs(self.object_position[1] - self.gripper_position[1])
        z_difference = self.gripper_position[2] - self.object_position[2]
        return x_difference <= self.x_threshold \
               and y_difference <= self.y_threshold \
               and 0 > z_difference <= 0.025 \
               and self.getFingerWidth() < self.finger_threshold \
               and self.closing


    def train(self):
        import pdb; pdb.set_trace()
        frame_idx = 0
        self.task = 1
        while True:
            frame_idx += 1
            # execute one move
            done = self.policy_net.doAction()

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
                print('Epsilon:', self.policy_net.epsilon_tracker.percievedEpsilon())

                self.tb_writer.add_scalar('Score for Epoch', self.score, self.episode)
                self.tb_writer.add_scalar('Perceived Mean Score', self.reward_tracker.meanScore(), self.episode)
                self.tb_writer.add_scalar('Actual Mean Score', np.mean(self.reward_tracker.rewards), self.episode)
                self.tb_writer.add_scalar('Remaining Anneals', self.remaining_anneals, self.episode)
                self.tb_writer.add_scalar('Steps in this Episode', self.movement_count, self.episode)
                self.tb_writer.add_scalar('Epsilon', self.policy_net.epsilon_tracker.percievedEpsilon(), self.episode)

                self.score = 0
                self.movement_count = 0
                self.episode += 1

                self.stage_count = 0
                print('Starting Episode:', self.episode)

            # if self.remaining_anneals > 0 and self.reward_tracker.meanScore() > 0.9:
            #     self.updateRewardRadius()
            # if self.remaining_anneals == 0 and self.reward_tracker.meanScore() == 1:
            #     return

            self.optimizeModel()
            if frame_idx % self.params['target_net_sync'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.episode == NUM_EPISODES:
                print("DONE WITH ALL EPISODES")
                return

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


    # minimum x: 1.0m
    # no maximum x
    # practical maximum x: 1.5
    # maximum y: 1.16
    # minimum y: 0.45
    # minimum z: 0.33
    # maximum z: 0.75
    # good height above the table: 0.47
    # def collectData(self):
    #     if os.path.isdir('dataset'):
    #         shutil.rmtree('dataset')
    #     os.mkdir('dataset')
    #     gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
    #     # go to starting position in back left corner above table
    #     while gripper_position[0] > 1.0:
    #         self.env.step([-1, 0, 0, 0])
    #         self.env.render()
    #     while gripper_position[1] > 0.45:
    #         self.env.step([0, -1, 0, 0])
    #         self.env.render()
    #     while gripper_position[2] < 0.72:
    #         self.env.step([0, 0, 1, 0])
    #         self.env.render()
    #     state = self.preprocessDataCollection(self.env.render(mode='rgb_array'))
    #     import pdb; pdb.set_trace()


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
    parser.add_argument('task', type=int)
    args = parser.parse_args()
    gpu_num = args.gpu
    task_num = args.task
    print('GPU:', gpu_num)
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
    trainer = Trainer(task_num)
    print('Trainer Initialized')
    print("Prefetching Now...")
    trainer.train()





