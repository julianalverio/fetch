#!/usr/bin/env python3
import random
import numpy as np
import torch
import copy
import argparse
from tensorboardX import SummaryWriter
import shutil
from env_handler import EnvHandler
import os
from action_utils import DQN
import sys
sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/full_system')


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
        'memory_size':      32000,
        'replay_initial':   7900,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5 * 4,
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
    def __init__(self):
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

        self.reward_tracker1 = RewardTracker()
        self.reward_tracker2 = RewardTracker()
        self.reward_tracker3 = RewardTracker()
        self.reward_tracker4 = RewardTracker()
        self.tb_writer = SummaryWriter('results')

        # keeping track of physical objects
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        self.initial_gripper_position = copy.deepcopy(self.env.sim.data.get_site_xpos('robot0:grip'))

        # state variables
        self.task1_episode_counter = 0
        self.task2_episode_counter = 0
        self.task3_episode_counter = 0
        self.task4_episode_counter = 0

        # some useful constants
        self.target_height = 0.47  # get the block at least this high
        self.x_threshold = 0.01143004  # to be prepared to grip, gripper x must be no further away than this
        self.y_threshold = 0.01121874  # to be prepared to grip, gripper y must be no further away than this
        self.z_threshold = 0.435  # to be prepared to grip, gripper z must be no higher than this
        self.finger_threshold = 0.047  # in order to grip the block your fingers must be at least this narrow
        self.previous_height = self.initial_object_position[2]  # for negative reward when you decrease in height
        self.height_threshold = 0.58  # to have lifted the block, you must be higher than this
        self.drop_height = 0.45  # when putting down an object, you can be no higher than this

    def train(self):
        frame_idx = 0
        for episode in range(NUM_EPISODES):
            print('Starting Episode:%s' % episode)
            task = random.randrange(1, 5)
            self.env_handler.reset(task)
            for iteration in range(MAX_STEPS):
                # execute one move
                reward, done = self.policy_net.doAction(task)

                if len(self.env_handler.dqn.memory) < self.params['replay_initial']:
                    # if done:
                    #     break
                    continue

                # is this round over?
                if done:
                    print('Episode Completed:', episode)
                    print('Task: %s' % task)
                    print('Score for Epoch %s' % reward)
                    print('Steps in this episode:', iteration)
                    print('Epsilon:', self.policy_net.epsilon_tracker.percievedEpsilon())
                    if task == 1:
                        self.reward_tracker1.add(reward)
                        average_score = self.reward_tracker1.meanScore()
                        self.tb_writer.add_scalar('Task 1 Score', reward, self.task1_episode_counter)
                        self.tb_writer.add_scalar('Task 1 Average Score', self.reward_tracker1.meanScore(), self.task1_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 1', iteration, self.task1_episode_counter)
                        self.task1_episode_counter += 1
                    if task == 2:
                        self.reward_tracker2.add(reward)
                        average_score = self.reward_tracker2.meanScore()
                        self.tb_writer.add_scalar('Task 2 Score', reward, self.task2_episode_counter)
                        self.tb_writer.add_scalar('Task 2 Average Score', self.reward_tracker2.meanScore(), self.task2_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 2', iteration, self.task2_episode_counter)
                        self.task2_episode_counter += 1
                    if task == 3:
                        self.reward_tracker3.add(reward)
                        average_score = self.reward_tracker3.meanScore()
                        self.tb_writer.add_scalar('Task 3 Score', reward, self.task3_episode_counter)
                        self.tb_writer.add_scalar('Task 3 Average Score', self.reward_tracker3.meanScore(), self.task3_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 3', iteration, self.task3_episode_counter)
                        self.task3_episode_counter += 1
                    if task == 4:
                        self.reward_tracker4.add(reward)
                        average_score = self.reward_tracker4.meanScore()
                        self.tb_writer.add_scalar('Task 4 Score', reward, self.task4_episode_counter)
                        self.tb_writer.add_scalar('Task 4 Average Score', self.reward_tracker4.meanScore(), self.task4_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 4', iteration, self.task4_episode_counter)
                        self.task4_episode_counter += 1

                    print('Average score: %s' % average_score)
                    self.tb_writer.add_scalar('Epsilon', self.policy_net.epsilon_tracker.percievedEpsilon(), episode)
                    break

                self.policy_net.optimizeModel(self.target_net)
                frame_idx += 1
                if frame_idx % self.params['target_net_sync'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())


    # METHODS FOR TESTING

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



def cleanup():
    if os.path.isdir('results'):
        shutil.rmtree('results')
    csv_txt_files = [x for x in os.listdir('.') if '.TXT' in x or '.csv' in x]
    for csv_txt_file in csv_txt_files:
        os.remove(csv_txt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=int)
    args = parser.parse_args()
    gpu_num = args.gpu
    print('GPU:', gpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    seed = random.randrange(0, 100)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('cleaning up...')
    cleanup()
    print('Creating Trainer Object')
    trainer = Trainer()
    print('Trainer Initialized.')
    trainer.train()





