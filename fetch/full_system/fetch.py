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
import sys
sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/full_system')
from gym.envs.robotics import fetch_env
from gym.wrappers.time_limit import TimeLimit
from buffer_prioritized import PrioritizedReplayBuffer as Memory

NUM_EPISODES = 3000
MAX_ITERATIONS = 700


HYPERPARAMS = {
        'replay_size':      200,  # normally 8k
        'replay_initial':   7900,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
}

class Dueling_DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Dueling_DQN, self).__init__()
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
            nn.Linear(in_features=conv_out_size + 1, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(in_features=conv_out_size + 1, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )


    def forward(self, x, task):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, task], dim=1)

        adv = self.fc_adv(x)
        val = self.fc_val(x).expand(x.size(0), self.num_actions)
        return val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)


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
            nn.Linear(conv_out_size + 1, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )


    def forward(self, x, task):
        # check that everything here works
        x = self.conv(x).view(x.size()[0], -1)
        x = torch.cat([x, task], dim=1)
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

#
# class ReplayMemory(object):
#     def __init__(self, capacity, transition):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#         self.transition = transition
#
#     def push(self, *args):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = self.transition(*args)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
#
#     def showCapacity(self):
#         print('Buffer Capacity:', len(self.memory) * 1. / self.capacity)



class Trainer(object):
    def __init__(self):
        self.params = HYPERPARAMS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = self.makeEnv()
        self.best_score = float('-inf')

        #Actions:
        # 0 -- increment X
        # 1 -- decrement X
        # 2 -- increment Y
        # 3 -- decrement Y
        # 4 -- increment Z
        # 5 -- decrement Z
        # 6 -- open gripper and keep it open
        # 7 -- close gripper and keep it closed

        self.initial_gripper_position = copy.deepcopy(self.env.sim.data.get_site_xpos('robot0:grip'))

        self.action_space = 8
        self.observation_space = [3, 102, 205]
        self.policy_net = Dueling_DQN(self.observation_space, self.action_space).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.epsilon_tracker = EpsilonTracker(self.params)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.params['learning_rate'])
        self.reward_tracker0 = RewardTracker()
        self.reward_tracker1 = RewardTracker()
        self.reward_tracker2 = RewardTracker()
        self.reward_tracker3 = RewardTracker()
        self.step_tracker0 = RewardTracker()
        self.step_tracker1 = RewardTracker()
        self.step_tracker2 = RewardTracker()
        self.step_tracker3 = RewardTracker()
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'task'))
        # self.memory = ReplayMemory(self.params['replay_size'], self.transition)
        self.memory = Memory(self.params['replay_size'], self.transition, alpha=0.8, beta_0=0.2, beta_delta=0.0004)
        self.tb_writer = SummaryWriter('results')
        self.gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.object_position = self.env.sim.data.get_site_xpos('object0')
        self.object1_position = self.env.sim.data.get_site_xpos('object1')

        self.stage_count = 0
        self.target_height = 0.47  # get the block at least this high
        self.x_threshold = 0.019  # to be prepared to grip, gripper x must be no further away than this
        self.y_threshold = 0.019  # to be prepared to grip, gripper y must be no further away than this
        self.z_threshold = 0.435  # to be prepared to grip, gripper z must be no higher than this

        self.closing = False
        self.opening = False

        # state variables
        self.task0_episode_counter = 0
        self.task1_episode_counter = 0
        self.task2_episode_counter = 0
        self.task3_episode_counter = 0
        self.task = 0.
        self.stage = 0

        self.finger_threshold = 0.049  # in order to grip the block your fingers must be at least this narrow
        self.height_threshold = 0.48  # to have lifted the block, the block must be higher than this
        self.drop_height = 0.44  # when putting down an object, you can be no higher than this
        self.preparatory_height = 0.48057


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

    def resetSceneForPutDown(self):
        self.resetSceneForPickUp()
        # 0.58 is the height threshold for picking something up
        while self.gripper_position[2] < 0.58:
            self.env.step([0, 0, 1, -1])
        [self.env.render() for _ in range(4)]
        self.closing = True
        self.opening = False

    def getGripperWidth(self):
        right_finger = self.env.sim.data.get_body_xipos('robot0:r_gripper_finger_link')[2]
        left_finger = self.env.sim.data.get_body_xipos('robot0:l_gripper_finger_link')[2]
        return abs(right_finger - left_finger)


    def reset(self):
        self.env.reset()
        self.env.render()
        self.env.sim.nsubsteps = 2
        self.initial_object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        self.initial_object1_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object1'))
        if self.task == 1.:
            self.resetSceneForPickUp()
        if self.task in (2., 3.):
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
        if action.item() % 2 == 0:
            movement[action.item() // 2] += 1
        else:
            movement[action.item() // 2] -= 1
        if action.item() == 6:
            self.opening = True
            self.closing = False
        elif action.item() == 7:
            self.opening = False
            self.closing = True
        if self.opening:
            movement[-1] = 1
        elif self.closing:
            movement[-1] = -1
        return movement

    def openGripper(self):
        while self.getFingerWidth() < 0.1:
            self.env.step([0, 0, 0, 1])
        self.env.render()

    def closeGripper(self):
        while self.getFingerWidth() > 0.0001:
            self.env.step([0, 0, 0, -1])
        self.env.render()


    # for when stage_count == 0
    def addExperience(self):
        task_tensor = torch.tensor([self.task], device=self.device).unsqueeze(0)
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.action_space)], device=self.device)
        else:
            action = torch.argmax(self.policy_net(self.state, task_tensor), dim=1).to(self.device)
        action_converted = self.convertAction(action)
        if action.item() in (6, 7):
            [self.env.step(action_converted) for _ in range(8)]
        self.env.step(action_converted)
        next_state = self.preprocess(self.env.render(mode='rgb_array'))
        reward, done = self.getReward()
        if done:
            next_state = None

        self.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state, task_tensor)

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
        # transitions = self.memory.sample(self.params['batch_size'])
        transitions, ISWeights, tree_idx = self.memory.sample(self.batch_size)
        ISWeights = torch.tensor(ISWeights, device=self.device)
        batch = self.transition(*zip(*transitions))
        next_states = batch.next_state
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_tasks = torch.cat([batch.task[i] for i in range(self.params['batch_size']) if next_states[i] is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        task_batch = torch.cat(list(batch.task))
        state_action_values = self.policy_net(state_batch, task_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.params['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_tasks).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.params['gamma']) + reward_batch
        abs_errors = abs(expected_state_action_values.unsqueeze(1) - state_action_values)
        loss = torch.sum((abs_errors ** 2) * ISWeights)
        self.memory.batch_update(tree_idx, abs_errors.detach().cpu().numpy() + 1e-6)
        # loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
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
        if self.initial_object1_position[2] - self.object_position[2] > 0.1:
            print('I KNOCKED THE BLACK BLOCK OFF THE TABLE')
            return -3., True
        if self.initial_object1_position[2] - self.object1_position[2] > 0.1:
            print('I KNOCKED THE RED BLOCK OFF THE TABLE')
            return -3., True
        if self.task == 0.:
            target_position = copy.deepcopy(self.object_position)
            target_position[2] = self.preparatory_height
            distance = np.linalg.norm(target_position - self.gripper_position)

            # if it gets to the target position
            x_check = target_position[0] - self.gripper_position[0] < self.x_threshold
            y_check = target_position[1] - self.gripper_position[1] < self.y_threshold
            z_check = target_position[2] - self.gripper_position[2] < 0.04

            if self.stage == 0 and x_check and y_check and z_check:
                self.stage = 1
                return 0.25, False
            # if self.stage == 1 and x_check and y_check:
            #     reward = 0.25 + self.preparatory_height - self.


            else:
                return -2*distance, False

            # if self.validGrip():
            #     print('I GOT A VALID GRIP!')
            #     return 1., True
            # if self.gripPrep():
            #     print('PREPARED TO GRIP')
            #     return 0.5, False
            # return reward, False

        if self.task == 1.:
            if not self.validGrip():
                return -1., True
            if self.object_position[2] > self.height_threshold:
                return 1., True
            else:
                progress = self.object_position[2] - self.initial_object_position[2]
                total_distance_needed = self.drop_height - self.initial_object_position[2]
                return progress / total_distance_needed, False

        if self.task == 2.:
            dropped = self.dropped()
            if dropped and self.gripper_position[2] < self.drop_height:
                return 1., True
            if dropped and self.gripper_position[2] >= self.drop_height:
                return -1., True

            progress = self.height_threshold - self.gripper_position[2]
            total_distance_needed = self.height_threshold - self.drop_height
            return progress / total_distance_needed, False

        if self.task == 3.:
            x_check = abs(self.object_position[0] - self.object_position[0]) < 0.02
            y_check = abs(self.object_position[1]) - self.object_position[1] < 0.02
            dropped = self.dropped()
            if dropped:
                if self.object_position[2] <= self.object1_position[2] + 0.02 * 2 and x_check and y_check:
                    return 1., True
                else:
                    return -1., False

            target = self.object1_position + np.array([0, 0, 0.03])
            reward = -np.linalg.norm(self.object_position - target)
            return reward, False


    def validGrip(self):
        x_check = abs(self.object_position[0] - self.gripper_position[0]) <= self.x_threshold
        y_check = abs(self.object_position[1] - self.gripper_position[1]) <= self.y_threshold
        z_check = 0 > (self.gripper_position[2] - self.object_position[2]) <= 0.025
        return x_check and y_check and z_check and self.closing and self.getFingerWidth() < self.finger_threshold

    def dropped(self):
        x_check = abs(self.object_position[0] - self.gripper_position[0]) <= self.x_threshold
        y_check = abs(self.object_position[1] - self.gripper_position[1]) <= self.y_threshold
        z_check = 0 > (self.gripper_position[2] - self.object_position[2]) <= 0.025
        return not (x_check and y_check and z_check and self.getFingerWidth() < self.finger_threshold)

    def gripPrep(self):
        x_check = abs(self.object_position[0] - self.gripper_position[0]) <= self.x_threshold
        y_check = abs(self.object_position[1] - self.gripper_position[1]) <= self.y_threshold
        z_check = 0 > (self.gripper_position[2] - self.object_position[2]) <= 0.025
        return x_check and y_check and z_check


    def train(self):
        frame_idx = 0
        for episode in range(NUM_EPISODES):
            self.task = float(random.randrange(0, 4))
            print('Task:', self.task)
            self.reset()
            for iteration in range(MAX_ITERATIONS):
                # execute one move
                frame_idx += 1
                reward, done = self.addExperience()
                if reward < -1:
                    reward = -1

                # are we  done prefetching?
                # if not self.memory.tree.done_prefetching:
                if len(self.memory) < self.params['replay_initial']:
                    if done:
                        break
                    continue
                # if len(self.memory) == self.params['replay_initial']:
                #     print("Done Prefetching.")
                #     break
                if iteration == MAX_ITERATIONS - 1:
                    done = True

                # is this round over?
                if done:
                    print('Episode Completed:', episode)
                    print('Task: %s' % self.task)
                    print('Score for Epoch %s' % reward)
                    print('Steps in this episode:', iteration)
                    print('Epsilon:', self.epsilon_tracker.percievedEpsilon())
                    score = self.reward_tracker0.meanScore() + self.reward_tracker1.meanScore() + self.reward_tracker2.meanScore() + self.reward_tracker3.meanScore()
                    if score > self.best_score:
                        torch.save(self.policy_net, '%s.path' % episode)
                        print('Saved!')
                    if self.task == 0.:
                        self.step_tracker0.add(iteration)
                        self.reward_tracker0.add(reward)
                        average_score = self.reward_tracker0.meanScore()
                        self.tb_writer.add_scalar('Task 0 Score', reward, self.task0_episode_counter)
                        self.tb_writer.add_scalar('Task 0 Average Score', self.reward_tracker0.meanScore(),
                                                  self.task0_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 0', iteration, self.task0_episode_counter)
                        self.tb_writer.add_scalar('Average steps for Task 0', self.step_tracker0.meanScore(), self.task0_episode_counter)
                        self.task0_episode_counter += 1
                    if self.task == 1.:
                        self.step_tracker1.add(iteration)
                        self.reward_tracker1.add(reward)
                        average_score = self.reward_tracker1.meanScore()
                        self.tb_writer.add_scalar('Task 1 Score', reward, self.task1_episode_counter)
                        self.tb_writer.add_scalar('Task 1 Average Score', self.reward_tracker1.meanScore(),
                                                  self.task1_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 1', iteration, self.task1_episode_counter)
                        self.tb_writer.add_scalar('Average steps for Task 1', self.step_tracker1.meanScore(), self.task1_episode_counter)
                        self.task1_episode_counter += 1
                    if self.task == 2.:
                        self.step_tracker2.add(iteration)
                        self.reward_tracker2.add(reward)
                        average_score = self.reward_tracker2.meanScore()
                        self.tb_writer.add_scalar('Task 2 Score', reward, self.task2_episode_counter)
                        self.tb_writer.add_scalar('Task 2 Average Score', self.reward_tracker2.meanScore(),
                                                  self.task2_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 2', iteration, self.task2_episode_counter)
                        self.task2_episode_counter += 1
                        self.tb_writer.add_scalar('Average steps for Task 2', self.step_tracker2.meanScore(), self.task2_episode_counter)
                    if self.task == 3.:
                        self.step_tracker3.add(iteration)
                        self.reward_tracker3.add(reward)
                        average_score = self.reward_tracker3.meanScore()
                        self.tb_writer.add_scalar('Task 3 Score', reward, self.task3_episode_counter)
                        self.tb_writer.add_scalar('Task 3 Average Score', self.reward_tracker3.meanScore(),
                                                  self.task3_episode_counter)
                        self.tb_writer.add_scalar('Steps in episode for Task 3', iteration, self.task3_episode_counter)
                        self.tb_writer.add_scalar('Average steps for Task 3', self.step_tracker3.meanScore(), self.task3_episode_counter)
                        self.task3_episode_counter += 1

                    print('Average score: %s' % average_score)
                    self.tb_writer.add_scalar('Epsilon', self.epsilon_tracker.percievedEpsilon(), episode)
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
    if os.path.isdir('results_continuous'):
        shutil.rmtree('results_continuous')
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





