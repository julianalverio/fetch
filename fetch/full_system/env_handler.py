#!/usr/bin/env python3
import random
import copy
import sys
sys.path.insert(0, '/storage/jalverio/venv/fetch/fetch/full_system')
from gym.envs.robotics import fetch_env
from gym.wrappers.time_limit import TimeLimit


class EnvHandler(object):
    def __init__(self, dqn):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        self.env = fetch_env.FetchEnv('fetch/pick_and_place.xml', has_object=True, block_gripper=False, n_substeps=20,
                                 gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
                                 obj_range=0.15, target_range=0.15, distance_threshold=0.05,
                                 initial_qpos=initial_qpos, reward_type='sparse')
        self.env = TimeLimit(self.env).unwrapped
        self.gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.opening = False
        self.closing = False
        self.dqn = dqn

    def getEnv(self):
        return self.env

    def reset(self, task=0):
        self.env.reset()
        self.env.render()
        self.env.sim.nsubsteps = 2
        if task == 2:
            self.resetSceneForPickUp()
        elif task == 3:
            self.resetSceneForPutDown()

        self.dqn.opening = self.opening
        self.dqn.closing = self.closing
        self.dqn.setState(self.env.render(mode='rgb_array'))
        self.dqn.object_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object0'))
        self.dqn.object1_position = copy.deepcopy(self.env.sim.data.get_site_xpos('object1'))



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



    ####### Some Useful Methods

    def move(self, movement, count=0):
        for _ in range(count):
            self.env.step(movement)
            self.env.render()

    def open(self, count=30):
        for _ in range(count):
            self.env.step([0, 0, 0, 1])
        self.renderalot()

    def renderalot(self, count=6):
        for _ in range(count):
            self.env.render()

    def drop(self, count=30):
        for _ in range(count):
            self.env.step([0, 0, -1, 0])
        self.renderalot(5)

    def close(self, count=30):
        for _ in range(count):
            self.env.step([0, 0, 0, -1])
        self.renderalot()
