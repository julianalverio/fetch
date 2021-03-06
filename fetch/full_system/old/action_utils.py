import torch
import random
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from collections import namedtuple
import torch.optim as optim
import cv2
import copy

np.random.seed(5)
random.seed(5)
torch.backends.cudnn.deterministic = True
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)


TRANSITION = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, device, hyperparams):
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
            nn.Linear(512, num_actions)
        )

        self.memory = ReplayMemory(TRANSITION, hyperparams['memory_size'])
        self.epsilon_tracker = EpsilonTracker()
        self.optimizer = optim.Adam(self.parameters(), lr=hyperparams['learning_rate'])
        self.hyperparams = hyperparams
        self.state = None
        self.opening = False
        self.closing = False
        self.num_actions = num_actions
        self.env = None
        self.gripper_position = None
        self.object_position = None
        self.object1_position = None
        self.initial_object_position = None
        self.initial_gripper_position = None

        # some useful constants
        self.target_height = 0.47  # get the block at least this high
        self.x_threshold = 0.01143004  # to be prepared to grip, gripper x must be no further away than this
        self.y_threshold = 0.01121874  # to be prepared to grip, gripper y must be no further away than this
        self.z_threshold = 0.435  # to be prepared to grip, gripper z must be no higher than this
        self.finger_threshold = 0.047  # in order to grip the block your fingers must be at least this narrow
        # self.previous_height = self.initial_object_position[2]  # for negative reward when you decrease in height
        self.height_threshold = 0.58  # to have lifted the block, you must be higher than this
        self.drop_height = 0.45  # when putting down an object, you can be no higher than this


    def initializeState(self, env):
        self.state = self.preprocess(env.render(mode='rgb_array'))
        self.env = env
        self.gripper_position = self.env.sim.data.get_site_xpos('robot0:grip')
        self.object_position = self.env.sim.data.get_site_xpos('object0')
        self.object1_position = self.env.sim.data.get_site_xpos('object1')
        self.initial_object_position = copy.deepcopy(self.object_position)
        self.initial_gripper_position = copy.deepcopy(self.gripper_position)


    def forward(self, x, task):
        import pdb; pdb.set_trace()
        x = self.conv(x).view(x.size()[0], -1)
        x = torch.cat(x, torch.tensor(task))
        return self.fc(x)


    def optimizeModel(self, target_net):
        transitions = self.memory.sample(self.hyperparams['batch_size'])
        batch = TRANSITION(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(list(batch.state))
        action_batch = torch.cat(list(batch.action))
        reward_batch = torch.cat(list(batch.reward))
        import pdb; pdb.set_trace()
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


    def doAction(self, task):
        if random.random() < self.epsilon_tracker.epsilon():
            action = torch.tensor([random.randrange(self.num_actions)], device=self.device)
        else:
            action = torch.argmax(self(self.state, task), dim=1).to(self.device)

        self.env.step(self.convertAction(action))

        next_state = self.preprocess(self.env.render(mode='rgb_array'))
        reward, done = self.getReward(task)  # only done when you finish the task; not when episode times out
        if done:
            next_state = None

        self.memory.push(self.state, action, torch.tensor([reward], device=self.device), next_state)
        self.state = next_state
        return reward, done

    '''
    Task 1: Grab
    Task 2: Lift
    Task 3: Put down
    Task 4: Stack
    '''
    def getReward(self, task):
        # if the block falls off the table
        if self.initial_object_position[2] - self.object_position[2] > 0.1:
            return -1., True
        if random.random() > 0.999:
            import pdb; pdb.set_trace()
        if task == 1:
            if self.validGrip():
                return 1., True
            return 0., False

        if task == 2:
            if not self.validGrip():
                return -1., True
            if self.gripper_position[2] > self.height_threshold:
                return 1., True
            else:
                return 0., False

        if task == 3:
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] < self.drop_height:
                return 1., True
            if self.gripper_position[2] - self.object_position[2] >= 0.025 \
                    and self.gripper_position[2] >= self.drop_height:
                return -1., True
            else:
                return 0., False

        if task == 4:
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
        z_difference = self.gripper_position[2] - self.object_position[2]
        return x_difference <= self.x_threshold \
               and y_difference <= self.y_threshold \
               and 0 > z_difference <= 0.025 \
               and self.getFingerWidth() < self.finger_threshold \
               and self.closing

    def getFingerWidth(self):
        right = self.env.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint')
        left = self.env.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint')
        return right + left


    # Take an action and encode it into a length-4 vector that you call env.step(vector) on
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
        if action.item() == 10:
            return movement
        if action.item() % 2 == 0:
            movement[action.item() // 2] += 1
        else:
            movement[action.item() // 2] -= 1
        if self.opening:
            movement[-1] = 1
        elif self.closing:
            movement[-1] = -1
        return movement

    def preprocess(self, state):
        state = state[230:435, 50:460]
        state = cv2.resize(state, (state.shape[1]//2, state.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)/256
        state = np.swapaxes(state, 0, 2)
        return torch.tensor(state, device=self.device).unsqueeze(0)


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



