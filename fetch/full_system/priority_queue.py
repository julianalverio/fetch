import numpy as np
from collections import namedtuple


class SumTree(object):
    # capacity: number of leaf nodes that contain experiences
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.done_prefetching = False

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.data_pointer == 0 and not self.done_prefetching:
            self.done_prefetching = True
            print('Done pre-fetching.')

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity, batch_size=32, alpha=0.6, beta_0=0.4, beta_delta=0.001):
        self.alpha = alpha
        self.beta_0 = beta_0
        self.beta = beta_0
        self.beta_delta = beta_delta
        self.batch_size = batch_size
        self.absolute_error_upper = 1.
        self.small_delta = 0.01
        self.tree = SumTree(capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

    def push(self, *args):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        self.tree.add(max_priority, self.transition(*args))

    """
    First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    Then a value is uniformly sampled from each range
    We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    Then, we calculate IS weights for each minibatch element
    """
    def sample(self):
        minibatch = []
        batch_idx = np.empty((self.batch_size,), dtype=np.int32)
        batch_ISWeights = np.empty((self.batch_size, 1), dtype=np.float32)

        segment_width = self.tree.total_priority / self.batch_size
        self.beta = np.min([1., self.beta + self.beta_delta])

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * self.batch_size) ** (-self.beta)

        for i in range(self.batch_size):
            a, b = segment_width * i, segment_width * (i + 1)
            priority_value = np.random.uniform(a, b)

            index, priority, experience = self.tree.get_leaf(priority_value)
            sampling_probabilities = priority / self.tree.total_priority
            batch_ISWeights[i, 0] = np.power(self.batch_size * sampling_probabilities, -self.beta) / max_weight

            batch_idx[i] = index
            # check on this
            minibatch.append(experience)

        return batch_idx, minibatch, batch_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.small_delta
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
