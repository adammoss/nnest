import numpy as np
from collections import deque


class Buffer(object):
    def __init__(self, max_size=1000):
        self.data = deque()
        self.max_size = max_size

    def insert(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) == 1:
            data = np.expand_dims(data, 0)
        for _ in range(len(self.data) - self.max_size + data.shape[0]):
            if len(self.data) > 0:
                self.data.popleft()
        for d in data[:self.max_size]:
            self.data.append(d)

    def __call__(self, batch_size=None):
        return np.array(self.data)
