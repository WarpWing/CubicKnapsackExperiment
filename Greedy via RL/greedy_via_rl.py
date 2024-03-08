import numpy as np
import random
from collections import deque

class CubicKnapsack:
    def __init__(self, n, density, seed):
        self.n = n
        self.density = density
        self.seed = seed
        self.weights = []
        self.values = []
        self.max_weight = 0
        self.generate_instance()

    def generate_instance(self):
        random.seed(self.seed)
        self.weights = [random.randint(1, 1000) for _ in range(self.n)]
        self.values = [random.randint(1, 1000) for _ in range(self.n)]
        self.max_weight = int(self.n * self.density * 1000)

    def reset(self):
        return np.zeros(self.n, dtype=int)

    def step(self, action):
        state = action.copy()
        weight = sum(self.weights[i] for i in range(self.n) if state[i])
        value = sum(self.values[i] for i in range(self.n) if state[i])
        if weight > self.max_weight:
            reward = 0
        else:
            reward = value
        done = True
        return state, reward, done

    @property
    def state_shape(self):
        return self.n,

    @property
    def num_actions(self):
        return 2 ** self.n

class GreedyDQN:
    def __init__(self, input_shape, num_actions, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros(num_actions)

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action_values = self.Q.copy()
            for i in range(self.num_actions):
                action = np.binary_repr(i, width=self.input_shape[0])
                action = np.array([int(x) for x in action])
                weight = sum(problem.weights[j] for j in range(self.input_shape[0]) if action[j])
                if weight > problem.max_weight:
                    action_values[i] = -np.inf
            action = np.argmax(action_values)
        return np.binary_repr(action, width=self.input_shape[0])

    def train(self, state, action, reward, next_state, done):
        action_idx = int(action, 2)
        next_state_values = self.Q.copy()
        for i in range(self.num_actions):
            next_action = np.binary_repr(i, width=self.input_shape[0])
            next_action = np.array([int(x) for x in next_action])
            weight = sum(problem.weights[j] for j in range(self.input_shape[0]) if next_action[j])
            if weight > problem.max_weight:
                next_state_values[i] = -np.inf
        next_best_action = np.argmax(next_state_values)
        next_best_value = self.Q[next_best_action]
        if done:
            self.Q[action_idx] = reward
        else:
            self.Q[action_idx] = max(reward, next_best_value)
        self.epsilon *= self.epsilon_decay

# Initialize the cubic knapsack problem
problem = CubicKnapsack(n=50, density=0.25, seed=2024)

# Create the greedy DQN agent
dqn = GreedyDQN(input_shape=problem.state_shape, num_actions=problem.num_actions)

# Training loop
num_episodes = 10000
for episode in range(num_episodes):
    state = problem.reset()
    done = False
    while not done:
        action = dqn.choose_action(state)
        action = np.array([int(x) for x in action])
        next_state, reward, done = problem.step(action)
        dqn.train(state, action, reward, next_state, done)
        state = next_state

# Evaluate the trained model
average_reward = evaluate_model(dqn, problem)
print(f"Average reward: {average_reward}")
