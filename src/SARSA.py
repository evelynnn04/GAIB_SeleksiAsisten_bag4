import numpy as np
import random

# Class ini khusus buat soal seleksi GAIB 
class SARSA:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=10):
        '''
        alpha: learning ratenya 
        gamma: discount factor, range-nya 0-1, semakin deket 1 berarti agent semakin mertimbangin reward di masa depan
        epsilon: ini kaya treshholdnya gitu, let say epsilon = 0,1 berarti 10% eksplorasi (random) & 90% eksploitasi (baca dari q table)
        episodes: banyak iterasinya 
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((10, 2)) 

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1)
        else:
            return np.argmax(self.q_table[state])

    def take_action(self, state, action):
        # Move left
        if action == 0:  
            next_state = max(0, state - 1)
        # Move right
        else:  
            next_state = min(9, state + 1) 

        if next_state == 0:  # jatoh
            reward = -100
            next_state = 3  
        elif next_state == 9:  # makan apel
            reward = 100
            next_state = 3  
        else:
            reward = -1

        return next_state, reward

    def learn(self):
        for episode in range(self.episodes):
            state = 2
            action = self.choose_action(state)  
            total_reward = 0

            while True:
                next_state, reward = self.take_action(state, action)
                next_action = self.choose_action(next_state) 

                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
                )
                # print(f"q_table[{state}, {action}] += {self.alpha} * ({reward} + {self.gamma} * q_table[{next_state}, {next_action}] - q_table[{state}, {action}]) = {self.q_table[state, action]}")

                state = next_state
                action = next_action
                total_reward += reward

                if total_reward >= 500 or total_reward <= -200:
                    break

if __name__ == '__main__':
    model_sarsa = SARSA(episodes=10)
    model_sarsa.learn()
    state = 2
    path = []
    total_reward = 0

    while True:
        action = np.argmax(model_sarsa.q_table[state])
        state, reward = model_sarsa.take_action(state, action)
        path.append(state)
        total_reward += reward
        
        if total_reward >= 500 or total_reward <= -200:
            break

    print("Path: ")
    print(path)
    print("Q Table: ")
    print(model_sarsa.q_table)
