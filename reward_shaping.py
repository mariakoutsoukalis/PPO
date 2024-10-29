import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pandas as pd
import time


LR = 5e-6
GAMMA = 0.99
EPS_CLIP = 0.15
K_EPOCHS = 15
ENTROPY_COEFF = 0.05
UPDATE_INTERVAL = 500
HIDDEN_SIZE = 333
MIN_BATCH_SIZE = 200

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.actor = nn.Linear(HIDDEN_SIZE, action_dim)
        self.critic = nn.Linear(HIDDEN_SIZE, 1)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = self.fc(state)
        return self.actor(x), self.critic(x)

    def act(self, state):
        mu, _ = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist.sample(), dist, std

    def evaluate(self, state, action):
        mu, value = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        action_logprobs = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_logprobs, entropy, value

class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        states, actions, rewards, dones, next_states, log_probs = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + (1 - d) * GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        values = torch.cat([self.policy.evaluate(s.unsqueeze(0), a.unsqueeze(0))[2] for s, a in zip(states, actions)])
        advantages = returns - values.detach().squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(K_EPOCHS):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for i in range(0, len(indices), MIN_BATCH_SIZE):
                sampled_indices = indices[i:i + MIN_BATCH_SIZE]
                batch_states = states[sampled_indices]
                batch_actions = actions[sampled_indices]
                batch_log_probs = log_probs[sampled_indices]
                batch_returns = returns[sampled_indices]
                batch_advantages = advantages[sampled_indices]

                new_log_probs, entropy, values = self.policy.evaluate(batch_states, batch_actions)
                ratios = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advantages

                loss = -torch.min(surr1, surr2) + 0.5 * (batch_returns - values).pow(2) - ENTROPY_COEFF * entropy
                loss = loss.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []


def train_agent(env, agent, num_episodes=100):
    training_rewards = []
    noise_values = []
    entropy_values = []
    episode_lengths = []
    reward_stddevs = []
    training_times = [] 

    for episode in range(num_episodes):
        start_time = time.time()  

        state_info = env.reset()
        if isinstance(state_info, tuple):
            state = state_info[0]
        else:
            state = state_info

        state = (state - np.mean(state)) / (np.std(state) + 1e-8)

        episode_reward = 0
        steps = 0

        for _ in range(1000):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, dist, std = agent.policy.act(state_tensor)
            action = action.detach().numpy().flatten()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            noise_values.append(std.detach().numpy().mean())  

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, done, truncated, _ = step_output
                done = bool(done or truncated)  
            else:
                next_state, reward, done, _ = step_output
                done = bool(done)

            next_state = (next_state - np.mean(next_state)) / (np.std(next_state) + 1e-8)

            x_vel = next_state[2]
            y_vel = next_state[3]
            angle = next_state[4]
            leg1_contact = next_state[6]
            leg2_contact = next_state[7]

            reward -= 0.1 * (abs(x_vel) + abs(y_vel))
            reward -= 0.05 * abs(angle)
            if leg1_contact and leg2_contact:
                reward += 10
            x_pos = next_state[0]
            if abs(x_pos) > 1.0:
                reward -= 5

            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32)).sum().item()
            entropy_values.append(dist.entropy().mean().item())

            agent.store((state, action, reward, done, next_state, log_prob))

            state = next_state
            episode_reward += reward
            steps += 1

            if len(agent.memory) >= UPDATE_INTERVAL:
                agent.update()

            if done:
                break

        training_rewards.append(episode_reward)
        episode_lengths.append(steps)

       
        if len(training_rewards) >= 100:
            reward_stddev = np.std(training_rewards[-100:])
        else:
            reward_stddev = np.std(training_rewards)
        reward_stddevs.append(reward_stddev)

        end_time = time.time()  
        training_times.append(end_time - start_time) 

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward}")

    return training_rewards, noise_values, entropy_values, episode_lengths, reward_stddevs, training_times


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPO(state_dim, action_dim)
training_rewards, noise_values, entropy_values, episode_lengths, reward_stddevs, training_times = train_agent(env, agent)


min_length = min(
    len(training_rewards),
    len(noise_values),
    len(entropy_values),
    len(episode_lengths),
    len(reward_stddevs),
    len(training_times)
)


training_rewards = training_rewards[:min_length]
noise_values = noise_values[:min_length]
entropy_values = entropy_values[:min_length]
episode_lengths = episode_lengths[:min_length]
reward_stddevs = reward_stddevs[:min_length]
training_times = training_times[:min_length]


df_metrics = pd.DataFrame({
    'Episode': range(1, min_length + 1),
    'Reward': training_rewards,
    'Rolling Average (window=100)': pd.Series(training_rewards).rolling(window=100).mean(),
    'Noise (Std Dev)': noise_values,
    'Entropy': entropy_values,
    'Episode Length': episode_lengths,
    'Reward Std Dev (Rolling 100)': reward_stddevs,
    'Training Time (seconds)': training_times,
    'LR': LR,  
    'EPS_CLIP': EPS_CLIP,  
    'ENTROPY_COEFF': ENTROPY_COEFF, 
})

df_metrics.to_excel('training_metrics2.xlsx', index=False)