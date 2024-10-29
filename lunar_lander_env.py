import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import pandas as pd 


LR_ACTOR = 3e-4
LR_CRITIC = 1e-6
GAMMA = 0.99
EPS_CLIP = 0.3
K_EPOCHS = 15
ENTROPY_COEFF = 0.07
UPDATE_INTERVAL = 500
HIDDEN_SIZE = 256
MIN_BATCH_SIZE = 88
GAE_LAMBDA = 0.99
MAX_GRAD_NORM = 1.0   
REWARD_SCALING = -1.0 


class RunningStat:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.sq_mean = 0

    def update(self, x):
        self.n += 1
        old_mean = self.mean
        self.mean += (x - self.mean) / self.n
        self.sq_mean += (x - old_mean) * (x - self.mean)

    def std(self):
        return np.sqrt(self.sq_mean / self.n) if self.n > 1 else 1.0


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim)
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        mu = self.actor_fc(state)
        value = self.critic_fc(state)
        return mu, value

    def act(self, state):
        mu, _ = self.forward(state)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist.sample(), dist

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
        self.optimizer_actor = optim.Adam(self.policy.actor_fc.parameters(), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.policy.critic_fc.parameters(), lr=LR_CRITIC)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=1000, gamma=0.9)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=1000, gamma=0.9)
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

        
        returns, advantages = [], []
        G, A = 0, 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + (1 - dones[t]) * GAMMA * G
            delta = rewards[t] + (1 - dones[t]) * GAMMA * self.policy.critic_fc(states[t].unsqueeze(0)).detach() - self.policy.critic_fc(states[t].unsqueeze(0)).detach()
            A = delta + (1 - dones[t]) * GAMMA * GAE_LAMBDA * A
            returns.insert(0, G)
            advantages.insert(0, A)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        
        global ENTROPY_COEFF
        ENTROPY_COEFF = max(0.01, ENTROPY_COEFF * 0.99)


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

                
                critic_loss = 0.5 * (batch_returns - values).pow(2).mean()
                loss = -torch.min(surr1, surr2) + critic_loss - ENTROPY_COEFF * entropy
                loss = loss.mean()

                
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

        self.scheduler_actor.step()
        self.scheduler_critic.step()
        self.memory = []


def train_agent(env, agent, num_episodes=100):
    training_rewards = []
    noise_values = []
    entropy_values = []
    episode_lengths = []
    reward_stddevs = []
    training_times = []
    running_stat = RunningStat()

    for episode in range(num_episodes):
        state_info = env.reset()
        if isinstance(state_info, tuple):
            state = state_info[0]
        else:
            state = state_info

        running_stat.update(state)
        state = (state - running_stat.mean) / (running_stat.std() + 1e-8)

        episode_reward = 0
        episode_length = 0
        start_time = time.time()

        episode_noise = []  
        episode_entropy = []  

        for _ in range(1500):  
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action, dist = agent.policy.act(state_tensor)
            action = action.detach().numpy().flatten()

            # Reduce Gaussian noise for better balance
            noise = np.random.normal(0, 0.05, size=action.shape)
            action = action + noise
            action = np.clip(action, env.action_space.low, env.action_space.high)
            episode_noise.append(np.mean(noise))  # Collect noise value for this step

            step_output = env.step(action)
            if len(step_output) == 5:
                next_state, reward, done, truncated, _ = step_output
                done = done or truncated
            else:
                next_state, reward, done, _ = step_output

            reward *= REWARD_SCALING

            running_stat.update(next_state)
            next_state = (next_state - running_stat.mean) / (running_stat.std() + 1e-8)

            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float32)).sum().item()
            agent.store((state, action, reward, done, next_state, log_prob))

            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_entropy.append(dist.entropy().mean().item())  

            if len(agent.memory) >= UPDATE_INTERVAL:
                agent.update()

            if done:
                break

        training_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        training_times.append(time.time() - start_time)
        reward_stddevs.append(running_stat.std())
        noise_values.append(np.mean(episode_noise))  
        entropy_values.append(np.mean(episode_entropy))  

        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward}")

        
        if len(training_rewards) >= 100:
            avg_reward = np.mean(training_rewards[-100:])
            if avg_reward >= 200:
                print(f"Solved at episode {episode + 1}, Average Reward: {avg_reward}")
                break

    return training_rewards, noise_values, entropy_values, episode_lengths, reward_stddevs, training_times


env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = PPO(state_dim, action_dim)
training_rewards, noise_values, entropy_values, episode_lengths, reward_stddevs, training_times = train_agent(env, agent, num_episodes=1000)


data = {
    'Episode': list(range(1, len(training_rewards) + 1)),
    'Training Reward': training_rewards,
    'Noise Value': noise_values,
    'Entropy Value': entropy_values,
    'Episode Length': episode_lengths,
    'Reward Std Dev': reward_stddevs,
    'Training Time': training_times,
    'Entropy Coeff': [ENTROPY_COEFF] * len(training_rewards),
    'Reward Scaling': [REWARD_SCALING] * len(training_rewards),
    'Gamma': [GAMMA] * len(training_rewards)
}
df = pd.DataFrame(data)
df.to_excel('lunar_lander_training_data4.xlsx', index=False)