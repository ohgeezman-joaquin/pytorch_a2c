import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.1):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 128)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.prelu1(x)
        x = self.fc2(x)
        x = self.prelu2(x)
        return torch.softmax(self.fc3(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_size, dropout_rate=0.1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 128)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.prelu1(x)
        x = self.fc2(x)
        x = self.prelu2(x)
        return self.fc3(x)

class A2CAgent(nn.Module):
    def __init__(self, state_size, action_size, actor_lr=0.001, critic_lr=0.005, gamma=0.99, dropout_rate=0.1):
        super(A2CAgent, self).__init__()
        self.gamma = gamma

        # 建立 Actor 和 Critic 網絡
        self.actor = ActorNetwork(state_size, action_size, dropout_rate)
        self.critic = CriticNetwork(state_size, dropout_rate)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def calculate_returns(self, rewards, discount_factor, normalize=True):
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + R * discount_factor
            returns[t] = R
        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def get_action(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def calculate_advantages(self, returns, values, normalize=True):
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update(self, states, actions, rewards, next_states, dones):
        values = self.critic(states).squeeze(1)
        next_values = self.critic(next_states).squeeze(1) * (1 - dones)

        returns = self.calculate_returns(rewards, self.gamma)
        advantages = self.calculate_advantages(returns, values)

        log_probs = torch.log(self.actor(states).gather(1, actions.unsqueeze(1)).squeeze(1))
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.smooth_l1_loss(values, returns.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def run_episode(self, env, max_timesteps, device):
        state = torch.tensor(env.reset(), dtype=torch.float32, device=device)
        total_reward = 0

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []

        for _ in range(max_timesteps):
            action = self.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.float32, device=device)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            state = next_state
            total_reward += reward.item()

            if done:
                break

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        next_states = torch.stack(next_states)

        self.update(states, actions, rewards, next_states, dones)
        return total_reward
