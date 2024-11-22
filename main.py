import gymnasium as gym
import torch
from method.a2c import A2CAgent  # 引入 A2C Agent
import numpy as np
from lunar_lander import LunarLanderEnvironment
from collections import deque
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_checkpoint(agent, episode, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"a2c_checkpoint_episode_{episode}.pth")
    torch.save({
        'episode': episode,
        'actor_state_dict': agent.actor.state_dict(),  # 保存 Actor 網絡
        'critic_state_dict': agent.critic.state_dict(),  # 保存 Critic 網絡
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    
    env = LunarLanderEnvironment()

    # 初始化 A2C agent
    state_size = 8
    action_size = 4
    agent = A2CAgent(state_size = state_size, 
                     action_size = action_size, 
                     actor_lr=0.0005, 
                     critic_lr=0.0005, 
                     gamma=0.99, 
                     dropout_rate=0.1
                     ).to(device)

    num_episodes = 10000
    # 訓練回合數
    max_steps = 300  # 每回合最大步數
    reward_mean_num = 10

    rewards = []
    reward_window = deque(maxlen=reward_mean_num)

    for episode in range(num_episodes):

        # max_steps = max(1000 - 5 * (episode // 10), 300)

        total_reward = agent.run_episode(env, max_steps, device)
        reward_window.append(total_reward)
        rewards.append(total_reward)

        # 每 10 個回合計算並打印平均獎勳
        if episode % 10 == 0:
            average_reward = np.mean(reward_window)
            print(f"Episode: {episode}, Average Reward: {average_reward:.2f}")

            # 如果平均回報 >= 200，保存模型並停止訓練
            if average_reward >= 230:
                print(f"Stopping training early: Average Reward reached {average_reward:.2f} at Episode {episode}.")
                save_checkpoint(agent, episode, checkpoint_dir="checkpoints")
                break

        # 每 100 個回合顯示 max_steps
        if episode % 100 == 0:
            print("max_steps:", max_steps)

    save_checkpoint(agent, episode, checkpoint_dir="checkpoints")
    env.close()
    plot_rewards(rewards)

if __name__ == "__main__":
    main()

