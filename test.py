import torch
import time
from method.a2c import A2CAgent
from lunar_lander import LunarLanderEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/a2c_checkpoint_episode_5820.pth"

def load_checkpoint(agent, checkpoint_path):
    """
    加載模型的 checkpoint。
    
    Args:
        agent (A2CAgent): A2C代理
        checkpoint_path (str): 模型檢查點的路徑
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")


def run_inference(agent, env, episodes=100, max_steps=300, render_delay=0.05):
    """
    執行模型推理並顯示 LunarLander 動畫。

    Args:
        agent (A2CAgent): 訓練好的 A2C 代理
        env (LunarLanderEnvironment): LunarLander 環境
        episodes (int): 推理的回合數
        max_steps (int): 每回合的最大步數
        render_delay (float): 動畫顯示的間隔時間（秒）
    """
    for episode in range(episodes):
        state = env.reset()  # 初始化狀態
        total_reward = 0
        done = False

        print(f"Starting Episode {episode + 1}...")

        for step in range(max_steps):
            # 將狀態轉換為 PyTorch Tensor 並移至設備
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)

            # 使用模型選擇動作
            action = agent.get_action(state_tensor)
            
            # 執行動作並獲得下一狀態和回報
            next_state, reward, done, _ = env.step(action)

            # 累加總回報
            total_reward += reward

            # 更新當前狀態
            state = next_state

            # 渲染動畫
            env.render()
            # time.sleep(render_delay)

            # 若遊戲結束，跳出
            if done:
                break

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()

def main():
    """
    主函數：初始化環境和模型，運行推理過程。
    """
    # 初始化 LunarLander 環境
    render_mode = "human"
    env = LunarLanderEnvironment(render_mode=render_mode)

    # 定義狀態和動作空間大小
    state_size = 8
    action_size = 4

    # 初始化 A2C Agent
    agent = A2CAgent(
        state_size=state_size,
        action_size=action_size,
        actor_lr=0.0001,
        critic_lr=0.0005,
        gamma=0.99,
        dropout_rate=0.1
    ).to(device)

    # 加載訓練好的模型參數
    load_checkpoint(agent, checkpoint_path)

    # 運行推理並顯示 LunarLander 動畫
    run_inference(agent, env, episodes=10, render_delay=0.05)

if __name__ == "__main__":
    main()
