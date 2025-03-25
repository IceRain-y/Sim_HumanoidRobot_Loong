# import gym
# from gym import spaces
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import mujoco
# from mujoco import viewer

# # 自定义MuJoCo环境
# class CustomMujocoEnv(gym.Env):
#     def __init__(self, model_path):
#         # 加载模型
#         self.model = mujoco.MjModel.from_xml_path(model_path)
#         self.data = mujoco.MjData(self.model)
        
#         # 设置观察空间和动作空间（需根据实际模型调整）
#         self.action_space = spaces.Box(
#             low=-1.0, high=1.0,
#             shape=(self.model.nu,),  # 控制维度
#             dtype=np.float32
#         )
        
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf,
#             shape=(self._get_obs().shape[0],),
#             dtype=np.float32
#         )
        
#         # 初始化渲染器（可选）
#         self.viewer = None

#     def _get_obs(self):
#         """根据机器人状态构造观测向量"""
#         obs = np.concatenate([
#             self.data.qpos.flat[1:],  # 排除根关节位置
#             self.data.qvel.flat,
#             self.data.cinert.flat,
#             self.data.cvel.flat,
#             self.data.qfrc_actuator.flat
#         ])
#         return obs.astype(np.float32)

#     def step(self, action):
#         # 应用控制信号
#         self.data.ctrl[:] = action
        
#         # 前向模拟
#         mujoco.mj_step(self.model, self.data)
        
#         # 获取观测
#         obs = self._get_obs()
        
#         # 计算奖励
#         reward = self._reward_fn()
        
#         # 终止判断
#         done = self._termination_check()
        
#         return obs, reward, done, {}

#     def reset(self):
#         mujoco.mj_resetData(self.model, self.data)
#         return self._get_obs()

#     def _reward_fn(self):
#         """自定义奖励函数"""
#         # 直立奖励
#         torso_height = self.data.qpos[2]
#         height_reward = 1.0 - abs(torso_height - 1.0)  # 假设目标高度1m
        
#         # 平衡奖励
#         orientation = self.data.qpos[3:7]  # 四元数表示

#         # 将四元数转换为旋转矩阵
#         rot_matrix = np.zeros(9)
#         mujoco.mju_quat2Mat(rot_matrix, orientation)
#         rot_matrix = rot_matrix.reshape(3, 3)

#         # 计算z轴方向
#         up_vector = rot_matrix @ np.array([0, 0, 1])
#         balance_reward = up_vector[2]  # z分量

#         # up_vector = orientation @ np.array([0, 0, 1])  # 计算上方向量
#         # balance_reward = up_vector[2]  # z分量越大越直立
        
#         # 速度惩罚
#         velocity_penalty = 0.01 * np.square(self.data.qvel).sum()
        
#         # 动作平滑惩罚
#         action_penalty = 0.001 * np.square(self.data.ctrl).sum()
        
#         return height_reward + balance_reward - velocity_penalty - action_penalty

#     def _termination_check(self):
#         """终止条件"""
#         # 跌倒判断
#         return self.data.qpos[2] < 0.5  # 躯干高度低于0.5米

#     def render(self, mode='human'):
#         if self.viewer is None:
#             self.viewer = viewer.launch_passive(self.model, self.data)
#         self.viewer.sync()

# # PPO策略网络
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, action_dim),
#             nn.Tanh()
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.Tanh(),
#             nn.Linear(256, 256),
#             nn.Tanh(),
#             nn.Linear(256, 1)
#         )
        
#     def forward(self, x):
#         return self.actor(x), self.critic(x)

# # 训练参数
# BATCH_SIZE = 4096
# EPOCHS = 10
# GAMMA = 0.99
# LAMBDA = 0.95
# EPSILON = 0.2
# ENTROPY_COEF = 0.01

# def train():
#     # 初始化环境
#     env = CustomMujocoEnv("/home/icerain-y/Gym/urdf/OpenLoong/AzuLoong.xml")
    
#     # 初始化策略网络
#     policy = ActorCritic(
#         state_dim=env.observation_space.shape[0],
#         action_dim=env.action_space.shape[0]
#     )
#     optimizer = Adam(policy.parameters(), lr=3e-4)
    
#     # 训练循环
#     for episode in range(1000):
#         states = []
#         actions = []
#         rewards = []
#         dones = []
#         values = []
        
#         # 数据收集
#         state = env.reset()
#         for _ in range(BATCH_SIZE):
#             state_tensor = torch.FloatTensor(state)
#             with torch.no_grad():
#                 action_mean, value = policy(state_tensor)
                
#             # 添加探索噪声
#             action = action_mean + torch.randn_like(action_mean) * 0.1
#             action = action.clamp(-1, 1).numpy()
            
#             next_state, reward, done, _ = env.step(action)
            
#             # 存储轨迹数据
#             states.append(state)
#             actions.append(action)
#             rewards.append(reward)
#             dones.append(done)
#             values.append(value.item())
            
#             state = next_state
#             if done:
#                 state = env.reset()
        
#         # 计算优势函数
#         # returns = np.zeros_like(rewards)
#         # advantages = np.zeros_like(rewards)
#         # last_value = policy(torch.FloatTensor(state))[1].item()

#         # 计算优势函数时转换为Tensor
#         advantages = torch.FloatTensor(advantages).to(device)
#         returns = torch.FloatTensor(returns).to(device)
        
#         # 使用GAE计算优势
#         gae = 0
#         for t in reversed(range(len(rewards))):
#             delta = rewards[t] + GAMMA * (1 - dones[t]) * last_value - values[t]
#             gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
#             advantages[t] = gae
#             last_value = values[t]
#             returns[t] = advantages[t] + values[t]
        
#         # 转换为张量
#         states = torch.FloatTensor(np.array(states))
#         actions = torch.FloatTensor(np.array(actions))
#         returns = torch.FloatTensor(returns)
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
#         # PPO更新
#         for _ in range(EPOCHS):
#             action_means, values = policy(states)

#             # 计算策略损失（保持Tensor运算）
#             dist = torch.distributions.Normal(action_means, 0.1)  # 假设固定标准差
#             new_log_probs = dist.log_prob(actions).sum(dim=1)

#             ratio = torch.exp(new_log_probs - old_log_probs.detach())
            
#             # 计算策略损失
#             # ratio = torch.exp(-0.5 * ((actions - action_means) ** 2).sum(dim=1)) # 高斯策略

#             surr1 = ratio * advantages
#             surr2 = torch.clamp(ratio, 1-EPSILON, 1+EPSILON) * advantages
#             policy_loss = -torch.min(surr1, surr2).mean()
            
#             # 价值函数损失
#             value_loss = 0.5 * (returns - values.flatten()).pow(2).mean()
            
#             # 熵正则化
#             entropy = 0.5 * (1 + torch.log(2*torch.tensor(np.pi)) + 
#                            (action_means ** 2).sum(dim=1)).mean()
            
#             total_loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
            
#             optimizer.zero_grad()
#             total_loss.backward()
#             torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
#             optimizer.step()
        
#         # 输出训练进度
#         if episode % 10 == 0:
#             avg_reward = np.mean(rewards)
#             print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

# if __name__ == "__main__":
#     train()

######################################################################################

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import mujoco
from mujoco import viewer

# 自定义 MuJoCo 环境
class CustomMujocoEnv(gym.Env):
    def __init__(self, model_path):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 设置观察空间和动作空间（需根据实际模型调整）
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),  # 控制维度
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._get_obs().shape[0],),
            dtype=np.float32
        )
        
        # 初始化渲染器（可选）
        self.viewer = None

    def _get_obs(self):
        """根据机器人状态构造观测向量"""
        obs = np.concatenate([
            self.data.qpos.flat[1:],  # 排除根关节位置
            self.data.qvel.flat,
            self.data.cinert.flat,
            self.data.cvel.flat,
            self.data.qfrc_actuator.flat
        ])
        return obs.astype(np.float32)

    def step(self, action):
        # 应用控制信号
        self.data.ctrl[:] = action
        
        # 前向模拟
        mujoco.mj_step(self.model, self.data)
        
        # 获取观测
        obs = self._get_obs()
        
        # 计算奖励
        reward = self._reward_fn()
        
        # 终止判断
        done = self._termination_check()
        
        return obs, reward, done, {}

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def _reward_fn(self):
        """自定义奖励函数"""
        # 直立奖励
        torso_height = self.data.qpos[2]
        height_reward = 1.0 - abs(torso_height - 1.0)  # 假设目标高度1m
        
        # 平衡奖励
        orientation = self.data.qpos[3:7]  # 四元数表示

        # 将四元数转换为旋转矩阵
        rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(rot_matrix, orientation)
        rot_matrix = rot_matrix.reshape(3, 3)

        # 计算 z 轴方向
        up_vector = rot_matrix @ np.array([0, 0, 1])
        balance_reward = up_vector[2]  # z 分量

        # 速度惩罚
        velocity_penalty = 0.01 * np.square(self.data.qvel).sum()
        
        # 动作平滑惩罚
        action_penalty = 0.001 * np.square(self.data.ctrl).sum()
        
        return height_reward + balance_reward - velocity_penalty - action_penalty

    def _termination_check(self):
        """终止条件"""
        # 跌倒判断
        return self.data.qpos[2] < 0.5  # 躯干高度低于 0.5 米

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

# PPO 策略网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

# 训练参数
BATCH_SIZE = 4096
EPOCHS = 10
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
ENTROPY_COEF = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # 初始化环境
    env = CustomMujocoEnv("/home/icerain-y/Gym/urdf/OpenLoong/AzuLoong.xml")
    
    # 初始化策略网络
    policy = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=3e-4)
    
    # 训练循环
    for episode in range(1000):
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        
        # 数据收集
        state = env.reset()
        for _ in range(BATCH_SIZE):
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action_mean, value = policy(state_tensor)
                
            # 添加探索噪声
            action = action_mean + torch.randn_like(action_mean) * 0.1
            action = action.clamp(-1, 1).cpu().numpy()
            
            next_state, reward, done, _ = env.step(action)
            
            # 存储轨迹数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            # 渲染环境
            env.render()  # 在这里调用渲染功能
            
            state = next_state
            if done:
                state = env.reset()
        
        # 转换为张量
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        values = torch.FloatTensor(values).to(device)

        # 计算优势函数
        advantages = torch.zeros(BATCH_SIZE, dtype=torch.float32).to(device)
        returns = torch.zeros(BATCH_SIZE, dtype=torch.float32).to(device)
        last_value = policy(states[-1].unsqueeze(0))[1].item()

        # 使用 GAE 计算优势
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * (1 - dones[t]) * last_value - values[t]
            gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
            last_value = values[t]
            returns[t] = advantages[t] + values[t]

        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        for _ in range(EPOCHS):
            action_means, values = policy(states)

            # 计算策略损失
            dist = torch.distributions.Normal(action_means, 0.1)  # 假设固定标准差
            new_log_probs = dist.log_prob(actions).sum(dim=1)

            with torch.no_grad():
                old_action_means, _ = policy(states)
                old_dist = torch.distributions.Normal(old_action_means, 0.1)
                old_log_probs = old_dist.log_prob(actions).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = 0.5 * (returns - values.flatten()).pow(2).mean()
            
            # 熵正则化
            entropy = 0.5 * torch.log(2 * torch.tensor(np.pi) * 0.1 ** 2) + 0.5
            
            total_loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
        
        # 输出训练进度
        if episode % 10 == 0:
            avg_reward = rewards.mean().item()  # 使用 PyTorch 的 mean 方法
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    train()