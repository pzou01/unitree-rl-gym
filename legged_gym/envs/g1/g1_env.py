
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        # 人形需要去拿到foot的信息 人形就是左右脚了
        self.feet_num = len(self.feet_indices) # feet indices 在  feed name 在asset 那拿到的
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # 获取这个每个关节姿态 每个关节还是经典的 13个信息
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)  # 变成 num env --- num body 13
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13) #然后把num env里面每个body的13个信息闹到 后
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :] #根据feetindices 去除那13个信息
        self.feet_pos = self.feet_state[:, :, :3] # 位置就是这个状态0-3
        self.feet_vel = self.feet_state[:, :, 7:10] # 速度是 7-10
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot() # 然后把这个加入 init buffer 去 初始化 给这个缓存空间并且拿到值 后面用

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim) # 要这个去更新这个foot的值 因为父类没有这个
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state() # 然后在这里 在父类加这个类进去
        # 父类就是做了一个重新采样commands 然后 计算误差 更新地形和推理 只是准备command 提供下一步的目标速度和朝向
        #所以子类我们要继承后在这里加东西
        # 这个函数调用的时候是 物理引擎刚更新完毕 → 还没算奖励/观测 → 允许我们做修改 这个时候我们保证数据是刚更新完的

        #在这里加步态相位
        """
        phase 有两个要求：要依赖时间（episode_length_buf）  要作为 obs 的一部分被策略用到  reward 也可能依赖 phase（比如脚摆动惩罚/落脚惩罚）
        obs 和 reward 都在 callback 之后 才计算！如果你不把 phase 放在这里，那就会出错：太早算（物理没更新） → 时间不准太晚算（obs 已经生成） → 策略看不到 phase
        放在 init/reset → 只有第一步有效，没有周期性所以：这个时机是给你“准备下一步观测、下一步奖励”用的。phase 就必须在这里生成。
        apply action →
        physics simulate →
        (post_physics_step_callback) ← 你在这里  
        compute reward  
        compute done  
        compute observations  
        return (obs, reward)
        什么内容必须在“奖励/观测生成之前”更新？脚是否着地（用于 slip reward、support reward）
        脚的速度（用于 foot clearance reward）
        步态相位 phase（用于 gait conditioning）
        触地点位置（用于抓地/落脚 reward）
        身体倾斜信息（用于 balance reward）
        这些全都来自 刚刚模拟完的物理状态。
        因此：
        post_physics_step_callback 就是整个 Isaac Gym 里唯一一个
        “可以基于最新物理状态，修改下一步 reward/obs 输入” 的钩子（hook）。
        所以所有 humanoid 相关逻辑都必须放这里。
        """

        # 这里 phase 在这里就是“走路节奏的进度条”。 相位 phase ∈ [0, 1) 0.0 表示“这一小步刚开始” 0.5 表示“走到一半” 接近 1.0 表示“快结束这个周期了，下一步又从 0 开始
        # 就像节拍器从 0% 走到 100%，然后重新归零，再 0 → 100%，不断循环。
        period = 0.8 # 一个完整步态周期的时间长度 0.8s 机器人理想状态下一次完整“左—右—左—右”的循环节奏，用 0.8s 来走完。
        offset = 0.5 # 个是左右腿之间的相位差： = 半个周期 = 180° 相位差 当左腿处于相位 0.0 时 → 右腿处于相位 0.5 当左腿处于 0.25 时 → 右腿处于 0.75 两条腿永远错开半个周期
        # 这个地方就是核心 self.episode_length_buf * self.dt  当前环境从这一幕 episode 开始到现在，已经过了多少秒  % period 对 period = 0.8 取模 走了 1.3 秒 → 1.3 % 0.8 = 0.5 秒（当前周期的中间） 走了 2.1 秒 → 2.1 % 0.8 = 0.5 一样的周期位置
        self.phase = (self.episode_length_buf * self.dt) % period / period # / period 归一化到 [0, 1)： 然后  相位 phase =（当前周期内的时间） /（周期总时长） 当前周期内时间 = 0.4 s 周期 = 0.8 s phase = 0.4 / 0.8 = 0.5（走到这一周期一半）
        self.phase_left = self.phase # self.phase ∈ [0, 1)    # 一个随时间匀速转一圈“进度条” 然后左腿相位就是这个时刻的值
        # 右腿相位
        self.phase_right = (self.phase + offset) % 1 #永远比左腿晚半个周期（交替步伐） % 1 保证相位依然在 [0, 1) 里循环。
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1) #  self.phase_left 原本形状是 [num_envs] unsqueeze(1) → 变成 [num_envs, 1] 两个 [num_envs, 1] concat → [num_envs, 2]
        #  也就是对每个环境 env，得到一行： [ phase_left, phase_right ]

        # 然后这个值 后面拿过来
        # 计算每条腿的 sin/cos
        # 设计基于相位的 gait reward（比如某个相位必须在空中/在地面）

        # 用处也就是 就是 sin 和 cos 作为 obs的一部分
        # 用在 计算gait 和 contact 有关的 reward
        # 如果在obs中既可以看到当前身体姿态和命令和动作和当前的相位  多了两个 观测维度 然后 就会根据做这个观测 来学习不同phase 做不同的动作  然后用reward 去调整
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        # 上面的phase的值 拿过来用了 这里用的是 self.phase，也就是总相位（可以理解为左腿相位）。
        # sin(2π phase)、cos(2π phase)  把一个周期变量编码成连续、光滑、不在 0/1 处分裂的特征
        # 这样网络能理解“相位接近 0 和接近 1 是邻近的”，而不是两个远离的数
        # 给策略一个“时间节奏 / 步态时钟”（gait clock）：
        # 告诉策略：
        # 现在是“迈步开始”？还是“迈步中间”？还是“该准备落脚”？
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel, # base的角速度
                                    self.projected_gravity, #在base的重力
                                    self.commands[:, :3] * self.commands_scale, # vx vy yaw rate 命令
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #关节位置
                                    self.dof_vel * self.obs_scales.dof_vel, # 关节速度
                                    self.actions, # 关节动作
                                    sin_phase, # 步态相位
                                    cos_phase # 步态相位
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, # 多了一个base的线速度
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        # 接触奖励
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num): # 训练两个脚
            is_stance = self.leg_phase[:, i] < 0.55 # 如果当前周期是 小于0.55的那么就一个 是 在stance 站立支撑
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1 # 然后取z轴的接触力  取i的foot然后就应该有接触力
            res += ~(contact ^ is_stance) # 然后这里是 异或 符号 如果 contact 是 true 就是在地上 加上 支撑true 异或就是 false 同理 摆动期和无支撑 都是 true 就false 两个不一致就是true 那么不一致 就是 true 然后前面一个~ 取反 符号
        # 每只脚每个环境，如果当前相位下的支撑/摆动与实际接触模式一致 → 加 1 然后这里的 脚的支持 就是相位支撑和摆动是和实际接触一直就加一
        return res  # 这个 res: 形状 [num_envs]，数值越大表示“步态时序越符合预期” 然后每个环境的 越多就reward越大
    
    def _reward_feet_swing_height(self):
        # 摆动期的脚高度
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1. # 拿到每个脚的接触力
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact # 对z轴上的接触力 计算位置 然后高度我希望是接近0.08米  ----- 只在 没有接触的脚（摆动期） 上计算误差
        return torch.sum(pos_error, dim=(1)) # 摆动脚高度在 0.08 左右 → 误差小 → 惩罚小 → 间接鼓励脚抬到这个高度
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0 #没摔倒姿态在这个期间内就 活着就有奖励
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        # 脚在地上时不要乱滑
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1. # 脚的也就是告诉我哪个脚在地面上 接触了
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1) #  上面拿到了 feet的速度 乘上脚接触
        penalize = torch.square(contact_feet_vel[:, :, :3]) # 脚在接触姿态下的速度平方 用来惩罚
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        # 约束髋关节姿态（防止奇怪姿势）
        """
        取的是 DOF 中第 [1,2,7,8] 四个关节：很可能是左右髋外展/内收、左右髋旋转之类的
        DOF对这几个关节的角度平方求和 → 越偏离 0 惩罚越大直觉：
        不希望人形做出很奇怪的“开胯、扭髋、外八字”姿态
        希望髋关节附近更“自然中立”一些。
        """
        return torch.sum(torch.square(self.dof_pos[:,[1,2,7,8]]), dim=1)
    