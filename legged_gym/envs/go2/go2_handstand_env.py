import torch
from isaacgym import gymtorch
from legged_gym.envs.base.legged_robot import LeggedRobot


class GO2HandstandEnv(LeggedRobot):
    """
    Go2 Handstand task:
    - 前腿(FL, FR)为支撑脚；
    - 后腿(RL, RR)不接触地面；
    - 机身倒立稳定；
    - 保留 LeggedRobot 通用的控制 & 训练流程，仅替换 obs / reward / termination。
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        assert len(self.feet_indices) >= 4, "Expected at least 4 feet for Go2."
        self.front_feet_indices = self.feet_indices[[0, 1]]
        self.hind_feet_indices = self.feet_indices[[2, 3]]

        # handstand 目标关节姿态，已经有：
        self.dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}
        self.pose_targets = self._build_pose_from_dict(
            self.cfg.handstand_pose.joint_angles
        )  # [num_envs, num_dof]

        # ==== 定义后腿关节 ====
        rear_joint_names = [
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        rear_idx = []
        for n in rear_joint_names:
            if n in self.dof_name_to_idx:
                rear_idx.append(self.dof_name_to_idx[n])
            else:
                print(f"[WARN] rear joint name {n} not in dof_name_to_idx")
        self.rear_dof_idx = torch.tensor(
            rear_idx, dtype=torch.long, device=self.device
        )

        # 以 handstand 姿态为中心，把后腿锁在附近
        # 可以用 default_dof_pos 或 pose_targets，这里用 handstand 目标更合理：
        self.rear_center = self.pose_targets[0, self.rear_dof_idx].clone()
        self.rear_limit = 0.15  # 允许 ±0.15 rad 小幅摆动

    # =========================================================
    # 构造目标姿态 tensor
    # =========================================================
    def _build_pose_from_dict(self, pose_dict):
        pose = self.default_dof_pos[0].clone()  # 从默认站姿出发 这里的 default dof pos 就是我们init state的值 这里clone
        for name, angle in pose_dict.items():
            if name in self.dof_name_to_idx:
                pose[self.dof_name_to_idx[name]] = angle
        return pose.unsqueeze(0).repeat(self.num_envs, 1)  # [num_envs, num_dof] # 把所有关节用倒立的姿态 覆盖后 返回

    # =========================================================
    # 初始化 / buffer
    # =========================================================
    def _init_buffers(self):
        super()._init_buffers() # 为了并行环境 gpu在别的 所有并行环境的 姿态缓存区  --_init_buffers() 是父类里负责「注册所有状态张量」和「创建 RL 输入输出 buffer」的函数。
        # 也就是父类 就拿到了 root state 13个 3+4+3+3  ||dof pos 12  和 dof vel 12  24个  || contact forces 每个bodies的3个力
        self._init_feet_states() # 除了父类的这张图我还想额外记录脚部的状态 每个foot的位置和速度 前后脚的接触 父类值提供了 最基础的 contact forces 不知道什么是脚 所以 扩展了 init buffer

    def _init_feet_states(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # 获取了所有的刚体的信息 每个刚体有13个信息 --- 机器人就是刚体 和 关节约束joint 组成的
        # 刚体】──【关节】──【刚体】──【关节】──【刚体】  刚体 (rigid body)：一个可以在三维空间中独立拥有 位置、姿态、速度、角速度 的实体。
        # 关节 (joint)：约束两个刚体之间的相对运动（例如铰链、滑块、球形关节）
        # 在 .xml 或 .urdf 模型文件里，每个 <link> 元素代表一个刚体。  Isaac Gym 读取 URDF 时，会为每个 link 创建一个 rigid body。 “rigid body 数量 = URDF 里 link 的数量”。
        # 那么 go2 有 base 1个 腿3个 hip thigh cale 然后4个腿 加起来就是 13个刚体
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state) #gymtorch.wrap_tensor() 让我们可以用 PyTorch 的方式直接访问 GPU 上的数据；这个 tensor 与仿真状态共享内存（即 zero-copy，不会复制内存）。
        # 这意味着后面只要调用：self.gym.refresh_rigid_body_state_tensor(self.sim)  物理仿真步的数据会自动更新到这块 GPU buffer 上。

        # 这里就是 把 numenv x 机器人个数 13 变成了 numenv numrobot 13
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13) #因为我们通常会一次性仿真上千个 robot 环境。  这里用 .view() 把一大块连续内存重新 reshape 成：这样我们就能方便地按“第 i 个机器人”的索引来取状态了。
        # 取出「脚部刚体」的状态子集 self.feet_indices 是一个列表（父类 _create_envs() 时生成），比如 [3, 7, 11, 15]， 表示“机器人在 rigid body 数组里哪些 index 对应脚部”。
        # 这一步的结果是一个形状：[self.num_envs, num_feet, 13] 的张量，只保存脚的状态。
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        # 拆出位置和速度  3 → 取出世界坐标下的 XYZ 位置； 7:10 → 取出线速度部分（第 7~9 分量）。 这就是我们任务中要用的关键信息。
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _update_feet_states(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim) #刷新一次后 给新的值进去 保证我们刚才初始化的3个姿态的值是最新的
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self._update_feet_states()
        return

    # =========================================================
    # Observations：46维版本
    # =========================================================
    """
    def compute_observations(self): # 覆写父类的 obs 因为我们的obs不一样了
        """
    """
        obs = [
            base_ang_vel * scale (3),
            projected_gravity (3),
            (dof_pos - default_dof_pos) * scale (12),
            dof_vel * scale (12),
            actions (12),
            front_feet_contact (2),
            hind_feet_contact (2),
        ] = 46
    """
    """
        base_ang = self.base_ang_vel * self.obs_scales.ang_vel # base 角速度
        grav = self.projected_gravity # 重力方向
        dof_pos_err = (self.dof_pos - self.pose_targets) * self.obs_scales.dof_pos # 关节差
        dof_vel = self.dof_vel * self.obs_scales.dof_vel # 关节角速度

        front_contact = (self.contact_forces[:, self.front_feet_indices, 2] > 5.0).float()
        hind_contact = (self.contact_forces[:, self.hind_feet_indices, 2] > 5.0).float()

        self.obs_buf = torch.cat(
            (
                base_ang,
                grav,
                dof_pos_err,
                dof_vel,
                self.actions,
                front_contact.view(self.num_envs, -1),
                hind_contact.view(self.num_envs, -1),
            ),
            dim=-1,
        )
        self.privileged_obs_buf = self.obs_buf.clone()

        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

        return self.obs_buf
    """

    def compute_observations(self):
        base_ang = self.base_ang_vel * self.obs_scales.ang_vel
        grav = self.projected_gravity
        dof_pos_err = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel = self.dof_vel * self.obs_scales.dof_vel

        front_contact = (self.contact_forces[:, self.front_feet_indices, 2] > 5.0).float()
        hind_contact = (self.contact_forces[:, self.hind_feet_indices, 2] > 5.0).float()

        obs = torch.cat(
            (base_ang, grav, dof_pos_err, dof_vel, self.actions,
             front_contact.view(self.num_envs, -1),
             hind_contact.view(self.num_envs, -1)),
            dim=-1,
        )

        # 连续量统一加零均值噪声（向量逐元素尺度）
        if self.add_noise:
            obs += (2.0 * torch.rand_like(obs) - 1.0) * self.noise_scale_vec

            # 可选：对**接触标志**做“随机丢包/误触发”，增强鲁棒性
            # 例如 1% 概率把 0->1 或 1->0（谨慎使用，别太大）
            flip_p = 0.01
            if flip_p > 0:
                # 只对最后4维做
                flips = (torch.rand(self.num_envs, 4, device=self.device) < flip_p).float()
                obs[:, -4:] = torch.clamp(
                    torch.logical_xor((obs[:, -4:] > 0.5), (flips > 0.5)).float(),
                    0, 1
                )

        self.obs_buf = obs
        self.privileged_obs_buf = obs.clone()
        return self.obs_buf

    # =========================================================
    # Termination：适配 handstand
    # =========================================================
    def check_termination(self):
        # base非法接触 contact forced 是 num env - num body - 3  这里的 termination_contact_indices 指明「哪些刚体如果与地面接触是非法的」
        # 取出 所以环境中 所有禁止接触的刚体 的 接触力  例如 [4096, 2, 3] 这里2是 base 和 torso这个go2没有 有的机器狗有 是胸腔
        # 然后 norm 是 对每个刚体的接触力向量求模长（也就是总接触力大小）。
        # 最后变成 [4096, 2]    # 每个环境的 base、torso 的接触力大小
        # 如果接触力超过 1N，就认为发生了「明显接触」。
        # 如果某个环境中任意一个非法刚体接触地面，就认为该环境需要 reset。 然后输出 shape 4096 布尔利息表示需要重置或者继续
        #在这里就是 我改成了base hip thigh calf 都不可以接触地面 否则结束
        """
            Handstand 终止条件（覆盖父类）:
            1) 任何非法刚体(body in terminate_after_contacts_on) 与地面有明显接触
            2) 后腿脚触地
            3) 身体已经不在“倒立附近”姿态（防止侧躺、仰躺混奖励）
            4) 超时
            """
        # ---------- 1) 非法刚体接触 ----------
        # termination_contact_indices 在 LeggedRobot 里根据 cfg.asset.terminate_after_contacts_on 初始化好
        # contact_forces: [num_envs, num_bodies, 3]
        if self.termination_contact_indices.numel() > 0:
            illegal_contact = torch.any(
                torch.norm(
                    self.contact_forces[:, self.termination_contact_indices, :],
                    dim=-1,
                ) > 1.0,
                dim=1,
            )
        else:
            illegal_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)



        # ---------- 3) 姿态崩掉 ----------
        ## 2) 姿态崩掉（可选：侧翻/乱躺）
        cos_up = self.projected_gravity[:, 2]
        fallen = (cos_up > -0.2) & (cos_up < 0.2)   # 你也可以先关掉这条，等稳定了再打开


        # ---------- 4) 超时 ----------
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        # 合并
        self.reset_buf = illegal_contact | fallen | self.time_out_buf
    # =========================================================
    # Reward functions：与 cfg.rewards.scales 对齐
    #目标：让“站着或躺着”收益极低，只有“前腿顶着、后腿抬起、接近目标关节、整体倒立稳定”。
    # =========================================================
    def _reward_orientation(self): # 姿态方向奖励
        """
               用 projected_gravity 强推机器人进入“倒立半球”。

               g_z = 1   : 完美倒立        -> 奖励接近 1
               g_z = 0   : 侧躺/横着       -> 奖励约 e^{-3} ≈ 0.05
               g_z = -1  : 正常站立/仰面   -> 奖励约 e^{-12} ≈ 0
               """
        g_z = self.projected_gravity[:, 2].clamp(-1.0, 1.0)
        err = 1.0 - g_z  # 目标是 +1
        return torch.exp(-3.0 * err * err)


    def _reward_base_height(self): # 机身高度
        """
                handstand 时 base 会比 normal 高很多。
                只奖励低于 target_h 的误差，小于等于 target_h 视作完美(=1)。
                """
        base_h = self.root_states[:, 2]
        target_h = self.cfg.rewards.base_height_target  # handstand 建议改成 ~0.5-0.55
        # 低于目标高度才有误差；高过目标不扣（已经足够高了）
        err = (target_h - base_h).clamp(min=0.0)
        return torch.exp(-8.0 * err * err)

    def _reward_front_feet_contact(self): # 前脚接触就给reward
        """
                要求：两个前脚是主要支撑。
                contact_forces[...,2] > 5N 视作接触。
                平均一下：两个都踩着 -> 1，一个踩 -> 0.5，都没踩 -> 0。
                """
        contact = (self.contact_forces[:, self.front_feet_indices, 2] > 5.0).float()
        return contact.mean(dim=1)


    def _reward_hind_feet_no_contact(self): # 后脚接触就惩罚
        """
                要求：后脚悬空。
                有接触 -> 降到 0；都不接触 -> 1。
                """
        contact = (self.contact_forces[:, self.hind_feet_indices, 2] > 5.0).float()
        return 1.0 - contact.mean(dim=1)

    """
    def _reward_pose(self): #计算每个关节角和目标关节脚的差距
        
                #关节接近 handstand 目标姿态（pose_targets）。
                #用均方误差 + 指数，把“接近目标”变成一个平滑的 shaping。
                
        err = self.dof_pos - self.pose_targets  # [num_envs, num_dof]
        err_sq_mean = torch.mean(err * err, dim=1)  # 每个 env 一个标量
        return torch.exp(-5.0 * err_sq_mean)
    """

    def _reward_pose(self):
        """区分前支撑腿和后空中腿的软约束."""
        err = self.dof_pos - self.pose_targets  # [N, 12]

        # 举例：前腿误差全额算，后腿误差打个折
        # 按实际 DOF 顺序调整这些 index
        front_ids = [
            self.dof_name_to_idx['FL_hip_joint'],
            self.dof_name_to_idx['FL_thigh_joint'],
            self.dof_name_to_idx['FL_calf_joint'],
            self.dof_name_to_idx['FR_hip_joint'],
            self.dof_name_to_idx['FR_thigh_joint'],
            self.dof_name_to_idx['FR_calf_joint'],
        ]
        hind_ids = [
            self.dof_name_to_idx['RL_hip_joint'],
            self.dof_name_to_idx['RL_thigh_joint'],
            self.dof_name_to_idx['RL_calf_joint'],
            self.dof_name_to_idx['RR_hip_joint'],
            self.dof_name_to_idx['RR_thigh_joint'],
            self.dof_name_to_idx['RR_calf_joint'],
        ]

        front_err = err[:, front_ids]
        hind_err = err[:, hind_ids]

        # 前腿要“认真”对齐，后腿允许更自由：系数 1.0 vs 0.3 之类
        sq = (front_err ** 2).sum(dim=1) + 0.3 * (hind_err ** 2).sum(dim=1)

        return torch.exp(-4.0 * sq)


    def _reward_stability(self): # 检查在x和y平面上的线速度和角速度 应该保持稳定  越平稳就奖励越高
        """
                不乱晃：手倒立时横向速度 + pitch/roll 角速度越小越好。
                保留一点容忍度，避免死板。
                """
        lin_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)  # x,y 平移
        ang_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)  # roll, pitch 方向
        # 系数可以再调，先给个温和的
        return torch.exp(-2.0 * (lin_xy + 0.5 * ang_xy))

    def _reward_energy(self): # 力矩 x 角速度是 功率 对所有关节求和 总能耗
        """
                只有 cfg.rewards.scales.energy < 0 时才会被当作惩罚项用。
                目前 config 里 energy=0，可先忽略。
                """
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

    def _compute_torques(self, actions):
        """
        覆写父类:
        - 仍然用 P 控制
        - 但强行把后腿 target 限制在 handstand 附近 ±rear_limit
        """
        # 原始 target
        targets = self.default_dof_pos + self.cfg.control.action_scale * actions

        # 如果定义了后腿索引，则对后腿 target 做 clamp
        if hasattr(self, "rear_dof_idx") and self.rear_dof_idx.numel() > 0:
            # 允许相对 rear_center 有小幅变化
            low = self.rear_center - self.rear_limit
            high = self.rear_center + self.rear_limit

            # clamp 后腿 6 个关节
            targets[:, self.rear_dof_idx] = torch.clamp(
                targets[:, self.rear_dof_idx],
                low,
                high,
            )

        # 标准 PD torque
        torques = self.p_gains * (targets - self.dof_pos) - self.d_gains * self.dof_vel
        return torques

    def _reward_front_hip_neutral(self):
        """前髋关节尽量不要左右大幅张开。"""
        idx_fl = self.dof_name_to_idx['FL_hip_joint']
        idx_fr = self.dof_name_to_idx['FR_hip_joint']
        hip = self.dof_pos[:, [idx_fl, idx_fr]]  # [N, 2]
        # 惩罚 hip 偏离 0，给一个高斯型奖励
        return torch.exp(-4.0 * (hip ** 2).sum(dim=1))

    def _reward_front_feet_together(self):
        """前脚之间的水平距离不要太夸张（避免八字撑太开）。"""
        # 注意 feet_pos 已经在 _update_feet_states 里更新了: [N, num_feet, 3]
        fl = self.feet_pos[:, 0, :]  # 假设 feet_indices[0], [1] 是前脚；你那边已经这样设了
        fr = self.feet_pos[:, 1, :]

        # 只看平面距离（xy），不管高度
        diff_xy = fl[:, :2] - fr[:, :2]
        dist = torch.norm(diff_xy, dim=1)
        # 目标比如 ~0.12m，太宽开始掉奖励
        target = 0.12
        # 超出 target 的部分作为误差
        err = (dist - target).clamp(min=0.0)
        return torch.exp(-20.0 * err * err)

    def _reward_stay_still(self):
        """让 base 水平速度与航向角速度尽量小。"""
        lin_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)  # m/s
        yaw_rate = torch.abs(self.base_ang_vel[:, 2])  # rad/s
        # 0.05m/s 和 0.2rad/s 左右作为“很稳”的尺度
        return torch.exp(- (lin_xy / 0.05) ** 2 - (yaw_rate / 0.2) ** 2)

    def _reward_lin_vel_xy(self):
        return torch.norm(self.base_lin_vel[:, :2], dim=1)


    def _get_noise_scale_vec(self, cfg):
        """
        返回 shape = [obs_dim] 的噪声尺度向量，和当前 obs 的拼接顺序严格对齐。
        只在 BaseTask.__init__->_init_buffers 时被调用一次。
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = cfg.noise.add_noise
        ns = cfg.noise.noise_scales
        lvl = cfg.noise.noise_level

        idx = 0

        # base_ang_vel (3)
        noise_vec[idx:idx + 3] = lvl * ns.gyro * self.obs_scales.ang_vel
        idx += 3

        # projected_gravity (3)
        noise_vec[idx:idx + 3] = lvl * ns.gravity
        idx += 3

        # (dof_pos - default_dof_pos) (12)
        noise_vec[idx:idx + self.num_actions] = lvl * ns.dof_pos * self.obs_scales.dof_pos
        idx += self.num_actions

        # dof_vel (12)
        noise_vec[idx:idx + self.num_actions] = lvl * ns.dof_vel * self.obs_scales.dof_vel
        idx += self.num_actions

        # actions (12) ——一般不加噪
        # 如果你想做一点模拟执行抖动，可给到很小的值：
        # noise_vec[idx:idx+self.num_actions] = lvl * 0.01
        idx += self.num_actions

        # front/hind contact flags (2+2) ——离散标志，**不要**直接加高斯噪声
        # 如要鲁棒性：可以做“随机丢包”或“传感器迟滞”，见下方可选方案
        # idx += 4

        return noise_vec
