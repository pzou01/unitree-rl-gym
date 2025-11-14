from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class GO2HandstandCfg(LeggedRobotCfg):

    class env(LeggedRobotCfg.env):
        num_observations = 46          # 手倒立自定义 obs 3 +3 + 3 +12 +12 +12 + 1base height
        num_privileged_obs = 46        # 简单起见 = obs；如不用 teacher 也可以设为 None
        num_actions = 12
        # 其他保持和父类一致:
        # num_envs, env_spacing, send_timeouts, episode_length_s, test 等按需要沿用

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.28]  # 稍微低一点
        default_joint_angles = {
            # 前腿弯曲，把肩膀压低
            'FL_hip_joint': 0.3,
            'FR_hip_joint': -0.3,
            'FL_thigh_joint': 1.2,
            'FR_thigh_joint': 1.2,
            'FL_calf_joint': -2.0,
            'FR_calf_joint': -2.0,

            # 后腿往后撑一点
            'RL_hip_joint': 0.1,
            'RR_hip_joint': -0.1,
            'RL_thigh_joint': 0.3,
            'RR_thigh_joint': 0.3,
            'RL_calf_joint': -1.4,
            'RR_calf_joint': -1.4,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 35.0} # 刚度太大就会硬然后抖的厉害 --- 35-》 25
        damping = {'joint': 0.5} # 阻尼 大一点 0.5 -》 1
        action_scale = 0.3 # 防止抖动 可以把这个变小 0.3 -》 0.2
        decimation = 5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base","thigh", "calf", "hip","Head_lower","Head_upper"]
        self_collisions = 1

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            gyro = 0.2  # base_ang_vel
            gravity = 0.05  # projected_gravity
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1  # （这次没用到；留给你以后扩展）
            actions = 0.0  # 通常不对动作加噪（action_rate 正则会管）

    class rewards(LeggedRobotCfg.rewards):

        only_positive_rewards = False
        #base_height_target = 0.6
        base_height_target = 0.65
        soft_dof_pos_limit = 0.9

        class scales(LeggedRobotCfg.rewards.scales):
            # ========= 手倒立核心正奖励 =========
            # 倒立朝向（projected_gravity 对齐）
            orientation = 5.0

            # base 高度接近 handstand 目标
            base_height = 2.0

            # 前脚有支撑
            front_feet_contact = 3.0

            # 后脚离地（很关键）
            hind_feet_no_contact = 3.0

            # 关节接近 handstand pose_targets
            pose = 10.0

            # 不乱晃（平移/角速度小）
            stability = 1.0
            #anchor_drift = 1.5  # 对应 _reward_anchor_drift
            # ------- 为了保持不乱动
            stay_still = 2.0
            lin_vel_xy = -3


            # ========= 正则 & 安全（适中就好） =========
            # z 向速度 / xy 姿态震荡，我们已经用 stability 管了一部分，这里可以先关掉或很小，避免重复惩罚
            lin_vel_z = -0.001
            ang_vel_xy = -0.005

            # 扭矩 / 速度正则：轻一点，别压过主任务
            torques = -2e-4  # 2的负4到 3的负4
            dof_vel = -1e-5

            # 先关掉这个，等学会了再开
            dof_acc = 0.0

            # 动作平滑，给一点点
            action_rate = -0.006 # -5e-3  是-5x10的三次方 0.005 0.006 都是抖动的厉害

            # 关节超范围惩罚：保留但别爆炸
            dof_pos_limits = -1.0

            # 这些在 handstand 任务里用处小，可以先关掉
            dof_vel_limits = 0.0
            torque_limits = 0.0

            # 非法碰撞：比如大腿/小腿/hip 等，给大一点，让它不敢用乱撞刷 reward
            collision = -2.0

            # 控制前两个脚不要打开那么多
            front_hip_neutral = 2.0
            front_feet_together = 1.5

            feet_contact_forces = 0.0  # 先不用

            # ========= 终止项 =========
            # 对非 timeout 的摔倒/非法接触给惩罚；实际生效逻辑在 LeggedRobot.step 里
            termination = -5.0

            # ========= 全部关掉行走相关 =========
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            stumble = 0.0
            #stand_still = 0.0
            energy = 0.0  # 等站稳后想省电再开成负值
# handstand 目标姿态，来自你贴的 handstand keyframe（只包含关节，忽略 base 位姿）。
    """
    class handstand_pose:
        joint_angles = {
            # FL
            'FL_hip_joint':   0.0,
            'FL_thigh_joint': -0.686,
            'FL_calf_joint':  -1.16,
            # FR
            'FR_hip_joint':   0.0,
            'FR_thigh_joint': -0.686,
            'FR_calf_joint':  -1.16,
            # RL
            'RL_hip_joint':   0.0,
            'RL_thigh_joint': 1.7,
            'RL_calf_joint':  -1.853,
            # RR
            'RR_hip_joint':   0.0,
            'RR_thigh_joint': 1.7,
            'RR_calf_joint':  -1.853,
        }
    """

    class handstand_pose:
        joint_angles = {
            # 前腿：大腿抬一点、膝盖别弯那么狠，让腿更“撑直”
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': -0.89,  # 原来 -0.686，往 0 接近一点
            'FL_calf_joint': -1.5,  # 原来 -1.16，稍微“伸直”

            'FR_hip_joint': 0.0,
            'FR_thigh_joint': -0.89,
            'FR_calf_joint': -1.5,

            # 后腿：再团一点，避免踢到地，同时把重心更多压到前腿
            'RL_hip_joint': 0.0,
            'RL_thigh_joint': 1.7,  # 原来 1.7，稍微再抬一点
            'RL_calf_joint': -1.853,  # 原来 -1.853，再收一点

            'RR_hip_joint': 0.0,
            'RR_thigh_joint': 1.7,
            'RR_calf_joint': -1.853,
        }



    #
    # 如果你之后要 footstand 任务，可以用这个：
    #
    class footstand_pose:
        joint_angles = {
            # 来自 footstand keyframe
            'FL_hip_joint':   0.0,
            'FL_thigh_joint': 0.82,
            'FL_calf_joint':  -1.6,
            'FR_hip_joint':   0.0,
            'FR_thigh_joint': 0.82,
            'FR_calf_joint':  -1.68,
            'RL_hip_joint':   0.0,
            'RL_thigh_joint': 1.82,
            'RL_calf_joint':  -1.16,
            'RR_hip_joint':   0.0,
            'RR_thigh_joint': 1.82,
            'RR_calf_joint':  -1.16,
        }


class GO2HandstandCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go2_handstand'
