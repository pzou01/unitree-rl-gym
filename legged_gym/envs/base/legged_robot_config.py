from .base_config import BaseConfig

"""
这些嵌套 class 是“命名空间式配置”

class LeggedRobotCfg(BaseConfig): 里面这些：

class env

class terrain

class commands

class init_state

class control

class asset

class domain_rand

class rewards

class normalization

class noise

class viewer

class sim

class sim.physx

以及 LeggedRobotCfgPPO 里的 policy/algorithm/runner

都不是“要实例化才生效的逻辑类”，而是 配置容器：

用法就是：

env_cfg = LeggedRobotCfg()
# 或者直接用 class_to_dict(LeggedRobotCfg)

num_envs = env_cfg.env.num_envs
reward_scales = env_cfg.rewards.scales.__dict__
...



"""

class LeggedRobotCfg(BaseConfig):

    class env: # 整体环境
        num_envs = 4096 # 4096个并行的环境 
        num_observations = 48
        num_privileged_obs = None # 不返回 if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm 告诉done 是因为 摔倒还是自然解释 如果是 摔倒了 critic就要清零  时间到的话 critic 目标还是 bootstrapping
        episode_length_s = 20 # episode length in seconds  20 / simdt = 4000步
        test = False

    class terrain:
        mesh_type = 'plane' # "heightfield" # none 无地形 , plane 无限平地 , heightfield 高度场  or trimesh 三角网络 复杂人工地形
        horizontal_scale = 0.1 # [m]# 水平方向网格间距
        vertical_scale = 0.005 # [m] # 垂直方向高度变化尺度
        border_size = 25 # [m] # 边界大小 [m]
        curriculum = True # 表示难度渐进式训练。 环境会根据机器人表现自动提升地形难度。
        static_friction = 1.0 # 静态摩擦
        dynamic_friction = 1.0 # 动态摩擦
        restitution = 0.
        # rough terrain only:
        measure_heights = True # 机器人观测中包含地形高度信息 x 17个 y11个是否提供地面高度观测
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # 控制 地图矩阵
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state 表示一开始的最大初始等级
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels) 地形分成 10x20的patch 每个patch 是一个地形样本（平坡、台阶、离散块等）
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2] # 定义不同地形类型的比例 10的 平滑坡 10的粗滑坡 35的台阶上 25的台阶下 20的离散快
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces 三角网络 如果局部坡度超过阀值 就把面改为垂直

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s] 10s换了一次命令
        heading_command = True # if true: compute ang vel command from heading error 如果为 True，则根据 heading 误差计算 yaw 角速度命令
        class ranges: # 命令范围
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset: # 模拟器（Isaac Gym）如何加载这个机器人的“模型资源 asset 是连接 URDF 模型和仿真引擎的“物理配置接口
        # asset 就是告诉仿真器：
        # “我的机器人长什么样、哪些部位重要、用什么方式受力、出错时该怎么处理。”
        file = "" # urdf
        name = "legged_robot"  # actor name 模型实例名字
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = [] # 指定身体哪些部分接触地面时 会被惩罚（负奖励）。
        terminate_after_contacts_on = [] # 指定哪些部分一旦碰到地面就 立即结束 episode（摔倒）。
        disable_gravity = False # 关闭重力（悬浮调试时用）
        collapse_fixed_joints = True #  将固定关节的 body 合并为一个整体，加速计算 merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False #  固定机器人底座（= 不动），调试姿态时用 fixe the base of the robot
        default_dof_drive_mode = 3 #  关节控制模式（GymDofDriveModeFlags） see GymDofDriveModeFlags (0 is none 被动自由体, 1 is pos tgt 关节朝指定角度驱动  , 2 is vel tgt 控制角速度, 3 effort直接施加力或扭矩)
        self_collisions = 0 #  自碰撞设置（0=启用，1=禁用） 1 to disable, 0 to enable...bitwise filter
        # 几何和碰撞体优化
        replace_cylinder_with_capsule = True #  把碰撞体从圆柱体改成胶囊体（两端带半球） 更稳定、更快，不容易产生物理抖动。 replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up 把视觉模型从 “Y-up” 转换为 “Z-up” 坐标系这个 unitree的 机器人是躺进来的， 要变到z轴去，也就是 机器人的向上是y轴 但是isaac 上是z轴 我们要调整对应isaac

        # 动力学属性（物理仿真层）
        density = 0.001 #材料密度（kg/m³） 影响质量计算
        angular_damping = 0. # 角速度阻尼 模拟空气阻力
        linear_damping = 0. # 线速度阻尼 模拟空气阻力
        max_angular_velocity = 1000. # 最大角速度限制 防止爆算
        max_linear_velocity = 1000. # 最大线速度限制  防止数值发散
        armature = 0. # 电机转子惯性 一般为 0
        thickness = 0.01 # 碰撞体厚度偏置 避免模型重叠

    class domain_rand: # domain_rand 是 Domain Randomization（域随机化） 的配置区块，
        # 在强化学习中，如果你只在固定参数的仿真环境中训练机器人（比如摩擦力始终是 1.0），
        # 那么模型学到的策略只在那个“虚拟世界”里有效。
        # 而现实中的地面摩擦、质量、惯性、扰动永远不一样。
        randomize_friction = True # 是否在训练时随机化地面摩擦系数。 每个 episode 随机取不同摩擦力；
        friction_range = [0.5, 1.25] #  摩擦系数随机范围。
        randomize_base_mass = False  # 是否随机化机器人本体质量。
        added_mass_range = [-1., 1.] #随机附加质量范围（单位：kg 或相对比例，取决于实现）。
        push_robots = True # 是否在训练过程中随机“推”机器人。
        push_interval_s = 15 #  扰动的间隔时间，单位秒。
        max_push_vel_xy = 1. # 推的最大线速度（m/s），

    class rewards:
        # 可以看到这里 sacles 是一个类对象 我们后面使用也是通过变成字典来使用
        class scales:
            termination = -0.0 # 终结 
            tracking_lin_vel = 1.0 # 追踪线速度 
            tracking_ang_vel = 0.5 # 追踪角速度 
            lin_vel_z = -2.0 
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1. # 基本高度目标是1 这是人形机器人的高度吧 
        max_contact_force = 100. # forces above this value are penalized

    class normalization: #归一化的参数 为了让输入数据在训练时数值稳定
        # 强化学习的神经网络（Actor/Critic）输入的观测量（observation）来自物理世界：有的量特别大（速度几米/秒），有的特别小（关节角度几十分之一弧度），如果不归一化，网络会很难学，因为不同维度的数值尺度差太多。
        # 所以这个模块的目标就是：把观测数据和动作缩放到“合理的数值范围”（通常在 -1 到 1 之间），让网络学习更稳定、梯度传播更顺畅。
        # 定义了每个观测维度的缩放系数（scale factor），
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise: #对观测给噪声 让策略对感知误差更鲁棒（Robustness）
        """
        在真实世界中，机器人的传感器（IMU、编码器、高度传感器等）都会存在噪声：
        角速度会抖动，加速度计有偏置，编码器读数不准。如果仿真环境里一切都是完美的，那么训练出的策略一旦放到真实机器人上，很可能因为微小的测量误差就完全失效。
        所以这个 noise 模块的目的就是：在仿真训练中人工加入随机噪声，让网络学会“忽略微小误差”，提升鲁棒性和现实可迁移性（sim2real）。
        """
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01 # 关节角度噪声 模拟编码器读数误差（±0.01 弧度）
            dof_vel = 1.5 #  关节角速度噪声 模拟速度传感器误差
            lin_vel = 0.1 #机身线速度噪声 模拟 IMU 测得的线速度抖动
            ang_vel = 0.2 #机身角速度噪声 模拟陀螺仪噪声
            gravity = 0.05 #重力方向噪声 模拟 IMU 姿态估计误差（如倾角传感器漂移
            height_measurements = 0.1 #地形高度测量噪声 模拟激光/高度传感误差

    # viewer camera:
    class viewer:
        """
        在 Isaac Gym / legged_gym 里，一个环境通常包含上千个并行仿真（比如 num_envs = 4096），但你不可能同时显示 4096 只狗。
        所以系统通常会：选择其中一个（或少数几个）环境用于可视化；并在 GUI 窗口中渲染；由 viewer 参数定义摄像机的位置和目标。换句话说：
        viewer 就是控制“你在屏幕上看到什么、从哪里看”的那组参数。
        """
        ref_env = 0 # 表示 第一个环境  可以设置其他的来看第n个机器人
        pos = [10, 0, 6]  # [m] 摄像机位置 坐标 在 x y z
        lookat = [11., 5, 3.]  # [m]摄像机看的目标点 相机镜头对准的位置

    class sim:
        dt =  0.005
        substeps = 1 #每个 dt 里再细分多少内部子步。 1 就表示不用再细分  如果仿真不稳定，可以调大（更准但更慢）。
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z 指明“世界坐标系哪个方向是向上”

        class physx: # PhysX 的低层设置，用来平衡稳定性 vs 性能。按用途给你讲
            num_threads = 10 #用多少 CPU 线程来处理仿真（尤其刚体、碰撞预处理）。 多环境 + 大地形时开大一点可以加速
            solver_type = 1  # 0: pgs, 1: tgs 碰撞/约束求解器类型：0: PGS（projected Gauss-Seidel），传统松弛法，简单但有时软。 1: TGS（temporal Gauss-Seidel），更稳定，适合高频率、复杂接触。
            num_position_iterations = 4 # 每个时间步中，约束（关节、接触）解算迭代次数 越高 → 关节/接触更硬、更不穿模，但更慢。
            num_velocity_iterations = 0 #  速度约束迭代次数
            contact_offset = 0.01  # [m]碰撞壳的“缓冲厚度”，在真实形状外面再包一层 0.01m： 提前一点检测到接触； 降低互相穿透后才纠正的情况。
            rest_offset = 0.0   # [m] # 静止接触时的偏移量（碰撞形状表面间留多大缝）。
            bounce_threshold_velocity = 0.5 #0.5 [m/s] # 低于这个速度的碰撞不产生弹跳，只做非弹性接触
            max_depenetration_velocity = 1.0 # 纠正“穿透”时，最多允许物体被“弹回去”的速度。
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more GPU 上允许处理的最大接触对数量上限。 并行环境很多（比如 4k / 8k）时，需要够大防止溢出。
            default_buffer_size_multiplier = 5 # 内部缓冲区放大系数，给接触、约束等预留内存。 增大一点防止 “buffer overflow” 错误。
            contact_collection = 2 # 0: never, 1: last sub-step 只记录最后一个子步；, 2: all sub-steps (default=2)记录所有子步（更精细） ---- 控制输出多少接触信息：

class LeggedRobotCfgPPO(BaseConfig): # 给算法ppo的配置文件
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt