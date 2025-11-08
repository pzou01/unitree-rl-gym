from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil # gymapi 选择 物理引擎

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg # 这里就是保持一些设定好的参数的值

class LeggedRobot(BaseTask):

    # ----- 初始化
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg #  传入 cfg的参数
        self.sim_params = sim_params # 拿到simulation的值
        self.height_samples = None # 地形高度采样点 采样地面的高度 来判断地形 有 多种地形
        self.debug_viz = False # 是一个“是否打开调试可视化”的开关。 如果是true的话 在仿真的窗口会绘制绘制调试线、坐标、力矢量等； 用颜色或箭头显示奖励、接触力等调试信息。
        self.init_done = False # 初始化还没完成 最后 inti_done = true 就是完成了的意思
        self._parse_cfg(self.cfg) # 这个是解析配置 在仿真还没开始的时候我们先吧所有的参数解析 确定好来
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless) # 把观测的值 给到父类去初始化

        if not self.headless: # 如果不是headless 无头也就是 无图形界面 那就设置相机视角
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True # 初始化完成


    def _parse_cfg(self, cfg): #从配置文件 cfg 中取出参数 把人类可以读的单位 转变成 仿真中要用的数值 以步数 字典 张量为单位 存为self.xxx 后面使用
        # 一个step 执行多少次的sim 积分 然后 间隔多久一个step
        self.dt = self.cfg.control.decimation * self.sim_params.dt # dt 就是 control timestep  这里会不一样的是 decimation是这个动作你想持续多少个仿真步数 然后 x sim 就变成了每个step的时常
        # 观测 归一化 不同观测量的量级不同（速度、角度、力矩等），直接喂入神经网络不稳定 所以用 obs_scales 来对输入做缩放： 这里就是把这个参归一化的值拿出来而已
        self.obs_scales = self.cfg.normalization.obs_scales
        # 这里 class_to_dict() 也就是把reward的scales也是一个class 是一个类对象 变成字典  方便后面使用key去访问这些参数
        self.reward_scales = class_to_dict(self.cfg.rewards.scales) #这样后面就可以用self.reward_scales["torques"] 去进行使用
        # commands 代表机器人要追踪的目标命令（command targets）， 例如去哪里怎么动 你就是目标前进速度和 横向速度 角速度
        self.command_ranges = class_to_dict(self.cfg.commands.ranges) #这个也是一个class保持了3个速度的范围然后后面我们会“命令采样函数” _resample_commands() 里用来
        # 生成新的目标 也就是说 每个 episode 或每隔一段时间，环境会随机给机器人一个新的目标速度或角速度，机器人必须学会在这些命令范围内控制自己稳定前进。agent 就不是在学“走到固定方向”，而是在学“随时响应不同指令”——泛化性更强

        self.max_episode_length_s = self.cfg.env.episode_length_s # 一个episode最大的长度是多久 然后你除dt就可以知道执行多少step
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt) # 用 np.ceil() 向上取整；

        # domain rand领域随机化  定期给机器人干扰 push 隔多久给机器人一个推力
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt) # 隔多少s给机器人来一下除了dt就是隔多少step给机器人来一下

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        函数名 _init_buffers 就是 “初始化缓存区（buffers）”，
        它的核心作用是：

        🔹 在 GPU 上建立一系列 torch.Tensor 缓冲区，
        🔹 这些张量实时映射物理引擎（PhysX）的仿真状态，
        🔹 方便在 PyTorch 里直接读取环境状态、计算奖励、生成动作，
        🔹 同时也初始化控制需要的各种辅助变量（如 PD 增益、重力方向等）。

        一句话总结：_init_buffers() 把 物理仿真数据结构 → PyTorch张量接口建立了一个实时通信通道。
        """
        # get gym GPU state tensors 获取仿真状态张量
        # #向 Isaac Gym 申请一个可以读到物理引擎状态的 GPU 张量接口 只是在初始化阶段获取一个指向物理引擎内部内存的句柄 读取 PhysX 内存的实时视图 (zero-copy)。
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim) # （机器人）的 根部状态 root state 3[0,3] pos + 4 quat[3,7] + 3 lin_vel[7,10] + 3 ang_vel[10,13] 这里的速度和角度都是世界坐标下的
        # 除了 刚体树 建模体息中 root 根部的link 不依附任何其他的关节 其他地方都是这个的父节点开展的 quat这个4元姿态就是表示方向orientation  选择矩阵 4 x 4 那个 后面xyz就是 roll pitch yaw 前面的【1，000】表示没有选择
        # 所以 dof state 就是 其他所有的关节相对信息 是局部坐标系了
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # 只有关节的角度 和 角速度 对于 线速度 所有 link 的线速度、角速度都可以由 root 状态 + 所有关节角度与角速度 通过前向运动学（forward kinematics）计算出来；
        # 返回一个包含每个刚体（rigid body）所受**净接触力（net contact force）**的 GPU 张量。 第 i 个刚体在当前仿真步受到的总接触力向量（单位：牛顿）。 世界坐标
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim) # 对每个刚体（link、部件）； 把在该步仿真中所有与外界（地面、墙壁、其他物体）的接触力求和；存储成一个三维向量 (Fx, Fy, Fz)。
        # 然后在这里环境的step的时候会调用这里的 refresh 去更新物理状态
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        """
        isaac Gym 的底层物理引擎（PhysX）是用 CUDA C++ 写的。
        它在每个仿真 step 后，会把所有物理状态（位置、姿态、速度、接触力等）放在 GPU 内存的一块 buffer 里。
        但是这块 buffer 不是 PyTorch 的 tensor，而是 PhysX 内部的 GPU array，我们不能直接拿来参与 PyTorch 计算。
        所以就有了这个接口：import gymtorch 
        torch_tensor = gymtorch.wrap_tensor(raw_tensor)
        🔹wrap_tensor() 的作用就是
        “在不复制数据的前提下，把 PhysX 内部的 GPU buffer 包装成一个 PyTorch tensor 对象”。
        """
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # 把上面拿到的 root 和 dof 和 force 结果 wrap 处理后 就变成了pytorch可以使用 然后用pytorch的方法view等去进行处理
        # view 重新调整tensor的形状了  dof_state_tensor 的 shape 一般是 [num_envs * num_dof12个关节, 2] 2：每个关节有两个值 → 位置 或者 角度(position) 和速度或者 角速度(velocity)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] # 那就取出了最后一个维度的0的值也就是角度 “position” 是一个通用名字，用来表示该自由度的配置变量， 对转动关节就是角度，对滑动关节就是位移。
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1] # 取出最后一个维度的1位置的值 也就是速度 自由度的速度
        # 上面把结构从 [num_envs * num_dof, 2]  重新整理成三维结构 [num_envs, num_dof, 2]， 方便按照环境来查看每个机器人的状态了
        #  self.dof_pos → [num_envs, num_dof] 每个关节的角度 self.dof_vel → [num_envs, num_dof] 每个关节的角速度

        # root部分的代码
        self.base_quat = self.root_states[:, 3:7] # 4元quat 也就是 方向 3-7 【w,x,y,z]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat) # 把四元数转换成 roll/pitch/yaw 三个角度，更直观 变成了 [num_env,3]
        self.base_pos = self.root_states[:self.num_envs, 0:3] # 0-3也就是 xyz root的位置

        # 力从 shape = [num_envs * num_bodies, 3] 每一行是一个刚体（body）的 3D 接触力向量 [Fx, Fy, Fz]。 变成了 [num_envs, num_bodies, 3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
                                                                            3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on 训练/控制循环里会频繁用到的缓存变量”一次性准备好
        # 提前在 GPU 上为后续要频繁写入的变量“开一块固定形状的内存”，并初始化为 0。
        # 计数 和 杂项
        self.common_step_counter = 0 # 全局步数计数器；在 post_physics_step() 里自增，可用于周期性事件或日志
        self.extras = {} # 给算法返回的附加信息容器（例如 episode 汇总、time_outs）；在 reset_idx() 填充。

        # 噪声和参考向量
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # 观测噪声的逐维缩放系数向量；在 compute_observations() 里当 self.add_noise 为真时相加
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)) # 世界坐标系下的重力方向（每个 env 一份）；常配合四元数做坐标变换，见 projected_gravity
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1)) # 机器人机体系中的“前向”基向量；在 _post_physics_step_callback() 里用于计算朝向/heading 误差（转 yaw 命令）。

        # 控制 相关的缓存  动作/力矩/PD 增益
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False) # 待下发的电机力矩缓存；在 step() 中由 _compute_torques(self.actions) 写入，然后 set_dof_actuation_force_tensor() 下发。
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # 关节 PD 控制器的 P/D 增益；在本函数末尾“加载默认关节角 & PD 增益”那段 for 循环里按关节名填写。_compute_torques() 用到。
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) #
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False) # 当前/上一时刻的策略动作；
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        # 速度 缓存  便于奖励/观测/平滑
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # 上一时刻各关节角速度；用于 _reward_dof_acc()（角加速度惩罚）与平滑。
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13]) # 上一时刻根部线/角速度（6 维）；可用于诊断或自定义奖励/滤波；在 post_physics_step() 末尾更新。

        # 指令 高层 命令 跟踪目标
        # 由 _resample_commands() 随机采样或按课程更新；  在 compute_observations() 里拼进观测； 在跟踪奖励 _reward_tracking_lin_vel / _reward_tracking_ang_vel 中作为目标。
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading 任务命令缓冲（通常是 [v_x, v_y, yaw_rate, heading]）；

        # 命令的归一化系数（观测里用来把命令量纲/尺度对齐）；注：这里只给前三个（线速度 x/y、角速度 z）。
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, requires_grad=False, )  # TODO change this
        # 触地 和步态相关
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)# 每只脚离地累计时间；在 _reward_feet_air_time() 里： 离地计时、首次落地给奖励； 落地后清零（通过与 contact_filt 逻辑一起更新）。

        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False) # 上一时刻脚部接触状态；与当前接触（由 contact_forces 判断）做 OR/first_contact 逻辑，稳定步态事件检测。

        # 机体 系量  把世界系旋转到机体系） 也就是吧世界坐标下的的速度和角速度经过 4元方向 变成 和 以 base为视角的速度值
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # 体系下的根部线速度；用于观测与速度跟踪奖励（使学习对全局朝向不敏感）。
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 机体系下的根部角速度；用于观测与角速度跟踪/惩罚
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 机体系下的重力方向；其 xy 分量大小反映了俯仰/横滚偏离水平的程度，直接用于

        # joint positions offsets and PD gains 初始化每个关节的默认角度（默认姿态）和 PD 控制器的增益参数。 机器人初始姿态 和 控制刚度。
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # 先创建一个长度为 num_dof（关节数）的全 0 张量，用来存每个关节的“目标角度”
        # 循环每个关节
        for i in range(self.num_dofs): # 给每个关节一个“出厂默认姿态”角度，比如四足站立时膝盖微屈。
            name = self.dof_names[i] # 这就是所谓的 offset（偏置），因为之后控制器输出的目标角度是相对这个偏置的。
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle # 从dof 读取名字 然后 在 初始化的角度根据名字取出 角度 然后 存到这里
            found = False
            for dof_name in self.cfg.control.stiffness.keys(): # 给每个关节分配 PD 控制增益
                if dof_name in name: # 根据名字从cfg中取出每个关节 应该给到多少的kd 和kp
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name] #  stiffness（刚度） kp
                    self.d_gains[i] = self.cfg.control.damping[dof_name] #  damping（阻尼 kd
                    found = True
            if not found: # 如果没匹配到的关节 就直接设置为 0
                self.p_gains[i] = 0. # 一些关节不是主动控制的 例如 自由摆动的尾巴 配置文件没设置 就设置为0
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]: # 如果控制勒是是p和v的 也就是基于位置速度控制的没有kd和kp值就报错
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) # 扩展维度 把 [num_dof] → [1, num_dof]； 为了后面可以 broadcast 到多个环境（num_envs）

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

            为每个观测量分配对应的“噪声缩放系数”，
            用于在训练时给观测添加小扰动，增加学习的鲁棒性（robustness）。
            为什么要加噪声（核心目的）

            现实世界中，机器人传感器测量是不完美的： 线速度测量（IMU）有偏差； 姿态传感器（陀螺仪）有漂移； 编码器测量关节角有误差。
            如果训练环境里所有观测都是“完美的无噪信号”， 模型到真实世界（sim2real）时就会崩溃。
            所以我们在仿真中人为加入噪声，让 agent 学会： “即使观测略微不准，也能做出稳定决策。”
            这就是 domain randomization（领域随机化） 的一个部分。

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0]) # 初始化一个与单个观测维度相同大小的零向量，
        self.add_noise = self.cfg.noise.add_noise # 是否启用
        noise_scales = self.cfg.noise.noise_scales # 噪音强度
        noise_level = self.cfg.noise.noise_level # 权重
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel # 线速度
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # 角速度
        noise_vec[6:9] = noise_scales.gravity * noise_level # 重力方向
        noise_vec[9:12] = 0. # commands # 命令 也就是 目标速度  不加noise 也就是命令就是理想输入
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos #关节角度  模拟编码器误差
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel #关节角速度  模拟关节速度误差
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions # 上一时刻动作

        return noise_vec


    # ----- 环境创建 舞台 搭建
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        create_sim() 是整个物理世界的搭建入口函数：
        它做三件事：
        ① 初始化仿真引擎 →
        ② 创建地面（ground plane） →
        ③ 批量生成多个独立环境（envs）。

        为什么最后直接调用 _create_ground_plane() 和 _create_envs()？ 这是 面向对象封装（OOP） 的一种模式：
        create_sim() 是一个高层接口，统一“搭建整个仿真世界”的过程；
        _create_ground_plane() 和 _create_envs() 是下层具体实现；
         “_” 开头 → 表示是内部方法，不直接在外部被调用；
         但可以被子类（例如 Go1Env, HumanoidEnv）重写。
         这样做的好处是：✅ 把通用逻辑（创建世界）写在基类，把个性化内容（比如机器人结构）写在子类。
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly 重力轴的方向 z 轴 向上  四足、人形机器人 y轴是 某些工业机械臂环境
        #  创建仿真实例 这是 Isaac Gym 的底层 API 调用。 它相当于“启动一个物理世界”的过程。 可以理解为：“创建了一个空的世界”
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane() # 创建地面：
        self._create_envs() #  环境（env）是 Isaac Gym 的核心结构， 每个 env 是一个独立的、并行运行的小世界。 在每个环境中放一个机器人；

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        在物理仿真世界中加入一个无限大的地面平面（plane），
        并根据配置文件（cfg）设置它的物理属性：摩擦系数、弹性恢复系数等。
        为什么地面要单独创建？因为： 有些任务（例如 rough terrain 训练）会替换地面为高度图； 有些环境（机械臂）根本不需要地面；
        有些环境需要多个平面（如不同摩擦区域）。把地面单独封装成 _create_ground_plane() 就方便在子类中自由替换或重载。
        """
        plane_params = gymapi.PlaneParams() #  Isaac Gym 的地面参数对象。 PlaneParams 是一个结构体，包含定义平面属性的所有字段：
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) #  平面的法线方向（决定地面的朝向）
        plane_params.static_friction = self.cfg.terrain.static_friction #  静摩擦系数（物体开始滑动前的阻力）
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction #  动摩擦系数（滑动时的摩擦力）
        plane_params.restitution = self.cfg.terrain.restitution #  弹性恢复系数（碰撞后反弹的程度）
        self.gym.add_ground(self.sim, plane_params)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """

        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, 导入 机器人 文件 里面配置了 各种关节和 物理熟悉
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # 定位并且加载机器人配置文件 为什么要拆成 root + file？ Isaac Gym 的加载函数定义是
        #  gym.load_asset(sim, asset_root, asset_file, asset_options)
        # asset_root: 文件所在的目录 asset_file: 具体的文件名； asset_options: 载入时的配置选项（驱动模式、阻尼、是否固定 base 等
        # 最后也就是下面  robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)就把你的机器人模型真正加载进模拟器里了
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR) # 这个是写入的一个路径 /home/user/legged_gym/resources/robots/go1/urdf/go1.urdf
        asset_root = os.path.dirname(asset_path) #取目录的部分  只要  /home/user/legged_gym/resources/robots/go1/urdf
        asset_file = os.path.basename(asset_path) # 资产所在的文件夹路径  go1.urdf

        # asset_options = gymapi.AssetOptions() 是 Isaac Gym 加载资产（URDF / MJCF 模型）时的配置对象，用来告诉仿真器在加载模型时如何设置物理、视觉、关节和重力等特性。
        asset_options = gymapi.AssetOptions() # 通过设置这些选项，你可以： 控制模型的物理特性（密度、阻尼、惯量等） 控制几何简化（是否合并固定关节） 控制关节驱动模式（位置 / 速度 / 力矩）决定是否受重力作用等
        # 默认的关节驱动模式
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        # 将固定关节（fixed joints）合并为一个刚体 True：减少仿真中刚体数量（提升性能、降低内存）； False：保留固定连接的每个刚体（更精细的碰撞几何）。
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints # 通常都是 true
        # 是否把圆柱体形状替换为“胶囊体”（capsule）。 胶囊体对物理仿真更稳定（不易卡住、不易穿透）  尤其在四足机器人腿部碰撞中更推荐胶囊体。
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        # 是否反转视觉模型的坐标方向  有的模型导入后视觉 mesh 朝向与碰撞几何不一致，可以通过这个选项修正。
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        # 是否固定机器人的基座（base link）。  True：基座不会移动（常用于机械臂或固定底座的机器人）； False：基座自由移动（例如四足机器人、无人机）。
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        #设置刚体的密度，用于自动计算质量。 若 URDF 没有显式质量参数，会用 density × volume 计算。
        asset_options.density = self.cfg.asset.density
        # angular_damping  阻尼大 → 角速度衰减快（动作更“粘”）；
        asset_options.angular_damping = self.cfg.asset.angular_damping
        # 线性速度阻尼（平动阻尼）。 影响速度衰减速度。太大会让机器人“像泡在水里”一样；太小可能抖动。
        asset_options.linear_damping = self.cfg.asset.linear_damping
        # 限制刚体的最大角速度。 防止数值发散（例如强力碰撞时角速度爆炸导致不稳定）。
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        # 限制最大线速度。
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        # 关节转动的“附加惯性项”。
        asset_options.armature = self.cfg.asset.armature
        #  用于碰撞形状的“厚度偏移”。
        asset_options.thickness = self.cfg.asset.thickness
        #  是否禁用重力。
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # 最后这里的  就把你的机器人模型真正加载进模拟器里了
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # 然后封装起来的 robot asset 就保护了所有的资料 描述这个机器人长什么样、有哪些部件、物理特性是什么
        # 然后就可以从这个模版中去拿信息资料了
        self.num_dof = self.gym.get_asset_dof_count(robot_asset) # 从 机器人配置文件 asset 中能拿到关节数量是多少
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # 多少个刚体 例如  机身、四条腿、每条腿 3 段 = 1 + 12 = 13 个刚体。
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset) #  获取每个关节的属性结构体 包括关节的角度上下限 最大速度 和 最大力矩 pd值 控制模型
        # rigid shapes 碰撞形状 摩擦系数 恢复系数 碰撞厚度 滚动摩擦  扭转摩擦 等
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset) # 获取模型所有刚体**碰撞形状（rigid shapes）**的物理属性

        # 然后在创建环境的时候 就会根据这些模版的信息 去 进行 调整 最后实例化为真正的机器人


        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])


    # ------ 智能体 添加 演员 放上舞台并且设置属性

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties

            环境初始化阶段 被调用的一个回调函数 用来随机化 每个环境中物体 rigid shape 摩擦系数  以实现 domain randomization
            在不同的仿真环境（env）中，为机器人的接触物体设置不同的摩擦系数。
            这样可以让机器人学到在不同地面摩擦力下都能稳定行走，而不是只适应单一摩擦环境。
        """
        # 在不同的仿真环境（env）中，为机器人的接触物体设置不同的摩擦系数。
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range # 是否启动 摩擦税计划
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id): #props：由物理引擎（例如 Isaac Gym）在加载 URDF 模型时传进来的关节属性集合，类型是 numpy array 或结构体。
        """ Callback(callback（回调函数） = “当某件事发生时系统自动帮你调用的函数”。)不是你主动去调用它，而是程序框架在特定时机“回过头来”调用它。
            allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties 从 props 中提取每个 DOF 的属性
        """
        # 对关节属性进行限制 给到位置和速度以及扭矩的限制
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id): # 上面是 rigid shape 刚体的几何外表 所有是表面的一些特性
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass # 这里是 一个独立运动的body 物体 有质量 惯性 重力
        if self.cfg.domain_rand.randomize_base_mass:# 如果有这个mass的值 那我们就把
            rng = self.cfg.domain_rand.added_mass_range # rng random range 随机质量的变化范围
            props[0].mass += np.random.uniform(rng[0], rng[1]) # 这里props [0] 是 baselink 也就是主体去干 然后 我们会随机给到mass的值去进行 扰动
        return props


    # ----- 探索机制 让 数据多样和 有调整 域随机化 随机指令 扰动 随机重置

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations post physics step callback = “物理步结束后的回调”
        ，   物理引擎已经根据上一帧的 action 完成积分、更新了刚体的位置、速度、接触等；接下来，Isaac Gym 在进入“计算奖励 / 判断 done / 生成观测”之前，会自动执行这个 callback；
            这个函数就是让你在仿真结果出来后、但奖励与观测还没算之前，对环境做一点补充处理，比如：更新或重采样新的 command（为下一步准备目标）；
            根据当前姿态计算新的朝向角速度命令；记录地形高度、施加随机推力等。
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
            根据目标和航向计算角度速度指令，计算测量的地形高度并随机推动机器人。
            也就是 施加控制指令（action）
            2. 物理引擎进行一次积分（physics step）
            3. ✅ _post_physics_step_callback() ← 在这里执行
            “commands”就是对机器人要做什么的目标要求，通常会作为观察量的一部分喂给策略网络，且奖励函数也会用它来计算“跟踪误差”。
            也就是说我希望机器人在这个状态下 执行 command是什么也就是我希望你速度和方向是什么，然后你观测就是在这个状态下 输入了 command 我策略学会输出action
            来告诉每个关节要做什么
            在训练的时候 会 被定时 重新采样 让机器人学会跟随目标速度和 朝向
            把“下一步要跟随的目标”先准备好，装进“下一步要给策略看的观测”里。
            在这里执行是为了下一个step的命令做准备
            4. 计算终止条件（done）
            5. 计算奖励（reward）
            6. 收集观测值（observation）
        """
        #在每个物理仿真更新完成后（机器人状态更新完、力已经施加完），系统会自动“回调”这个函数，让你在计算奖励、done 条件和 observation 之前做一些额外操作
        # 哪些环境需要重新采样命令 找出所有刚好走到“该换命令”那一步的环境 env_ids；这些环境调用 _resample_commands，产生新一批 commands（目标指令）。
        # 直观理解：比如每 0.4 秒换一次命令，dt=0.02s，那每 20 步换一次目标速度/朝向
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)# 对这些环境 重新生成目标命令 例如 期望速度 角速度
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)  # 根据当前 base 的姿态 self.base_quat，计算它的“前向向量”；
            heading = torch.atan2(forward[:, 1], forward[:, 0]) #用 atan2(y, x) 算出当前朝向角 计算 目标朝向 (self.commands[:, 3]) 与 当前朝向 (heading) 的差值；
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) #用 wrap_to_pi 把差值映射到 [-π, π]； 乘以 0.5（缩放），再 clip 到 [-1, 1]。
            # self.commands[:, 2] 就变成了 “根据当前朝向误差自动计算的转向命令”； 也就是机器人需要的 角速度指令（yaw rate command）

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed

        对部分的环境 随机生成新的 commands
        期望的前线速度 横线速度
        期望的绕z轴角速度
        和期望头的朝向
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        随机给机器人一个推力冲击impulse 测试平衡和恢复能力
        """
        env_ids = torch.arange(self.num_envs, device=self.device) # 哪些环境要推
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0] # 计算多少步之后推一次
        if len(push_env_ids) == 0: # 没有要推的就直接返回
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy # 推力从这个线速度上进行
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), # 7到9是base身体的线速度xy
                                                    device=self.device)  # lin vel x/y

        env_ids_int32 = push_env_ids.to(dtype=torch.int32) #把刚才的 rootstate 同步回传到物理引擎中 也就是告诉 isaac gym这些有新的速度了 用这个来仿真
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default  positions.
        Velocities are set to zero.随机初始化机器人各关节角度，让姿态多样化，同时关节速度清零
        重置关节的位置速度 位置随机从0.5到1.5 乘 启动位置 这样每次都不会站在一样 防止策略只在某个状态又用 并且关节速度都是0 静止

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
            重置 root的状态 也就是 位置和速度 速度随机 位置 是中心点位置的 一米左右
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        curriculum 是 课程 也就是 逐步增加难度
        我们不是一开始就给机器人很难的任务（比如跑得又快又稳），而是先让它学简单的，比如慢速前进、小角度转向，当它在这些简单任务上表现很好时，再扩大命令范围，让它尝试更极端的目标

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        # 如果在最近的线速度的跟踪奖励超过平均值80以上了就说明学会当前阶段的任务了 就要加难度了  速度变大了
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    # ---- 推动训练  把动作变成力矩 并且推进一次 训练步 产出 观测 奖励 结束 幸好

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()
        那就是 把 策略传入的action 开始变成了每个关节的扭矩然后推进物理 然后 产出obs reward 和 done

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions # 这里把动作归一化限制在一个范围呢
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device) # 这里是 -100到+100
        # step physics and render each frame
        self.render() # 如果开了可视化 那这里就更新画面 对物理没有影响
        for _ in range(self.cfg.control.decimation): #控制-物理解耦：decimation 循环 一个控制步里执行多次物理步（decimation）。
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)  # 这里就是action到torques的mapping
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # 把刚才算好的 扭矩给到引擎
            self.gym.simulate(self.sim) #推进一步 物理 积分
            if self.cfg.env.test: # 测试节拍对齐 让模拟时间和 真实时间对齐 方便可视化
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)

            if self.device == 'cpu': # 同步和刷新状态 refresh
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim) #为下一轮计算做准备
        self.post_physics_step() # 这里是物理步结束后的回调 会调用刚才那个  post physics step callback 然后调用 重新采样command

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs) # 观测也一个 clip  这里也是 -100 到 +100
        if self.privileged_obs_buf is not None: # 特权观测 给critic 看的 全面信息 在真实部署中不可以看到 actor看到的是感知的信息 我们看到更多的信息 只给critic
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras # 在 post_physics_step()内部有后三个值的调用 等下下面看就看到了

    def _compute_torques(self, actions):
        """ Compute torques from actions. 把action变成了位置和速度的目标通过 pd控制 然后 直接变成 torques
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale  # 这里是要把action 进行平滑变化 传入的action x 0.5了 在这里
        control_type = self.cfg.control.control_type # 控制模式 p 位置 v 速度 t 扭矩 # 这里config给的是p模式
        if control_type=="P": # 相对这个 目标位置的相对便宜 default 的dof pos
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # 然后也是一个clip去控制

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim) # 把引擎里最新的根状态（位置/姿态/线/角速度）和净接触力拷到可读的张量里
        self.gym.refresh_net_contact_force_tensor(self.sim) # 便于后续计算。（物理步刚跑完，先把状态“刷新到手”）

        self.episode_length_buf += 1 #每个 env 的步数 +1；全局步计数 +1。（重置判断、课程学习等会用）
        self.common_step_counter += 1

        # prepare quantities 提取位姿：位置、四元数、以及把四元数转成欧拉角（方便可视化/调试或某些奖励项）
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:]) # 四元 变成 roll pitch yaw 把速度与重力都转到机体坐标系（body frame）：
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) # 线速度 线/角速度以机体前/左/上为轴，便于做“前向速度跟踪”等奖励；
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13]) # 角速度
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec) # 把重力投到机体系可用于躯干姿态稳定奖励（重力方向应接近 -z 轴）。

        self._post_physics_step_callback() #物理步之后、奖励/观测之前的“钩子” 若到点：重采样 commands（为下一步准备新目标）； 若启用 heading：把目标朝向转成期望 yaw 角速度；
        # 关键点：这里产生的新 command 会进到下一步的观测；本步奖励通常仍用“本步的旧 command”评估（避免 off-by-one）。
        # compute observations, rewards, resets, ... #对下一个 命令进行了预估然后开始计算了 done和reward了
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids) # 对需要重置的 env 调 reset_idx（里面通常会调用你前面看到的 _reset_dofs()、_reset_root_states()、以及 update_command_curriculum() 等）。

        if self.cfg.domain_rand.push_robots:
            self._push_robots() # 如果开了扰动训练：在计算完奖励与 reset 之后，对仍在跑的 env 注入一次随机“推搡”（改 base 线速度）——影响下一步观测，让策略学会抗扰动。

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions) 生成下一步要喂给策略的 obs（把最新状态、以及刚准备好的 command打包进去）。

        # 缓存“上一帧”的动作/速度，用于下帧的差分项（比如 V 控制的加速度差分、平滑/正则项、或诊断）。
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 违规接触：不该着地的部位与地面发生了“有效接触”
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        #姿态超限：机体的俯仰/横滚角太大（翻车/倾倒） # pitch 限制（约 57.3°） # roll  限制（约 45.8°）
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
        # 超时：达到最大步长（不算“终止惩罚”，只是正常回合结束）
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        在多环境仿真（vectorized environments）中，比如 4096 个机器人同时训练；每个机器人（环境）都有自己的状态、奖励、done；不是所有环境同时终止，有些摔倒了，有些还在跑；
        所以——我们只需要 重置那些 done 的 envs；这些 “done 的环境” 的 ID 会被 check_termination() 记录在 reset_buf；然后 post_physics_step() 把它们提取出来：
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)


于是：
👉 reset_idx() 就是“只重置指定 env_id 的环境”，以便让它们重新开始新一回合。

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset robot states 重置关节状态 baseroot的状态  重新采样新的command
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers 清空这一回合的相关的缓存
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras 把结束的 env在这一个回合各个reward 平均值汇总
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum: # 把当前最大的速度记录下来 后面加难度
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm  区分哪一些是超时的 因为 超时不给惩罚
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0. #把当前的环境的所有奖励清零  self.rew_buf 是 shape = (num_envs,) 的向量，代表每个环境当前步的总 reward。
        for i in range(len(self.reward_functions)): #遍历所有 reward 函数值乘上权重 然后最后加到总的奖励去 然后累计episode sum
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards: #  可选地裁剪负奖励 一些任务（尤其早期的 locomotion baseline）会采用“只保留正奖励”；
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping 处理终止奖励
        if "termination" in self.reward_scales:  #若定义了终止奖励项（常见为摔倒惩罚）；
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, #base的线速度 当前移动速度 * 缩放因子
                                  self.base_ang_vel * self.obs_scales.ang_vel, # 角速度
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale, # 3代表期望的角速度 0是期望线速度x 1 是期望线速度y 3是期望角速度 yaw z轴 因为 roll pitch yaw
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #关节相对默认姿态的偏差
                                  self.dof_vel * self.obs_scales.dof_vel, #关节角速度
                                  self.actions # 动作
                                  ), dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise: # 你看看要不要加noise
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec



    #------- 管理和汇总 筛选权重>0的奖励项，权重乘以 dt，收集成可调用列表

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions 根据reward的sacles 的key来准备reward的清单 然后取进行 一个个加  因为sacle保存了所有的reward的权重和对应的名字 就代码要计算那么多个reward
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    #------------ reward functions 奖励函数的设定 19个  ----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) #取平方 z轴 如果上下 都懂就惩罚 z轴越大说明不稳定
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) # 这里dim1把 roll和pitch x和y轴的角速度都平方 不希望晃动

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) #这里也是 世界重力是  0 0 -1 如果站着重力就是 对的 如果斜了 x和y就不是0了然后就要 取值代表身体倾斜情况

    def _reward_base_height(self):
        # Penalize base height away from target 约束机体高度 z 轴
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target) # 在一个base的高度 不要乱跳或者趴地
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1) # 如果扭矩太大了 就惩罚 控制平滑

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1) #关节角速度转动大 惩罚

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) # 角速度变化大爷惩罚 计算的是离散的加速度 通过速度计算加速度
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1) #动作变化 过大

    # ------- 安全约束
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) #被选中的身体部位的接触力 如果大于0.1也就是碰到东西了 就会计算每个环境多少部位碰到东西
    # 从而禁止 部位碰撞
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf #对于摔倒或者失败给到的终极惩罚
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit 关节上下限 低于最低和大于最高的 这些偏差加起来 就是总的超出量 然后 进行惩罚
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self): #超出了速度的极限控制
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self): # 超出最大扭矩
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes) 对平面的速度跟踪 命令的速度的值和你身体的值 这里给的就是一个
        # exp(-error /a ) 误差越小，指数越接近 1；误差大，奖励迅速衰减到 0。 tracking_sigma：决定“容忍度”。越大 → 奖励曲线更平缓；越小 → 要求更严格
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)  角速度  身体z 轴角速度
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps 鼓励“步幅/摆动时间适中偏长”**的奖励，细节很多
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1. #足端 z 向接触力超过 1N 视为接触（去掉噪声）
        contact_filt = torch.logical_or(contact, self.last_contacts)  #  对 PhysX 网格接触不稳定做个“去抖”滤波。
        self.last_contacts = contact # 保存上一帧接触状态。
        first_contact = (self.feet_air_time > 0.) * contact_filt #  首次触地时结算奖励 只有“刚落地的那一刻”（上一段时间在空中，现在检测到接触）才触发结算。
        self.feet_air_time += self.dt # 只有触地这一瞬间按这只脚“空中时间减 0.5s”给奖励
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground 空中时间 > 0.5s → 正奖励（步子较长/节奏较慢）
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command 空中时间 < 0.5s → 负奖励（步子过密/小碎步）
        self.feet_air_time *= ~contact_filt # 触地的脚清零重新计时；仍在空中的脚继续累计
        return rew_airTime
    
    def _reward_stumble(self): #
        # Penalize feet hitting vertical surfaces 惩罚 罚“脚撞墙/绊脚
        # ：如果某只脚的水平力 ≫ 竖直支撑力（阈值系数这里取 5），就视为“脚在横向撞击（例如踢到了立面/台阶边）”，容易“绊一下”。
        #只要任意脚满足就记一次。返回值：布尔（True/False）。在总体 reward 里乘以负的 scale 时会自动转成 0/1（True→1，False→0）当作惩罚计数。直觉：正常落脚应该以竖直支撑力为主；横向力过大像是在刮擦/撞击。
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1) #水平接触力（x、y 分量） 竖直接触力（z 分量）
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands 0 指令时要站稳别乱动  平面速几乎为0的时候 就别抖和动
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces 罚“落脚太猛/冲击过大”
        #与阈值 max_contact_force 比较；只对超过阈值的部分计惩罚（用 clip(min=0.) 把没超的置 0）。作用：防止“砸地板”“硬着陆”，鼓励轻柔、可控的触地，提高舒适性与安全性（也更利于真实机器人
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1) # 看的是每只脚的接触力模长（包含 x,y,z）
