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
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
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

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props


    # ----- 探索机制 让 数据多样和 有调整 域随机化 随机指令 扰动 随机重置

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
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
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2),
                                                    device=self.device)  # lin vel x/y

        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

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

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5,
                                                          -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0.,
                                                          self.cfg.commands.max_curriculum)

    # ---- 推动训练  把动作变成力矩 并且推进一次 训练步 产出 观测 奖励 结束 幸好

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
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
        # prepare list of functions
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
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
