from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2RoughCfg( LeggedRobotCfg ): #基础了 leggedrbotcfg 然后进行的专门的参数修改针对go2的
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m] #初始化的位置
        default_joint_angles = { # = target angles [rad] when action = 0.0 初始化的定义的角度吗 12个关节
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P' # P 模式
        stiffness = {'joint': 20.}  # [N*m/rad] 刚度 kp
        damping = {'joint': 0.5}     # [N*m*s/rad] 阻尼 kd
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25 # 从 0.5 变 0.25了  q targe = q default + 0.25 * action
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4 # 输出 action 这个动作 跑4个物理步

    class asset( LeggedRobotCfg.asset ): # 模型接触的规则
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf' # 文件 名字
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]  # 惩罚的 部位 大腿 hip到knee的link 和 小腿 knee 到 ankle的link
        terminate_after_contacts_on = ["base"] # base 接触就done
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter # 根据注释是关闭/过滤自碰撞（具体逻辑看 base 实现）。
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25 # 目标高度改了
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class GO2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 # 算法的 entropy
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'

  
