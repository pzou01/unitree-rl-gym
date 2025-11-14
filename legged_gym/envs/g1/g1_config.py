from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47 # 这里变成了 47个观测了 在env中去仔细看看是什么情况
        num_privileged_obs = 50 # 然后有50个 特权空间 在 env 中研究一下
        num_actions = 12 #还是12个关节动作 但这次的12个是怎么理解的


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True # 随机
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters: 刚度 阻尼
        stiffness = {'hip_yaw': 100, # 髋关节 绕 z轴
                     'hip_roll': 100, # hip 绕 x轴
                     'hip_pitch': 100, #hip 绕 y轴
                     'knee': 150, # 膝盖 定的 绕 pitch
                     'ankle': 40, # 这个 有两个方向的 但是这里就定义了一个 env 检查一下
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT 一个指令 执行4次仿真
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll" # ankle roll 这个地方 就是 foot 对于 ankle pitch 不是foot 的 link
        penalize_contacts_on = ["hip", "knee"] # 跨和膝盖落地就惩罚
        terminate_after_contacts_on = ["pelvis"] # 这个是腰部把
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter 0是启动自碰撞 不允许自己发生碰撞
        flip_visual_attachments = False # 视觉模型 就是y 这里没动啥意思

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9 # 软限制 到了 0.9 就开始惩罚
        base_height_target = 0.78 # 目标高度控制在 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0 # 这个是命令的 的速度追踪
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0 #父类写好了
            dof_acc = -2.5e-7 # 关节角速度
            dof_vel = -1e-3 #关节线速度
            feet_air_time = 0.0 # 这里空中的 时间 居然没开 ？
            collision = 0.0 # 碰撞也没开
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15 # 这个是新的
            hip_pos = -1.0 #
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8 #噪音的方差
        actor_hidden_dims = [32] #为什么隐藏层反而变小了
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
