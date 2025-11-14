from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

# 新增 handstand go2的任务
from legged_gym.envs.go2.go2_handstand_config import GO2HandstandCfg, GO2HandstandCfgPPO
from legged_gym.envs.go2.go2_handstand_env import GO2HandstandEnv


from legged_gym.utils.task_registry import task_registry
# 传入了 Legged_ROBOT 机器人配置  和 对应的机器人的 config 和 算法的 config 和名字 然后在这里通过registry 进行了注册
#  这里会把 leggedrobot 基础后变成env 然后传入对应的 robot中去

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

#新增 handstand go2的任务
task_registry.register("go2_handstand",GO2HandstandEnv, GO2HandstandCfg(), GO2HandstandCfgPPO())

