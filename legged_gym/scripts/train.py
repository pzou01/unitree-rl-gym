import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args): #  注册任务 根据名字创建环境 根据 名字创建算法 runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args) #对这个 环境实例化
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args) # 把环境实例化的对象 进行创建 runner实例化
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True) # 然后对实例化的runner 进行learn

if __name__ == '__main__':
    args = get_args()
    train(args)
