'''
run batch training

im
res=[]
[res.append(a) for a in "--exp_name sacFishMovingVisualServoC-v0 --env_name 'FishMovingVisualServoContinous-v0' --num_agent_train_steps_per_iter 2000 --batch_size_initial 5000 --batch_size 5000 --n_iter 10 --video_log_freq -1 --sac_discount 0.99 --sac_n_layers 2 --sac_size 256 --sac_batch_size 1500 --sac_learning_rate 0.0003 --sac_init_temperature 0.1 --sac_n_iter 5000 --mbpo_rollout_length 0".split(' ')]
print(res)

'''
import subprocess
import os
import time
import numpy as np
from multiprocessing import Process

def run_process(command):
    subprocess.call(command)
    time.sleep(0.05)

algos= ['mbpo', 'sac']
processes = []
ENVS= ['cheetah-cs285-v0','reacher-cs285-v0','obstacles-cs285-v0']
# ENVS= ['FishMoving-v0','FishMovingVisualServoContinous-v0']
for env_id in ENVS:
    for i in range(len(algos)):
        args = ['--exp_name',
                f'{algos[i]}_{env_id}',
                '--env_name',
                f"{env_id}",#]
                '--num_agent_train_steps_per_iter',
                '2000',
                '--batch_size_initial',
                '5000',
                '--batch_size',
                '5000',
                '--n_iter', '40',
                '--video_log_freq', '-1',
                '--sac_discount', '0.99',
                '--sac_n_layers', '2',
                '--sac_size', '256',
                '--sac_batch_size', '1500',
                '--sac_learning_rate', '0.0003',
                '--sac_init_temperature', '0.1',
                '--sac_n_iter', '5000',
                '--mbpo_rollout_length', '0']

        if algos[i]=='mbpo':
            args[-1]='10'

        args = list(map(str, args))
        command = ["python", "scripts/run_hw4_mbpo.py"] + args

        process = Process(target=run_process, args=[command])
        processes.append(process)

# Start all processes
for process in processes:
    if not process.is_alive():  # Check if process has not been started
        process.start()

# Ensure all processes have finished execution
for process in processes:
    process.join()

