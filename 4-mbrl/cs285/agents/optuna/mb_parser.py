import pickle as pkl
import time
from typing import Any, Dict
import os
import sys
sys.path.append(os.path.abspath('./..'))
sys.path.append(os.path.abspath('./'))
import gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from pathlib import Path
import torch as th
from cs285.scripts.run_hw4_mbpo import MBPO_Trainer
import warnings
warnings.filterwarnings("ignore")#, category=DeprecationWarning)
from cs285.envs import register_envs
import yaml

register_envs()
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
print(f'working on {device}')


logdir='./optuna/'
Path(logdir).mkdir(parents=True, exist_ok=True)
N_TRIALS = 500
N_JOBS = 8
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 4
N_TIMESTEPS = int(3e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

def sample_mb(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    size = trial.suggest_categorical("size", [64,100,128,200,256])
    batch_size = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128, 256, 512, 1024])
    mpc_horizon = trial.suggest_uniform("mpc_horizon", 1, 40)
    ensemble_size = trial.suggest_categorical("ensemble_size", [3,5,10])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 0.05)
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(5e5)])
    batch_size_initial = trial.suggest_categorical("batch_size_initial", [int(1e3), int(0), int(1e4), int(5e3)])
    mpc_num_action_sequences = trial.suggest_categorical("mpc_num_action_sequences", [1000, 100, 500, 2000])
    # mpc_num_action_sequences = (1, "episode")
    cem_alpha = trial.suggest_categorical("cem_alpha", [0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
    cem_iterations = -1
    # cem_num_elites = trial.suggest_categorical('cem_num_elites', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    cem_num_elites = "auto"


    num_agent_train_steps_per_iter = "auto"

    return {
        "learning_rate": learning_rate,
        # "train_batch_size": train_batch_size, EP-len in robotics
        "ensemble_size": ensemble_size,
        "mpc_horizon": mpc_horizon,
        "mpc_num_action_sequences": mpc_num_action_sequences,
        "cem_iterations": cem_iterations,
        "cem_num_elites": cem_num_elites,
        "cem_alpha": cem_alpha,
        "num_agent_train_steps_per_iter": num_agent_train_steps_per_iter,
        "batch_size_initial": batch_size_initial,
        "batch_size": batch_size,
        "size": size,
    }


def sample_sac(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    sac_train_batch_size = trial.suggest_categorical("sac_train_batch_size", [16, 32, 64, 128, 256, 512, 1024])  ##steps used per gradient step
    sac_discount = trial.suggest_categorical("sac_discount", [0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    # sac_init_temperature = trial.suggest_uniform("itemp", 1e-5, 1)
    sac_learning_rate = trial.suggest_loguniform("lr", 1e-5, 0.05)
    sac_n_layers = trial.suggest_categorical("sac_n_layers", [2,3])
    sac_size = trial.suggest_categorical("sac_size", [64,128,256])
    sac_n_iter = 200
    num_agent_train_steps_per_iter = "auto"

    return {
        "sac_learning_rate": sac_learning_rate,
        "sac_n_layers": sac_n_layers,
        "sac_size": sac_size,
        "sac_discount": sac_discount,
        "sac_train_batch_size": sac_train_batch_size,
        # "sac_init_temperature"
    }


def objective(trial: optuna.Trial) -> float:

    # kwargs = {}#DEFAULT_HYPERPARAMS.copy()

    with open("./../params/mbpo.yml", "r") as f:
        kwargs = yaml.safe_load(f)
    # Sample hyperparameters
    kwargs.update(sample_sac(trial))
    kwargs.update(sample_mb(trial))

    # monitor_dir =
    envKwargs = {'frame_stack': 2, 'policy': 'CnnPolicy', 'n_envs': 1}

    rwd_method = 2
    env = gym.make('fish-moving-v0') #FishMoving()
    # env, env_name = make_cnn_render_env_fish(env, envKwargs)


    # Create env used for evaluation
    envEvaluation = gym.make('fish-moving-v0') #FishMoving()
    # envEvaluation, _ = make_cnn_render_env_fish(envEvaluation, envKwargs)
    trainer = MBPO_Trainer(kwargs)
    # and report the performance

    try:
        #model.learn(N_TIMESTEPS, callback=eval_callback)
        trainer.run_training_loop()
    except:
        # Sometimes, random hyperparams can generate NaN
        nan_encountered = True
    finally:
        # Free memory
        env.close()
        envEvaluation.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if trainer.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return trainer.rl_trainer.eval_return #eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        pass
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv(logdir+f"study_results_discrete_cartpole{time.time()}.csv")

    with open(logdir+f"study{time.time()}_SAC.pkl", "wb+") as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()