import copy
import glob
import os
import csv
import time
import json
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.intrinsic_storage import RolloutStorage
from evaluation import evaluate

from a2c_ppo_acktr.envs import VecNormalize
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian

import data_loader
from curiosity import InverseModel, ForwardDynamicsModel, IntrinsicCuriosityModule 
from vae import VAE, GenerativeIntrinsicRewardModule

from running_mean_std import RunningMeanStd
from baselines.common.mpi_moments import mpi_moments

from intrinsic_utils import get_all_save_paths, save_samples_by_frame
from state_process import process

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    args_dir, logs_dir, models_dir, samples_dir = get_all_save_paths(args, 'pretrain', combine_action=args.combine_action)
    eval_log_dir = logs_dir + "_eval"
    utils.cleanup_log_dir(logs_dir)
    utils.cleanup_log_dir(eval_log_dir)

    _, _, intrinsic_models_dir, _ = get_all_save_paths(args, 'learn_reward', load_only=True)
    if args.load_iter != 'final':
        intrinsic_model_file_name = os.path.join(intrinsic_models_dir, args.env_name + '_{}.pt'.format(args.load_iter)) 
    else:
        intrinsic_model_file_name = os.path.join(intrinsic_models_dir, args.env_name + '.pt'.format(args.load_iter)) 
    intrinsic_arg_file_name = os.path.join(args_dir, 'command.txt') 

    # save args to arg_file
    with open(intrinsic_arg_file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, logs_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
    else:
        raise NotImplementedError

    if args.use_intrinsic:
        obs_shape = envs.observation_space.shape
        if len(obs_shape)==3:
            action_dim = envs.action_space.n
        elif len(obs_shape)==1:
            action_dim = envs.action_space.shape[0]

        if 'NoFrameskip' in args.env_name:
            file_name = os.path.join(
                args.experts_dir, "trajs_ppo_{}.pt".format(
                    args.env_name.split('-')[0].replace('NoFrameskip', '').lower()))
        else:
            file_name = os.path.join(
                args.experts_dir, "trajs_ppo_{}.pt".format(
                    args.env_name.split('-')[0].lower()))

        rff = RewardForwardFilter(args.gamma)
        intrinsic_rms = RunningMeanStd(shape=())
        
        if args.intrinsic_module=='icm': 
            print('Loading pretrained intrinsic module: %s' % intrinsic_model_file_name)
            inverse_model, forward_dynamics_model, encoder = torch.load(intrinsic_model_file_name)
            icm =  IntrinsicCuriosityModule(envs, device, inverse_model, forward_dynamics_model, \
                                            inverse_lr=args.intrinsic_lr, forward_lr=args.intrinsic_lr,\
                                            )

        if args.intrinsic_module=='vae': 
            print('Loading pretrained intrinsic module: %s' % intrinsic_model_file_name)
            vae = torch.load(intrinsic_model_file_name)
            icm =  GenerativeIntrinsicRewardModule(envs, device, \
                                                   vae, lr=args.intrinsic_lr, \
                                                   )


    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)
            next_obs = obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, next_obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.use_intrinsic: 
            for step in range(args.num_steps):
                state = rollouts.obs[step]
                action = rollouts.actions[step]
                next_state = rollouts.next_obs[step]
                if args.intrinsic_module == 'icm':
                    state = encoder(state)
                    next_state = encoder(next_state)
                with torch.no_grad():
                    rollouts.rewards[step], pred_next_state = icm.calculate_intrinsic_reward(
                                                                                state, action, next_state, args.lambda_true_action)
            if args.standardize == 'True':
                buf_rews = rollouts.rewards.cpu().numpy()
                intrinsic_rffs = np.array([rff.update(rew) for rew in buf_rews.T])
                rffs_mean, rffs_std, rffs_count = mpi_moments(intrinsic_rffs.ravel())
                intrinsic_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
                mean = intrinsic_rms.mean
                std = np.asarray(np.sqrt(intrinsic_rms.var))
                rollouts.rewards = rollouts.rewards / torch.from_numpy(std).to(device)

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(models_dir, args.algo)
            policy_file_name = os.path.join(save_path, args.env_name + '.pt') 

            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], policy_file_name)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("{} Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(args.env_name, j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

if __name__ == "__main__":
    main()
