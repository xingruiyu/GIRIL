import copy
import glob
import os
import time
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
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from a2c_ppo_acktr.envs import VecNormalize
#from a2c_ppo_acktr.vec_check_nan import VecCheckNan
#from running_mean_std import RunningMeanStd


from a2c_ppo_acktr.distributions import Bernoulli, Categorical#, DiagGaussian
from a2c_ppo_acktr.utils import AddBias, init
FixedNormal = torch.distributions.Normal

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        #return FixedNormal(action_mean, action_logstd.exp())
        return FixedNormal(action_mean, action_mean.exp())


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, env_name, device):
        super(RandomAgent, self).__init__()
        self.action_space = action_space
        self.env_name = env_name
        self.device = device

        if 'NoFrameskip' in self.env_name:
            self.feature_size = 1024
        else:
            self.feature_size = 64

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = self.action_space.n
            self.dist = Categorical(self.feature_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = self.action_space.shape[0]
            self.dist = DiagGaussian(self.feature_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = self.action_space.shape[0]
            self.dist = Bernoulli(self.feature_size, num_outputs)
        else:
            raise NotImplementedError
   
        self.actor_features = torch.rand(self.feature_size)

    def act(self, deterministic=False):
        dist = self.dist(self.actor_features)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if 'NoFrameskip' in self.env_name:
            action = action.unsqueeze(0)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()
        return action, action_log_probs, dist_entropy

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

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
    elif args.algo == 'random':
        agent = RandomAgent(envs.action_space, args.env_name, device)
    else:
        raise NotImplementedError

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

        if args.use_linear_lr_decay and args.algo != 'random':
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                if args.algo == 'random':
                    value, recurrent_hidden_states = torch.rand([1]).to(device), torch.rand([1]).to(device)
                    action, action_log_prob, dist_entropy = agent.act()

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            #if np.isnan(obs.cpu().numpy()).any():
            #    obs = np.nan_to_num(obs.cpu().numpy())
            #    obs = torch.from_numpy(obs).float().to(device)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        if args.algo != 'random':
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
        else:
            value_loss, action_loss = 0.0, 0.0

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_{}_.pt".format(j)))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "{} Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(args.env_name, j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            #if len(envs.observation_space.shape) == 1:
            #    ob_rms = utils.get_vec_normalize(envs).ob_rms
            #    print('ob_rms', ob_rms)
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
