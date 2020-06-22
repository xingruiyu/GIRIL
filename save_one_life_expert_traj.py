import argparse
import os
import sys
import pickle

import numpy as np
import torch

from traj_envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize


sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/expert/ppo/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--load-iter',
    type=int,
    default=9000,
    help='load iter, (default: 9000)')
parser.add_argument(
    '--preprocess-type',
    type=str,
    default='none',
    help='how to preprocess states: none, norm, norm_and_noscore')
parser.add_argument(
    '--num-episode',
    type=int,
    default=10,
    help='number of episodes, (default: 53)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
parser.add_argument(
    '--render',
    action='store_true',
    default=False,
    help='whether to render the environment')
parser.add_argument(
    '--save-normal-traj',
    action='store_true',
    default=False,
    help='whether to save complete trajectories')
parser.add_argument(
    '--save-incomplete-traj',
    action='store_true',
    default=False,
    help='whether to save incomplete trajectories')
parser.add_argument(
    '--mask-out-type',
    type=str,
    default='fill_0',
    help='generate incomlete traj by remove or fill_0')
parser.add_argument(
    '--incomplete-ratio',
    type=float,
    default=0.1,
    help='incomplete ratio: ratio of state-actions to be removed from trajectories, (default: 0.1)')
args = parser.parse_args()

args.det = not args.non_det


def preprocess(obs, preprocess_type, simple_env_name):
    if preprocess_type == 'none':
        obs_processed = obs

def traj_1_generator(actor_critic, ob_rms, simple_env_name):
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    
    env = make_vec_envs(
        args.env_name,
        args.seed + 1,
        1,
        None,
        None,
        device=device,
        allow_early_resets=False)
    
    # Get a render function
    render_func = get_render_func(env)
    
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms
    
    if args.render:
        if render_func is not None:
            render_func('human')
    
    if args.env_name.find('Bullet') > -1:
        import pybullet as p
    
        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i
 

    masks = torch.zeros(1, 1)
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    
    done = False
    eps_states = []
    eps_actions = []
    eps_rewards = []

    steps = 0
    reward = 0
    eps_return = 0
    eps_length = 0

    obs = env.reset()

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, info = env.step(action)

        eps_states.append(preprocess(obs.cpu().numpy(), args.preprocess_type, simple_env_name)[0])
        eps_actions.append(action[0][0].cpu().numpy())
        eps_rewards.append(reward[0][0].cpu().numpy())

        steps += 1
        eps_length += 1
        eps_return += reward[0][0].cpu().numpy()

        masks.fill_(0.0 if done else 1.0)

        if args.render:
            if render_func is not None:
                render_func('human')

        if steps % 1000 ==0:
            print('steps', steps)
        if done:
            print('info: ', info)
            break

    env.close()

    eps_states = np.array(eps_states)
    eps_actions = np.array(eps_actions)
    eps_rewards = np.array(eps_rewards)
   
    print('eps_return', eps_return)
    print('eps_length', eps_length)
   
    return eps_states, eps_actions, eps_rewards, eps_return, eps_length

def save_normal_traj():
    atari_str = 'NoFrameskip'
    if atari_str in args.env_name:
        simple_env_name = args.env_name.split('-')[0].replace(atari_str, '').lower()
    else:
        simple_env_name = args.env_name.split('-')[0].lower()
    
    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = \
                torch.load(os.path.join(args.load_dir, args.env_name + "_{}_.pt".format(args.load_iter)))
    
    lengths = []
    rewards = []
    returns = []
    states = []
    actions = []

    for i in range(args.num_episode):
        print(args.env_name, 'Episode', i, '==========================')
        eps_states, eps_actions, eps_rewards, eps_return, eps_length = \
            traj_1_generator(actor_critic, ob_rms, simple_env_name)
        
        states.append(eps_states)
        actions.append(eps_actions)
        rewards.append(eps_rewards)
        returns.append(eps_return)
        lengths.append(eps_length)
   
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    returns = np.array(returns)
    lengths = np.array(lengths)

    print('states', states.shape)
    print('actions', actions.shape)
    print('rewards', rewards.shape, sum(rewards), rewards)
    print('lengths', lengths)
    traj = {}
    traj['states'] = states
    traj['actions'] = actions
    traj['rewards'] = rewards
    traj['returns'] = returns
    traj['lengths'] = lengths
 
    if args.preprocess_type == 'none':
        save_path = 'icml2020_trajs/{}episodes/original'.format(args.num_episode)

    try:
        os.makedirs(save_path)
    except OSError:
       pass
    
    file_name = '{}/trajs_ppo_{}.pt'.format(save_path, simple_env_name)
    pickle.dump(traj, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if args.save_normal_traj:
    print('saving complete expert %s trajs' % (args.env_name))
    save_normal_traj()

