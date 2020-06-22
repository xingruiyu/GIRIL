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
from curiosity import InverseModel, ForwardDynamicsModel, IntrinsicCuriosityModule, cnn_Encoder
from vae import VAE, GenerativeIntrinsicRewardModule

from running_mean_std import RunningMeanStd
from baselines.common.mpi_moments import mpi_moments

from intrinsic_utils import get_all_save_paths, save_samples_by_frame, accuracy 
from state_process import process

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # preprare save_path 
    args_dir, logs_dir, models_dir, samples_dir = get_all_save_paths(args, 'learn_reward')
    utils.cleanup_log_dir(logs_dir)

    intrinsic_model_file_name = os.path.join(models_dir, args.env_name) 
    print('intrinsic model file:', intrinsic_model_file_name)
    intrinsic_log_file_name = os.path.join(logs_dir, '0.monitor.csv') 
    print('intrinsic log file:', intrinsic_log_file_name)
    intrinsic_arg_file_name = os.path.join(args_dir, 'command.txt') 
    print('intrinsic arg file:', intrinsic_arg_file_name)

    # save args to arg_file
    with open(intrinsic_arg_file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
 
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('using device: %s' % device)

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, logs_dir, device, False)

    obs_shape = envs.observation_space.shape
    if 'NoFrameskip' in args.env_name:
        file_name = os.path.join(
            args.experts_dir, "trajs_ppo_{}.pt".format(
                args.env_name.split('-')[0].replace('NoFrameskip', '').lower()))
    else:
        file_name = os.path.join(
            args.experts_dir, "trajs_ppo_{}.pt".format(
                args.env_name.split('-')[0].lower()))

    intrinsic_train_loader = torch.utils.data.DataLoader(
        data_loader.ExpertDataset(file_name, num_trajectories=args.traj_num, \
                                train=True, train_test_split=1.0, return_next_state=True, 
                                subsample_frequency=args.subsample_frequency),
        batch_size=args.gail_batch_size,
        shuffle=True,
        drop_last=True)

    if len(obs_shape)==3:
        action_dim = envs.action_space.n
        latent_dim = 1024
    elif len(obs_shape)==1:
        action_dim = envs.action_space.shape[0]

    if args.intrinsic_module=='icm': 
        inverse_model = InverseModel(latent_dim=latent_dim, action_dim=action_dim, \
                                     hidden_dim=64).to(device)
        forward_dynamics_model = ForwardDynamicsModel(state_dim=latent_dim, action_dim=action_dim, \
                                                      hidden_dim=64).to(device)
        icm =  IntrinsicCuriosityModule(envs, device, inverse_model, forward_dynamics_model, \
                                        inverse_lr=args.intrinsic_lr, forward_lr=args.intrinsic_lr, \
                                        )
        if args.model_base == 'cnn':
            encoder = cnn_Encoder(conv_layers=32, conv_kernel_size=3, in_channels=obs_shape[0], latent_dim=latent_dim, action_dim=action_dim).to(device)
        
        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Batch_id,Forward_loss,Inverse_loss,Action_accuracy,IntrinsicReward,min_pred,mean_pred,max_pred,min_true,mean_true,max_true\n'
            f.write(head)

        print('Pretraining intrinsic module: %s for %s epochs' \
              %(args.intrinsic_module, args.intrinsic_epoch) )
        for e in range(args.intrinsic_epoch):  
            for i, expert_batch in enumerate(intrinsic_train_loader):
                state, action, next_state = expert_batch
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
                state_encoding = encoder(state)
                next_state_encoding = encoder(next_state)

                inverse_loss, forward_loss, action_logit = icm.fit_batch(state_encoding, action, \
                                                           next_state_encoding, train=True)
                action_accuracy, _ = accuracy(action_logit, action.long(), topk=(1, 2))
                if e % 10 == 0:
                    rewards, pred_next_state = icm.calculate_intrinsic_reward(state_encoding, action, next_state_encoding)
                    print('[%s]|[%s/%s] %s-%s Loss - Inverse loss: %s and Forward loss: %s, Action accuracy: %s, Rewards: %s' \
                          % (e, i, len(intrinsic_train_loader), args.intrinsic_module, args.env_name, \
                             inverse_loss.item(), forward_loss.item(), action_accuracy, torch.mean(rewards).item()))

                    result_str = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}'.format(\
                                   e,i,forward_loss.item(),\
                                   inverse_loss.item(), \
                                   action_accuracy.item(), \
                                   torch.mean(rewards).item(),\
                                   torch.min(pred_next_state).item(),\
                                   torch.mean(pred_next_state).item(),\
                                   torch.max(pred_next_state).item(),\
                                   torch.min(next_state).item(),\
                                   torch.mean(next_state).item(),\
                                   torch.max(next_state).item(),
                                   )
                    with open(intrinsic_log_file_name, 'a') as f:
                        f.write(result_str+'\n')
                    
                    #save_samples_by_frame(e, i, state_encoding, pred_next_state, next_state_encoding, \
                    #                      samples_dir, sample_size=8)

            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                                    intrinsic_model_file_name))
                torch.save([inverse_model, forward_dynamics_model, encoder], \
                            intrinsic_model_file_name+'_%s.pt'%(e+1))

        print('Saving the pretrained %s epochs %s as %s' % (e+1, args.intrinsic_module, \
                                                            intrinsic_model_file_name))
        torch.save([inverse_model, forward_dynamics_model, encoder], intrinsic_model_file_name+'.pt')
 
    if args.intrinsic_module=='vae': 
        vae = VAE(device, model_base=args.model_base, conv_layers=32, conv_kernel_size=3, \
                                latent_dim=latent_dim, action_dim=action_dim, \
                                hidden_dim=64, in_channels=obs_shape[0]*2, out_channels=obs_shape[0], \
                                ).to(device)
        gir =  GenerativeIntrinsicRewardModule(envs, device, vae, \
                                                   lr=args.intrinsic_lr, \
                                                   )

        with open(intrinsic_log_file_name, 'w') as f:
            head = 'Epoch,Batch_id,VAE_loss,Recon_loss,KLD_loss,Action_loss,Action_accuracy,IntrinsicReward,min_pred,mean_pred,max_pred,min_true,mean_true,max_true\n'
            f.write(head)

        for e in range(args.intrinsic_epoch):
            for i, expert_batch in enumerate(intrinsic_train_loader):
                state, action, next_state = expert_batch
                state = state.to(device)
                action = action.to(device)
                next_state = next_state.to(device)
        
                vae_loss, recon_loss, kld_loss, action_loss, action_logit = \
                                gir.fit_batch(state, action, next_state, \
                                train=True, kld_loss_beta=args.beta, lambda_action=args.lambda_action)
                action_accuracy, _ = accuracy(action_logit, action.long(), topk=(1, 2))
                if e % 10 == 0:
                    rewards, pred_next_state = gir.calculate_intrinsic_reward(state, action, next_state)
                    print('[%s]|[%s/%s] %s-%s Loss - VAE: %s, RECON: %s, KLD: %s, Action: %s | Accuracy - Action: %s'\
                            %(e, i, len(intrinsic_train_loader), args.intrinsic_module, args.env_name, \
                              vae_loss.item(), recon_loss.item(), kld_loss.item(), \
                              action_loss.item(), action_accuracy.item()),\
                              '| Reward:', torch.mean(rewards).item())

                    result_str = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}'.format(\
                                   e,i,vae_loss.item(),recon_loss.item(),\
                                   kld_loss.item(),\
                                   action_loss.item(), action_accuracy.item(), \
                                   torch.mean(rewards).item(),\
                                   torch.min(pred_next_state).item(),\
                                   torch.mean(pred_next_state).item(),\
                                   torch.max(pred_next_state).item(),\
                                   torch.min(next_state).item(),\
                                   torch.mean(next_state).item(),\
                                   torch.max(next_state).item(),
                                   )
                    with open(intrinsic_log_file_name, 'a') as f:
                        f.write(result_str+'\n')
                    
                    #save_samples_by_frame(e, i, state, pred_next_state, next_state, \
                    #                      samples_dir, sample_size=8)

            if (e+1) % args.intrinsic_save_interval == 0:
                print('Saving the pretrained %s epochs %s as %s' \
                      % (e+1, args.intrinsic_module, intrinsic_model_file_name))
                torch.save(vae, intrinsic_model_file_name+'_%s.pt'%(e+1))

        print('Saving the pretrained %s epochs %s as %s' \
              % (e+1, args.intrinsic_module, intrinsic_model_file_name))
        torch.save(vae, intrinsic_model_file_name+'.pt')
 
if __name__ == "__main__":
    main()
