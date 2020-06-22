import os
import torch
from torchvision import utils
import torch.nn.functional as F
from a2c_ppo_acktr.arguments import get_args
from collections import defaultdict, namedtuple
import numpy as np
from mpi4py import MPI

def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n+1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = comm.allreduce(localsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]

def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_save_path(args, what_to_return='logs', how_to_learn='pretrain', load_only=False, seed=None, intrinsic_module='vae', combine_action=False, beta=None):
    """
    what_to_return: args, logs, models, samples
    how_to_learn: learn policy by pretrain, learn reward by learn_reward
    """
    base = 'full_{0}_{1}_demo/'.format(args.traj_num, args.demo_type)

    if args.subsample_frequency != 4:
        base = base + 'subsample_%s_' % args.subsample_frequency

    if beta is not None:
        lambda_true_action = beta
    else:
        lambda_true_action = args.lambda_true_action 

    intrinsic_hypers = 'intrinsic_{}_modelBase_{}_lr_{}_bs_{}_epochs_{}_lamRecon_{}_beta_{}_lamAction_{}'.format(intrinsic_module, args.model_base, args.intrinsic_lr, args.gail_batch_size, args.intrinsic_epoch, args.lambda_recon, args.beta, args.lambda_action)
    if combine_action == 'True':
        intrinsic_hypers += '_combined_action_true_action_lambda_{}'.format(lambda_true_action)

    policy_hypers = 'policy_{}_lr_{}_numSteps_{:.0f}_envSteps_{:.0f}_valueCoef_{}_entropyCoef_{}'.format(args.algo, args.lr, args.num_steps, args.num_env_steps, args.value_loss_coef, args.entropy_coef)
    if args.algo == 'ppo':
        policy_hypers += '_ppoEpoch_{}_miniBatch_{}'.format(args.ppo_epoch, args.num_mini_batch)

    if how_to_learn == 'learn_reward':
        learn_reward_base = base + 'learn_reward/{}'.format(intrinsic_hypers)
        common = 'gym_atari/{0}/{0}-{1}'.format(args.env_name.replace('NoFrameskip-v4', ''), 0)
        sub_dir = os.path.join(what_to_return, learn_reward_base, common)
    else:
        if seed is not None:
            common = 'gym_atari/{0}/{0}-{1}'.format(args.env_name.replace('NoFrameskip-v4', ''), seed)
        else:
            common = 'gym_atari/{0}/{0}-{1}'.format(args.env_name.replace('NoFrameskip-v4', ''), args.seed)
        if args.gail:
            base += 'learn_policy/gail_{}_discrBase_{}/{}/'.format(args.adv_reward_type, args.model_base, policy_hypers)
        if args.vail:
            base += 'learn_policy/vail_{}_ic_{}_discrBase_{}/{}/'.format(args.adv_reward_type, args.i_c, args.model_base, policy_hypers)
        if args.use_intrinsic:
            if how_to_learn == 'pretrain':
                base += 'learn_policy/{0}_{1}_loadIter_{2}/{3}/'.format(how_to_learn, intrinsic_hypers, args.load_iter, policy_hypers)
            elif how_to_learn == 'bc':  
                base += 'learn_policy/{0}_modelBase{1}_lr_{2}_bs_{3}_{4}_epochs/'.format(how_to_learn, args.model_base, args.intrinsic_lr, args.gail_batch_size, args.intrinsic_epoch)
            elif how_to_learn == 'gail':  
                base += 'learn_policy/gail_{}_discrBase_{}/{}/'.format(args.adv_reward_type, args.model_base, policy_hypers)
            elif how_to_learn == 'vail':  
                base += 'learn_policy/vail_{}_ic_{}_discrBase_{}/{}/'.format(args.adv_reward_type, args.i_c, args.model_base, policy_hypers)

            if how_to_learn != 'gail' and how_to_learn != 'vail':
                if args.standardize == 'True':
                    if how_to_learn == 'bc':  
                        base = os.path.join(base, 'original_rewards/')
                    else:
                        base = os.path.join(base, 'standardized_rewards/')
                else:
                    base = os.path.join(base, 'original_rewards/')
        
        sub_dir = os.path.join(what_to_return, base, common)

        if args.debug == 'True':
            sub_dir = os.path.join('debug', sub_dir)

    if how_to_learn == 'learn_reward':
        save_path = os.path.join(args.reward_save_dir, sub_dir)
    else:
        save_path = os.path.join(args.policy_save_dir, sub_dir)
        
    if not load_only:
        try:
            os.makedirs(save_path)
        except OSError:
            pass

    return save_path

def get_all_save_paths(args, how_to_learn='learn_reward', load_only=False, combine_action='False'):
    args_dir = get_save_path(args, 'args', how_to_learn, load_only, combine_action=combine_action)
    logs_dir = get_save_path(args, 'logs', how_to_learn, load_only, combine_action=combine_action)
    models_dir = get_save_path(args, 'models', how_to_learn, load_only, combine_action=combine_action)
    samples_dir = get_save_path(args, 'samples', how_to_learn, load_only, combine_action=combine_action)
    
    return args_dir, logs_dir, models_dir, samples_dir

def save_samples(e, i, state, pred_next_state, next_state, sample_dir, sample_size=8, num_frams=1):
    sample_state = state[:sample_size, :num_frams, :, :]
    sample_next_state = next_state[:sample_size, :num_frams, :, :]
    sample_pred_next_state = pred_next_state[:sample_size, :num_frams, :, :]
    utils.save_image(
        torch.cat([sample_state, sample_pred_next_state, sample_next_state], 0),
        '{}/{}_{}.png'.format(sample_dir, str(e + 1).zfill(5), str(i).zfill(5)),
        nrow=sample_size,
        normalize=True,
        range=(-1, 1),
    )

def save_samples_by_frame(e, batch_idx, state, pred_next_state, next_state, sample_dir, sample_size=8):
    if pred_next_state.dim() == 4:
        for i in range(state.shape[1]): 
            frame_sample_dir = os.path.join(sample_dir, 'frame_{}'.format(i))
            try:
                os.makedirs(frame_sample_dir)
            except OSError:
                pass

            sample_state = state[:sample_size, i, :, :].unsqueeze(1)
            sample_next_state = next_state[:sample_size, i, :, :].unsqueeze(1)
            sample_pred_next_state = pred_next_state[:sample_size, i, :, :].unsqueeze(1)
            utils.save_image(
                torch.cat([sample_state, sample_pred_next_state, sample_next_state], 0),
                '{}/{}_{}.png'.format(frame_sample_dir, str(e).zfill(5), str(i).zfill(5)),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
    else:
        sample_state = state[:sample_size]
        sample_next_state = next_state[:sample_size]
        sample_pred_next_state = pred_next_state[:sample_size]
        utils.save_image(
            torch.cat([sample_state, sample_pred_next_state, sample_next_state], 0),
            '{}/{}_{}.png'.format(sample_dir, str(e).zfill(5), str(batch_idx).zfill(5)),
            nrow=sample_size,
            normalize=True,
            range=(-1, 1),
        )


