import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--vail',
        action='store_true',
        default=False,
        help='do imitation learning with vail')
    parser.add_argument(
        '--i-c',
        type=float,
        default=0.2,
        help='value of information bottleneck (default: 0.2)')
    parser.add_argument(
        '--adv-reward-type',
        type=str,
        default='logd',
        help='type of reward used in gail and vail, logd for -log(D(s,a)), log1-d for -log(1-D(s,a))')
    parser.add_argument(
        '--experts-dir',
        default='./tmp_experts',
        help='directory that contains expert demonstrations')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--traj-num',
        type=int,
        default=1,
        help='gail trajectory number (default: 1)')
    parser.add_argument(
        '--subsample-frequency',
        type=int,
        default=4,
        help='subsample frequency for loading expert data (default: 4)')
    parser.add_argument(
        '--gail-epoch', type=int, default=1, help='gail epochs (default: 1)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=10000,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./discrete_vae_icm_exps/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--reward-save-dir',
        default='./reward_save_dir/',
        help='directory to save agent logs (default: ./reward_save_dir/)')
    parser.add_argument(
        '--policy-save-dir',
        default='./policy_save_dir/',
        help='directory to save agent logs (default: ./policy_save_dir/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    # hyper-parameters for intrinsic reward
    parser.add_argument(
        '--use-intrinsic',
        action='store_true',
        default=False,
        help='use intrinsic reward to replace the true reward for imitation.')
    parser.add_argument(
        '--intrinsic-lr', type=float, default=3e-5, help='learning rate for training intrinsic module (default: 3e-5)')
    parser.add_argument(
        '--intrinsic-epoch', type=int, default=50000, help='intrinsic epochs (default: 50000)')
    parser.add_argument(
        '--intrinsic-module',
        default='vae',
        help='name of intrinsice module, icm, vae, (default: vae).')
    parser.add_argument(
        '--intrinsic-save-interval',
        type=int,
        default=100,
        help='intrinsic save interval, one save intrinsic module per n epochss (default: 100)')
    parser.add_argument(
        '--debug',
        choices=['True', 'False'],
        default='False',
        help='debug or not')
    parser.add_argument(
        '--standardize',
        choices=['True', 'False'],
        default='True',
        help='standardize intrinsic rewards')
    parser.add_argument(
        '--load-iter', default='final', help='load iter of pretrained reward (default: final)')
    parser.add_argument(
        '--lambda_action',
        type=float,
        default=100.0,
        help='weight for action loss/true action when calculating reward (default: 100.0)')
    parser.add_argument(
        '--lambda_true_action',
        type=float,
        default=1.0,
        help='weight for true action when calculating reward (beta in our paper, default: 1.0)')
    parser.add_argument(
        '--combine-action',
        choices=['True', 'False'],
        default='False',
        help='combine true action and pred action or not when calculate reward ')
    parser.add_argument(
        '--lambda_recon',
        type=float,
        default=1.0,
        help='weight for reconstruction loss (default: 1.0)')
    parser.add_argument(
        '--beta',
        type=float,
        default=1.0,
        help='weight for kld loss in vae (default: 1.0)')
    parser.add_argument(
        '--demo-type',
        default='original',
        help='demo type, choices: original (default: original)')
    parser.add_argument(
        '--model-base',
        default='cnn',
        help='model base (default: cnn)')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    return args
