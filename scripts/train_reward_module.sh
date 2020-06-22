#!/bin/bash

traj_num=1
experts_dir=icml2020_trajs/${traj_num}episodes/original/
num_steps=128
subsample=4 
intrinsic_module='vae'  
intrinsic_epoch=50000
intrinsic_lr=3e-5
batch_size=32
debug="False"
standardize="False"
model_base='cnn'

lambda_action=100.0 
lambda_recon=1.0
lambda_kl=1.0

seed=0

for env_name in SpaceInvaders BeamRider Breakout Qbert Seaquest KungFuMaster 
do
  python learn_reward.py --env-name ${env_name}NoFrameskip-v4 --algo ppo \
                           --experts-dir ${experts_dir} --intrinsic-lr ${intrinsic_lr} \
                           --seed ${seed} --intrinsic-epoch ${intrinsic_epoch} \
                           --intrinsic-module ${intrinsic_module} --beta ${lambda_kl} \
                           --lambda_action ${lambda_action} --lambda_recon ${lambda_recon} \
                           --gail-batch-size ${batch_size} --traj-num ${traj_num} \
                           --subsample-frequency ${subsample} --reward-save-dir reward \
                           --standardize ${standardize} --model-base ${model_base} \
                           --intrinsic-save-interval 5000 --debug ${debug}
done
