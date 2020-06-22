#!/bin/bash

traj_num=1
subsample=4 

entropy_coef=0.01
ppo_epoch=4
load_iter=100 #final
debug="False"
model_base='cnn'

intrinsic_epoch=50000  
intrinsic_lr=3e-5 
batch_size=32

lambda_action=100.0  
lambda_recon=1.0
lambda_kl=1.0  

combine_action='True'
lambda_true_action=1.0
standardize='True'
intrinsic_module='vae'
num_env_steps=50000000

for seed in 1 2 3 4 5
  do
  for env_name in SpaceInvaders BeamRider Breakout Qbert KungFuMaster 
  do
    python intrinsic_main.py --env-name ${env_name}NoFrameskip-v4 --algo ppo \
                             --traj-num ${traj_num} --subsample-frequency ${subsample} \
                             --use-gae --log-interval 1 --num-steps 128 --num-processes 8 \
                             --lr 2.5e-4 --entropy-coef ${entropy_coef} --value-loss-coef 0.5 \
                             --ppo-epoch ${ppo_epoch} --num-mini-batch 4 --gamma 0.99 --gae-lambda 0.95 \
                             --num-env-steps ${num_env_steps} --use-linear-lr-decay --clip-param 0.1 \
                             --use-intrinsic --intrinsic-lr ${intrinsic_lr} --debug ${debug} \
                             --lambda_action ${lambda_action} --beta ${lambda_kl} --lambda_recon ${lambda_recon} \
                             --intrinsic-module ${intrinsic_module} --load-iter ${load_iter} \
                             --policy-save-dir policy --reward-save-dir reward \
                             --seed ${seed} --intrinsic-epoch ${intrinsic_epoch} \
                             --standardize ${standardize} --model-base ${model_base} --gail-batch-size ${batch_size} \
                             --combine-action ${combine_action} --lambda_true_action ${lambda_true_action} 
  done
done

