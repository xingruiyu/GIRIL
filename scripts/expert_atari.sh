#!/bin/bash
seed=0
num_env_steps=10000000
for env_name in SpaceInvaders BeamRider Breakout Qbert Seaquest KungFuMaster 
do
  python expert_main.py --env-name ${env_name}NoFrameskip-v4 --algo ppo --use-gae --lr 2.5e-4 \
                 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 \
                 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 \
                 --log-dir expert/logs/gym/atari_expert/${num_env_steps}_steps/${env_name}/${env_name}-${seed} \
                 --seed ${seed} --save-dir expert/models/gym/atari_expert/${num_env_steps}_steps/ \
                 --num-env-steps ${num_env_steps}
done
