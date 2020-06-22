# gen traj for Atari
for env_name in SpaceInvaders #BeamRider Breakout Qbert Seaquest KungFuMaster 
do
  python save_one_life_expert_traj.py --env-name ${env_name}NoFrameskip-v4 \
                                      --load-dir expert/models/gym/atari_expert/10000000_steps/ppo/ \
                                      --load-iter 9764 --num-episode 1 --save-normal-traj \
                                      --preprocess-type none #--render
done

