# GIRIL 
ICML'20: Intrinsic Reward Driven Imitation Learning via Generative Model (Pytorch implementation).

=======

This is the code for the paper:
[Intrinsic Reward Driven Imitation Learning via Generative Model](https://arxiv.org/abs/2006.15061)  
Xingrui Yu, Yueming Lyu and Ivor W. Tsang  
Presented at [ICML 2020](https://icml.cc/Conferences/2020).  

If you find this code useful in your research then please cite  
```bash
@inproceedings{yu2020intrinsic,
  title={Intrinsic Reward Driven Imitation Learning via Generative Model},
  author={Yu, Xingrui and Lyu, Yueming and Tsang, Ivor},
  booktitle={International Conference on Machine Learning},
  pages={6672--6682},
  year={2020}
}
```  

Our implementation is based on the repo: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.

Please refer to the repo for installation, or

create experimental environment by:
```bash
conda env create -f imitation_mpi_env.yml
```

We achieve final results by following these steps:

1. train expert: 
   ```bash
   sh scripts/expert_atari.sh
   ```

2. generate one-life demonstration: 
   ```bash
   sh scripts/gen_one_life_demonstration.sh
   ```

3. train reward module: 
   ```bash
   sh scripts/train_reward_module.sh
   ```

4. policy optimizaion with the learned reward: 
   ```bash
   sh scripts/policy_optimization.sh
   ```
