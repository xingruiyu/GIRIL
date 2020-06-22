# GIRIL 
ICML'20: Intrinsic Reward Driven Imitation Learning via Generative Model (Pytorch implementation).

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

4. policy optimizaion with the leaned reward: 
   ```bash
   sh scripts/policy_optimization.sh
   ```
