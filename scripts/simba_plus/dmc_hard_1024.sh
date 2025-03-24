python run_parallel.py \
    --group_name simba_scaling \
    --exp_name simba_critic_width_2048 \
    --agent_config simba \
    --env_type dmc_hard \
    --device_ids 0 1 2 3 \
    --num_seeds 5 \
    --num_exp_per_device 4 \
    --server kaist \
    --overrides project_name='Simba_2501' \
    --overrides updates_per_interaction_step=2 \
    --overrides agent.critic_hidden_dim=2048 \

