python run_parallel.py \
    --group_name mujoco_video \
    --exp_name mujoco_video \
    --config_name online_rl \
    --agent_config hyper_simba \
    --env_type mujoco \
    --device_ids 0 1 2 \
    --num_seeds 5 \
    --num_exp_per_device 7 \
    --overrides project_name='Simba_2502' \