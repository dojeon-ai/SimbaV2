import argparse
import copy
import math
import random

import hydra
import numpy as np
import omegaconf
import tqdm
from dotmap import DotMap

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_dataset, create_envs, get_normalized_score
from scale_rl.evaluation import evaluate, record_video


def run(args):
    ###############################
    # configs
    ###############################

    args = DotMap(args)
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    omegaconf.OmegaConf.resolve(cfg)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    #############################
    # envs
    #############################
    train_env, eval_env = create_envs(**cfg.env)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    dataset = create_dataset(cfg.env.env_type, cfg.env.env_name)

    #############################
    # buffer
    #############################
    cfg.buffer.max_length = len(dataset)
    buffer = create_buffer(
        observation_space=observation_space, action_space=action_space, **cfg.buffer
    )
    buffer.reset()

    #############################
    # fill buffer
    #############################

    for i, timestep in tqdm.tqdm(
        list(enumerate(dataset)), desc="Filling buffer with dataset"
    ):
        buffer.add(timestep)

    #############################
    # agent
    #############################

    batch_size = cfg.buffer.sample_batch_size
    cfg.num_interaction_steps = int((len(dataset) / batch_size) * cfg.num_epochs)
    cfg.agent.learning_rate_decay_step = int(
        cfg.agent.learning_rate_decay_rate
        * cfg.num_interaction_steps
        * cfg.updates_per_interaction_step
    )

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    # iterate over buffer to update normalizers
    num_batches = int(np.floor(len(dataset) / batch_size))
    for batch_num in tqdm.tqdm(range(num_batches), desc="updating normalizers"):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        batch_indices = np.arange(start_idx, end_idx)
        batch = buffer.sample(sample_idxs=batch_indices)

        # update normalizers
        agent.sample_actions(i, copy.deepcopy(batch), training=True)

    #############################
    # train offline
    #############################

    logger = WandbTrainerLogger(cfg)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    eval_info["avg_normalized_return"] = get_normalized_score(
        cfg.env.env_type, cfg.env.env_name, eval_info["avg_return"]
    )
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    update_step = 0
    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1
    ):
        # update network
        batch = buffer.sample()
        update_info = agent.update(update_step, batch)
        logger.update_metric(**update_info)
        update_step += 1

        # evaluation
        if interaction_step % cfg.evaluation_per_interaction_step == 0:
            eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
            eval_info["avg_normalized_return"] = get_normalized_score(
                cfg.env.env_type, cfg.env.env_name, eval_info["avg_return"]
            )
            logger.update_metric(**eval_info)

        # metrics
        if interaction_step % cfg.metrics_per_interaction_step == 0:
            batch = buffer.sample()
            metrics_info = agent.get_metrics(batch, update_info)
            if metrics_info:
                logger.update_metric(**metrics_info)

        # TODO Support video recording
        # # video recording
        # if offline_step % cfg.offline.recording_per_offline_step == 0:
        #     video_info = record_video(agent, eval_env, cfg.num_record_episodes)
        #     logger.update_metric(**video_info)

        # logging
        if interaction_step % cfg.logging_per_interaction_step == 0:
            logger.log_metric(step=interaction_step)
            logger.reset()

    # final evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    eval_info["avg_normalized_return"] = get_normalized_score(
        cfg.env.env_type, cfg.env.env_name, eval_info["avg_return"]
    )
    logger.update_metric(**eval_info)
    logger.log_metric(step=interaction_step)
    logger.reset()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="offline_rl")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()

    run(vars(args))
