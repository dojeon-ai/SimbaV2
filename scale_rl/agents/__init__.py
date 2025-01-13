import gymnasium as gym
from typing import TypeVar
from omegaconf import OmegaConf
from scale_rl.agents.base_agent import BaseAgent
from scale_rl.agents.random_agent import RandomAgent
from scale_rl.agents.simba.simba_agent import SimbaAgent
from scale_rl.agents.hyper_simba.hyper_simba_agent import HyperSimbaAgent
from scale_rl.agents.hyper_simba_dev.hyper_simba_dev_agent import HyperSimbaDevAgent
from scale_rl.agents.wrappers import ObservationNormalizer, RewardNormalizer

Config = TypeVar('Config')


def create_agent(
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    cfg: Config,
) -> BaseAgent:

    cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
    agent_type = cfg.pop('agent_type')

    if agent_type == 'random':
        agent = RandomAgent(observation_space, action_space, cfg)

    elif agent_type == 'simba':
        agent = SimbaAgent(observation_space, action_space, cfg)

    elif agent_type == 'hyper_simba':
        agent = HyperSimbaAgent(observation_space, action_space, cfg)

    elif agent_type == 'hyper_simba_dev':
        agent = HyperSimbaDevAgent(observation_space, action_space, cfg)

    else:
        raise NotImplementedError

    # observation and reward normalization wrappers
    if cfg['normalize_observation']:
        agent = ObservationNormalizer(
            agent, 
            load_rms=cfg['load_observation_normalizer']
        )
    if cfg['normalize_reward']:
        agent = RewardNormalizer(
            agent, 
            gamma=cfg['gamma'], 
            g_max=cfg['normalized_g_max'],
            load_rms=cfg['load_reward_normalizer']
        )

    return agent