import argparse
import copy
import multiprocessing as mp
import os
import pprint
import time

import numpy as np


def run_with_device(server, device_id, config_path, config_name, overrides):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    if server == "kaist":
        cuda_id_to_egl_id = {
            0: 3,
            1: 2,
            2: 1,
            3: 0,
            4: 7,
            5: 6,
            6: 5,
            7: 4,
        }
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cuda_id_to_egl_id[int(device_id)])
    else:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(0)

    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(0)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["WANDB_START_METHOD"] = "thread"

    # Now import the main script
    from run_online import run

    args = {
        "config_path": config_path,
        "config_name": config_name,
        "overrides": overrides,
    }
    run(args)


if __name__ == "__main__":
    ###################################################################################
    # NOTE:
    # current implementation only support parallelism when num_seeds=num_exp_per_device
    ####################################################################################

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="online_rl")
    parser.add_argument("--agent_config", type=str, default="hyper_simba_dev")
    parser.add_argument("--env_type", type=str, default="acrobot_continual")
    parser.add_argument("--device_ids", default=[0], nargs="+")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--num_exp_per_device", type=int, default=1)
    parser.add_argument("--server", type=str, default="local")
    parser.add_argument("--group_name", type=str, default="test")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--num_sequence", type=int, default=2)
    parser.add_argument("--overrides", action="append", default=[])

    args = vars(parser.parse_args())
    seeds = (np.arange(args.pop("num_seeds")) * 1000).tolist()
    device_ids = args.pop("device_ids")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

    num_devices = len(device_ids)
    num_exp_per_device = args.pop("num_exp_per_device")
    pool_size = num_devices * num_exp_per_device

    # create configurations for child run
    experiments = []
    config_path = args.pop("config_path")
    config_name = args.pop("config_name")
    server = args.pop("server")
    group_name = args.pop("group_name")
    exp_name = args.pop("exp_name")
    agent_config = args.pop("agent_config")
    num_sequence = args.pop("num_sequence")

    # import library after CUDA_VISIBLE_DEVICES operation
    from scale_rl.envs.humanoid_bench import HB_LOCOMOTION_NOHAND_SEQUENTIAL

    env_type = args.pop("env_type")
    if env_type == "hb_sequential":
        envs = HB_LOCOMOTION_NOHAND_SEQUENTIAL
        env_configs = ["hb_locomotion"] * len(envs)

    # for sanity check
    elif env_type == "acrobot_continual":
        envs = ["acrobot-swingup"]
        env_configs = ["dmc"]

    else:
        raise NotImplementedError

    for sequence in range(num_sequence):
        cur_exp_name = exp_name + "_" + str(sequence)
        for env_idx, env_name in enumerate(envs):  # Then loop over environments
            for seed_idx, seed in enumerate(seeds):
                exp = copy.deepcopy(args)  # copy overriding arguments
                exp["config_path"] = config_path
                exp["config_name"] = config_name

                exp["overrides"].append("agent=" + agent_config)
                exp["overrides"].append("env=" + env_configs[env_idx])
                exp["overrides"].append("env.env_name=" + env_name)

                exp["overrides"].append("server=" + server)
                exp["overrides"].append("group_name=" + group_name)
                exp["overrides"].append("exp_name=" + cur_exp_name)
                exp["overrides"].append("seed=" + str(seed))

                # Sequentially load model and continue training
                if not ((sequence == 0) and (env_idx == 0)):
                    prev_exp_name = (
                        exp_name + "_" + str(sequence if env_idx > 0 else sequence - 1)
                    )
                    prev_env_name = envs[env_idx - 1] if env_idx > 0 else envs[-1]
                    load_path = (
                        "models"
                        + "/"
                        + group_name
                        + "/"
                        + prev_exp_name
                        + "/"
                        + prev_env_name
                        + "/"
                        + str(seed)
                    )
                    exp["overrides"].append("load_path=" + load_path)

                # Append the experiment to the list
                experiments.append(exp)

    pprint.pp(experiments)

    # run parallel experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn")
    available_gpus = device_ids
    process_dict = {gpu_id: [] for gpu_id in device_ids}

    # Dictionary to track the last running process for each seed
    # This ensures that sub-experiments for the same seed run sequentially (a->b->c).
    seed_process_map = {}

    for exp in experiments:
        # 1) Extract the seed for this experiment
        seed_value = None
        for override_item in exp["overrides"]:
            if override_item.startswith("seed="):
                seed_value = override_item.split("=")[1]
                break

        # 2) If there is already a process running for this seed,
        #    we must join() it to ensure it finishes before launching the next sub-experiment.
        if seed_value in seed_process_map:
            print(
                f"[Seed {seed_value}] Waiting for previous sub-experiment to finish..."
            )
            seed_process_map[seed_value].join()
            del seed_process_map[seed_value]

        wait = True
        # wait until there exists a finished process
        while wait:
            # Find all finished processes and register available GPU
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)

            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False
                    gpu_id, processes = min(
                        process_dict.items(), key=lambda x: len(x[1])
                    )
                    break

            time.sleep(10)

        # get running processes in the gpu
        processes = process_dict[gpu_id]
        exp["device_id"] = str(gpu_id)
        process = mp.Process(
            target=run_with_device,
            args=(
                server,
                exp["device_id"],
                exp["config_path"],
                exp["config_name"],
                exp["overrides"],
            ),
        )
        process.start()

        # save this process in seed_process_map for sequential gating
        processes.append(process)
        seed_process_map[seed_value] = process
        print(f"Process {process.pid} on GPU {gpu_id} started.")

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
