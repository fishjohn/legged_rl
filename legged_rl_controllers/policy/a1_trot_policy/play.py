# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from blind_locomotion_gym import BLIND_LOCOMOTION_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from blind_locomotion_gym.envs import *
from blind_locomotion_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 36)
    env_cfg.env.num_common_group = min(env_cfg.env.num_common_group, 18)
    env_cfg.env.num_adaptive_group = env_cfg.env.num_envs - env_cfg.env.num_common_group
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows-1
    env_cfg.terrain.terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 2
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history, commands, height_samples = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)
    terrain_raw_encoder = ppo_runner.get_inference_terrain_raw_encoder(
        device=env.device
    )
    terrain_propri_encoder = ppo_runner.get_inference_terrain_propri_encoder(
        device=env.device
    )

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            BLIND_LOCOMOTION_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.terrain_encoder_module.propri_encoder,
            path,
            "terrain_encoder",
            ppo_runner.alg.terrain_encoder_module.propri_encoder_input_dim,
        )

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 500  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        est = encoder(obs_history)
        terrain_latent = torch.cat(
            (
                terrain_raw_encoder(height_samples[: env_cfg.env.num_common_group, :]),
                terrain_propri_encoder(
                    obs_history[-env_cfg.env.num_adaptive_group :, :]
                ),
            ),
            dim=0,
        )
        actions = policy(
            torch.cat((est, obs, commands, terrain_latent), dim=-1).detach()
        )

        env.gaits[:, 0] = 2.5
        env.gaits[:, 5] = 0.05
        env.gaits[:, 6] = 0.3
        # env.gaits[:, 1:5] = to_torch([0.5, 0.5, 0.0, 0.5], device=env.device)  # troting
        # env.gaits[:, 1:5] = to_torch(
        #     [0.5, 0.25, 0.75, 0.75], device=env.device
        # )  # walking
        # env.gaits[:, 1:5] = to_torch(
        #     [0.5, 0.0, 0.5, 0.5], device=env.device
        # )  # pacing
        # env.gaits[:, 1:5] = to_torch(
        #     [0.0, 0.0, 0.0, 0.5], device=env.device
        # )  # pronking
        # env.gaits[:, 1:5] = to_torch(
        #     [0.0, 0.5, 0.5, 0.5], device=env.device
        # )  # bounding
        env.commands[:, :] = to_torch(
            [
                env.command_ranges["lin_vel_x"][1] * 0.9,
                env.command_ranges["lin_vel_y"][1] * 0.0,
                0,
            ],
            device=env.device,
        )
        # env.gaits[:, 5] = 0.15
        # if i < stop_state_log / 2:
        #     env.commands[:, :] = to_torch(
        #         [
        #             env.command_ranges["lin_vel_x"][1] * 0.9,
        #             env.command_ranges["lin_vel_y"][1] * 0.0,
        #             0,
        #         ],
        #         device=env.device,
        #     ).repeat(env.num_envs, 1)
        #     env.gaits[:, :] = to_torch(
        #         [2, 0.5, 0.5, 0.0, 0.5, 0.03], device=env.device
        #     ).repeat(env.num_envs, 1)
        # else:
        #     env.commands[:, :] = to_torch(
        #         [
        #             env.command_ranges["lin_vel_x"][1] * 0.9,
        #             env.command_ranges["lin_vel_y"][1] * 0.0,
        #             0,
        #         ],
        #         device=env.device,
        #     ).repeat(env.num_envs, 1)
        #     env.gaits[:, :] = to_torch(
        #         [2, 0.5, 0.5, 0.0, 0.5, 0.15], device=env.device
        #     ).repeat(env.num_envs, 1)
        obs, _, rews, dones, infos, obs_history, commands, height_samples = env.step(
            actions.detach()
        )
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(
                    BLIND_LOCOMOTION_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item()
                    * env.cfg.control.action_scale,
                    "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)