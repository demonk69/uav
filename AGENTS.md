# Project Rules

These rules are mandatory for all work in `/home/lab_726/uav_rendezvous_rl`.

1. All Isaac Lab commands must run through:

   ```bash
   /home/lab_726/IsaacLab/isaaclab.sh
   ```

2. Do not use Anaconda Python.

3. Before running Isaac Lab, clear:

   ```text
   CONDA_PREFIX
   CONDA_DEFAULT_ENV
   VIRTUAL_ENV
   PYTHONPATH
   PYTHONHOME
   ```

4. Do not modify:

   ```text
   /home/lab_726/IsaacLab
   /home/lab_726/isaacsim
   Pegasus Simulator
   system Python dependencies
   NVIDIA driver
   ```

5. Do not execute `sudo`.

6. This project is a non-contact offset rendezvous task. Collision is not allowed.

7. The Actor is forbidden from using:

   ```text
   target future states
   target future control commands
   complete future trajectories
   target motion mode labels
   trajectory generator parameters
   simulator privileged information unavailable at deployment
   ```

8. Fixed definitions:

   ```text
   p_rel_w = p_target_w - p_ego_w
   v_rel_w = v_target_w - v_ego_w
   e_offset_w = p_ego_w - p_target_w - b_des_w
   ```

9. This work session only executes M7B: simplified dynamics randomization, control execution delay, and wind disturbance robustness validation. M7C is not authorized.

10. Before each work session, reread:

    ```text
    AGENTS.md
    docs/environment_audit.md
    docs/implementation_plan.md
    docs/milestone_state.md
    ```

11. After each completed task, update `docs/milestone_state.md`.

12. Do not enter the next milestone or create a Git commit without user confirmation.

13. M7B must not modify the behavior of the accepted M2 through M7A tasks:

    ```text
    Isaac-Uav-Rendezvous-Direct-v0
    Isaac-Uav-Rendezvous-Baseline-v0
    Isaac-Uav-Rendezvous-RL-v0
    Isaac-Uav-Rendezvous-Recurrent-v0
    Isaac-Uav-Rendezvous-M6-Feedforward-Ablation-v0
    Isaac-Uav-Rendezvous-M7A-GRU-v0
    Isaac-Uav-Rendezvous-M7A-Feedforward-v0
    ```

14. In M7B, the Actor must not receive: randomized dynamics parameters (tau_velocity_scale, acceleration_limit_scale, speed_limit_scale, linear_drag), actual action delay, wind acceleration truth, gust acceleration truth, gust remaining steps, target motion mode labels, target generator parameters, target acceleration truth, future target states, future target commands, future segment schedules, complete future trajectories, or other simulator privileged information unavailable at deployment.

15. The M7B Critic may receive current-time privileged dynamics and wind information (tau_velocity_scale, acceleration_limit_scale, speed_limit_scale, linear_drag, normalized action_delay_steps, current wind_acceleration_w) but must not receive future wind, future gust, or future target information.

16. The primary M7B policy is feedforward PPO. GRU is a secondary ablation only and must not be trained to convergence or presented as the default candidate.

17. M7B uses the clean M7A observation pipeline (delay=0, dropout=0, noise=0). Observation degradation and dynamics randomization must not be combined until explicitly authorized.

18. Non-contact offset rendezvous remains the task: collision risk must not be rewarded or used as a training objective, and the safety distance d_safe must not be weakened.
