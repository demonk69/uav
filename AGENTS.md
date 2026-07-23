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

9. This work session only executes M6. M7 is not authorized.

10. Before each work session, reread:

    ```text
    AGENTS.md
    docs/environment_audit.md
    docs/implementation_plan.md
    docs/milestone_state.md
    ```

11. After each completed task, update `docs/milestone_state.md`.

12. Do not enter the next milestone or create a Git commit without user confirmation.
