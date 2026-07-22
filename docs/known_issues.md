# Known Issues

## M2 Placeholder Rigid Objects

On the current host, spawned sphere placeholders with `kinematic_enabled=True` triggered PhysX GPU `illegal memory access` errors during Isaac Lab startup/reset. M2 therefore uses gravity-disabled dynamic `RigidObject` sphere placeholders.

The M2 placeholders are only synchronized state and visualization carriers. They are not the final UAV dynamics model. The authoritative M2 truth state is the task tensor state maintained by `UavRendezvousEnv`, including `p_ego_w`, `v_ego_w`, `p_target_w`, and `v_target_w`; observations and termination checks read those tensors, and runtime audit checks that the placeholder `RigidObject` data stays synchronized to them.

Placeholder collision remains explicitly enabled in the asset schema because this is the stable configuration verified on this host. M2 remains non-contact: runtime audit checks that the center distance stays above `d_safe`, that synchronized asset positions and velocities match task tensors, and that ego has no physics-induced drift.

Do not retry `kinematic_enabled=True` during M2 closure. Do not modify Isaac Lab, Isaac Sim, Pegasus, system Python dependencies, or NVIDIA driver components for this issue.
