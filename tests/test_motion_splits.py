"""Unit tests for M3 train/validation/test target motion splits."""

import pytest
import torch

from uav_rendezvous_rl.motions import TargetMotionManagerCfg
from uav_rendezvous_rl.motions.configs import get_split_cfg
from uav_rendezvous_rl.motions.sampling import make_split_generator, sample_initial_target_state


def test_split_configs_use_separate_ranges_and_seed_domains() -> None:
    cfg = TargetMotionManagerCfg()

    assert cfg.train.target_pos_x_range != cfg.validation.target_pos_x_range
    assert cfg.validation.target_pos_x_range != cfg.test.target_pos_x_range
    assert cfg.train.seed_offset != cfg.validation.seed_offset
    assert cfg.validation.seed_offset != cfg.test.seed_offset


def test_split_sampling_respects_position_ranges() -> None:
    cfg = TargetMotionManagerCfg()
    num_envs = 8
    p_ego_w = torch.tensor([[0.0, 0.0, 1.5]]).repeat(num_envs, 1)
    samples = {}

    for split in ("train", "validation", "test"):
        split_cfg = get_split_cfg(cfg, split)
        generator = make_split_generator(42, cfg, split)
        p_target_w, _v_target_w = sample_initial_target_state(num_envs, p_ego_w, cfg, split_cfg, generator, "cpu")
        samples[split] = p_target_w
        assert bool(torch.all(p_target_w[:, 0] >= split_cfg.target_pos_x_range[0]).item())
        assert bool(torch.all(p_target_w[:, 0] <= split_cfg.target_pos_x_range[1]).item())
        assert bool(torch.all(p_target_w[:, 1] >= split_cfg.target_pos_y_range[0]).item())
        assert bool(torch.all(p_target_w[:, 1] <= split_cfg.target_pos_y_range[1]).item())

    assert not torch.equal(samples["train"], samples["validation"])
    assert not torch.equal(samples["validation"], samples["test"])


def test_unknown_split_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown target motion split"):
        get_split_cfg(TargetMotionManagerCfg(), "holdout")
