"""Canonical Ant-v5/PPO utilities for the ProxyGap project."""

from .ant_wrapper import ProxyGapAntWrapper, STEP_LOG_SCHEMA, make_proxygap_ant_env
from .experiment import (
    CHECKPOINT_FRACTIONS,
    DEFAULT_PPO_CONFIG,
    checkpoint_targets,
    evaluate_model,
    resolve_ppo_config,
    select_representative_evaluation_seed,
)
from .metrics import (
    CSV_SCHEMA,
    EpisodeMetrics,
    classify_termination,
    common_rescored_return,
    quaternion_tilt_angle,
)
from .protocol import protocol_freeze_status, validate_prospective_protocol

__all__ = [
    "CHECKPOINT_FRACTIONS",
    "CSV_SCHEMA",
    "DEFAULT_PPO_CONFIG",
    "EpisodeMetrics",
    "ProxyGapAntWrapper",
    "STEP_LOG_SCHEMA",
    "checkpoint_targets",
    "evaluate_model",
    "classify_termination",
    "common_rescored_return",
    "make_proxygap_ant_env",
    "quaternion_tilt_angle",
    "resolve_ppo_config",
    "select_representative_evaluation_seed",
    "protocol_freeze_status",
    "validate_prospective_protocol",
]
