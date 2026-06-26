"""Orbital Braille — multi-orb typehead prototype for VQC shard encoding."""

from .typehead import OrbitalTypehead, TypeheadConfig
from .decoder import decode_field, DecodeResult
from .lg_modes import lg_mode, lg_mode_full, project_oam_spectrum
from .altermagnetic import PWaveBMGL, apply_turbulence
from .quaternion_codec import Quaternion, encode_shard, decode_shard
from .stable_fonts import EmergentConstants, build_stable_font, font_separation
from .font_optimizer import optimize_font, FontOptResult
from .slm_typehead import (
    SLMConfig,
    SLMDevicePreset,
    SLM_PRESETS,
    SLMPackageMeta,
    export_hologram_package,
    gerchberg_saxton,
    slm_phase_pattern,
    slm_phase_sequence,
    save_phase_hologram,
    phase_to_levels,
)
from .turbulence import apply_free_space_channel, kolmogorov_phase_screen

__all__ = [
    "OrbitalTypehead",
    "TypeheadConfig",
    "decode_field",
    "DecodeResult",
    "lg_mode",
    "lg_mode_full",
    "project_oam_spectrum",
    "PWaveBMGL",
    "apply_turbulence",
    "Quaternion",
    "encode_shard",
    "decode_shard",
    "EmergentConstants",
    "build_stable_font",
    "font_separation",
    "optimize_font",
    "FontOptResult",
    "SLMConfig",
    "SLMDevicePreset",
    "SLM_PRESETS",
    "SLMPackageMeta",
    "export_hologram_package",
    "gerchberg_saxton",
    "slm_phase_pattern",
    "slm_phase_sequence",
    "save_phase_hologram",
    "phase_to_levels",
    "apply_free_space_channel",
    "kolmogorov_phase_screen",
]