"""Shared pytest fixtures."""

import pytest

from invLocAd.parameter_loading.load_biped_model_parameters import (
    load_biped_model_parameters,
)
from invLocAd.parameter_loading.load_controller_gain_parameters import (
    load_controller_gain_parameters,
)
from invLocAd.parameter_loading.load_learner_parameters import (
    load_learner_parameters,
)
from invLocAd.parameter_loading.load_protocol_parameters import (
    load_protocol_parameters,
)
from invLocAd.parameter_loading.load_sensory_noise_parameters import (
    load_sensory_noise_parameters,
)
from invLocAd.parameter_loading.load_stored_memory_parameters_control_vs_speed import (
    load_stored_memory_parameters_control_vs_speed,
)


@pytest.fixture
def model_parameters():
    """Return the default fixed parameters and controller with the belt-speed schedule attached."""
    fixed = load_biped_model_parameters({})
    fixed = load_sensory_noise_parameters(fixed)
    controller = load_controller_gain_parameters(fixed)
    fixed = load_learner_parameters(fixed)
    fixed = load_protocol_parameters(fixed)
    fixed = load_stored_memory_parameters_control_vs_speed(fixed)
    protocol = fixed["imposedFootSpeeds"]
    controller["tList_BeltSpeed"] = protocol["tList"]
    controller["PushoffFootSpeedNowList"] = protocol["footSpeed1List"]
    controller["PushoffAccelerationNowList"] = protocol["footAcc1List"]
    return fixed, controller
