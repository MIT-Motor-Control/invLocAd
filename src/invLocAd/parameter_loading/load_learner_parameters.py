"""Settings of the policy-gradient learner."""
from __future__ import annotations


def load_learner_parameters(param_fixed: dict | None = None) -> dict:
    """Add the learner settings as ``param_fixed["Learner"]``.

    Load the sensory noise parameters first; the prediction-error threshold depends on them.

    Args:
        param_fixed: Parameter dictionary to update in place; created when None.

    Returns:
        The updated ``param_fixed``.
    """
    if param_fixed is None:
        param_fixed = {}

    learner = param_fixed.get('Learner', {})

    # Gradient step size and standard deviation of the exploratory controller noise.
    learner['LearningRate'] = 0.00012
    learner['noiseSTDExploratory'] = 0.002
    # Include the learned body dynamics in the steady-state energy gradient.
    learner['includeInternalModel'] = 1
    # Recent iterations fitted by the dynamics and energy estimators; fitting starts once
    # this many iterations are complete.
    learner['numStepsToUseForEstimator'] = 30

    # Rates of moving toward the context-dependent memory and of updating it; the
    # exponents sharpen the direction-cosine weighting of each update.
    learner['LearningRateTowardMemory'] = 0.01

    learner['powerToTheMoveToMemory'] = 10
    learner['powerToTheMemoryFormation'] = 10

    learner['LearningRateForMemoryFormationUpdates'] = 0.03

    learner['numStepsPerIteration'] = 2

    # The gradient step norm is capped at trustRegionSize * sqrt(8).
    learner['trustRegionSize'] = 0.25 * learner['noiseSTDExploratory']
    learner['shouldWeUseTrustRegion'] = 1

    # Gating of the gradient by energy prediction error is disabled.
    learner['predictionErrorThreshold'] = 2 * param_fixed.get('noiseEnergySensory', 0)
    learner['shouldWeThresholdPredictionError'] = 0

    # Not read by the simulation; the estimators use a finite window without forgetting.
    learner['alphaForgettingForEstimator'] = 0

    param_fixed['Learner'] = learner
    return param_fixed
