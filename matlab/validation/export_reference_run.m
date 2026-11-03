function export_reference_run(outputPath, numIterations)
% Save a noise-free baseline run as the reference for scripts/validate_matlab_python_parity.py.
% Inputs:  outputPath    - MAT file to write (default: tempdir/invLocAd_reference.mat)
%          numIterations - learning iterations to simulate (default 10)
% The file holds objectiveValue (objective of the initial controller),
% controllerStore8D, pInput, stateVar0 and contextNow.

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile(tempdir, 'invLocAd_reference.mat');
end

if nargin < 2 || isempty(numIterations)
    numIterations = 10;
end

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
addpath(fullfile(matlabDir, 'src'));
set(0, 'DefaultFigureVisible', 'off');

rng(42, 'twister');
paramFixed = [];
paramFixed = loadBipedModelParameters(paramFixed);
paramFixed = loadSensoryNoiseParameters(paramFixed);
paramControllerGains = loadControllerGainParameters(paramFixed);
paramFixed = loadLearnerParameters(paramFixed);
paramFixed = loadProtocolParameters(paramFixed);
paramFixed = loadStoredMemoryParameters_ControlVsSpeed(paramFixed);

% Remove exploration and sensory noise so the run is deterministic.
paramFixed.Learner.noiseSTDExploratory = 0;
paramFixed.noiseEnergySensory = 0;
paramFixed.numIterations = numIterations;

pInput = loadLearnableParametersInitial(paramFixed);
stateVar0 = loadInitialBodyState(pInput);
[vA, vB] = getTreadmillSpeed(0, paramFixed.imposedFootSpeeds);
contextNow = [vA; vB];
objectiveValue = fObjective_AsymmetricNominal(pInput, stateVar0, paramControllerGains, paramFixed, 0);
controllerStore8D = simulateLearningStepByStep(paramFixed, pInput, stateVar0, contextNow, paramControllerGains);

save(outputPath, 'objectiveValue', 'controllerStore8D', 'pInput', 'stateVar0', 'contextNow');
disp(['Saved MATLAB reference to ' outputPath]);
end
