function result = run_manuscript_trial(output_file, options)
% Run one seeded manuscript-engine trial and save it, or reload a matching saved trial.
% Inputs:  output_file - MAT file for the completed trial
%          options.seed - random seed
%          options.speed_difference - d (m/s); the belts run at 1 - d/2 and 1 + d/2 m/s
%          options.learning_rate, options.exploration_noise - learner settings
%          options.energy_weight - lambdaEnergyVsSymmetry, the weight on the energy term
%          options.split_minutes - split-belt duration (min)
%          options.swing_cost - swing-cost multiplier
%          options.max_iterations - iteration limit; Inf runs the full protocol
%          options.transition_seconds - tied-to-split ramp duration (s)
%          options.speed_divisor - divisor from experimental to model belt speeds
% Outputs: result - outputs of run_sim (time, sla, sta, energy, speed, state, steptime_fast,
%              steptime_slow, steplength_fast, steplength_slow, param, fall_flag, min_vert_pos)
%              and metadata (options, MATLAB version, engine source text, final RNG state)
% An existing output_file is reloaded only if its options and engine source text match.
arguments
    output_file (1,1) string
    options.seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.speed_difference (1,1) double {mustBePositive, mustBeLessThan(options.speed_difference, 2)} = 0.7
    options.learning_rate (1,1) double {mustBeNonnegative, mustBeFinite} = 0.02
    options.energy_weight (1,1) double {mustBeGreaterThanOrEqual(options.energy_weight, 0), mustBeLessThanOrEqual(options.energy_weight, 1)} = 0.5
    options.split_minutes (1,1) double {mustBePositive, mustBeFinite} = 45
    options.exploration_noise (1,1) double {mustBeNonnegative, mustBeFinite} = 0.01
    options.swing_cost (1,1) double {mustBePositive, mustBeFinite} = 1
    options.max_iterations (1,1) double {mustBePositive} = Inf
    options.transition_seconds (1,1) double {mustBePositive, mustBeFinite} = 25
    options.speed_divisor (1,1) double {mustBePositive, mustBeFinite} = 2
end
if options.seed > 2^32 - 1 || (~isinf(options.max_iterations) && mod(options.max_iterations, 1) ~= 0)
    error('invLocAd:InvalidConfiguration', 'Seed must fit uint32 and max_iterations must be an integer or Inf.');
end
here = fileparts(mfilename('fullpath'));
old_path = path;
old_rng = rng;
cleanup = onCleanup(@() restore_session(old_path, old_rng));
addpath(fullfile(here, 'engine'), '-begin');
engine_files = dir(fullfile(here, 'engine', '*.m'));
source = struct();
for i = 1:numel(engine_files)
    file = fullfile(engine_files(i).folder, engine_files(i).name);
    source.(erase(engine_files(i).name, '.m')) = fileread(file);
end
if isfile(output_file)
    result = load(output_file);
    if ~isfield(result, 'metadata') || ~isequaln(result.metadata.options, options) || ...
            ~isequal(result.metadata.engine_source, source)
        error('invLocAd:CheckpointMismatch', 'Existing trial uses different settings or engine source. Choose a new output file.');
    end
    validate_result(result);
    fprintf('Loaded completed trial: %s\n', output_file);
    return
end
% Initial learned controller targets, as in loadLearnableParametersInitial.
controller = [0.328221262798818; 0.310751796902254; 0.153556843539029; ...
    0.328221491356562; 0.310751694570805; 0.153557221281688; ...
    -0.000000038953735; -0.000000038953735];
slow = 1 - options.speed_difference / 2;
fast = 1 + options.speed_difference / 2;
[result.time, result.sla, result.sta, result.energy, result.speed, result.state, ...
    result.steptime_fast, result.steptime_slow, result.steplength_fast, result.steplength_slow, ...
    result.param, result.fall_flag, result.min_vert_pos] = run_sim( ...
    options.speed_difference, options.split_minutes, slow, fast, options.exploration_noise, ...
    options.learning_rate, options.energy_weight, controller, options.swing_cost, options);
result.metadata = struct('schema_version', 1, 'options', options, ...
    'matlab_version', version, 'engine_source', source, ...
    'completed_iterations', size(result.param, 2), 'rng_final', rng, ...
    'kind', 'completed_trial', 'created_at', char(datetime('now', 'TimeZone', 'UTC')));
validate_result(result);
parent = fileparts(output_file);
if strlength(parent) == 0; parent = "."; end
if ~isfolder(parent); mkdir(parent); end
temporary = string(tempname(parent)) + ".mat";
remove_temporary = onCleanup(@() delete_if_present(temporary));
save(temporary, '-struct', 'result', '-v7');
movefile(temporary, output_file);
fprintf('Saved %d completed iterations: %s\n', size(result.param, 2), output_file);
end

function validate_result(result)
% Check the saved-trial schema and that the required results are finite.
if ~isfield(result, 'metadata') || result.metadata.schema_version ~= 1 || ~strcmp(result.metadata.kind, 'completed_trial')
    error('invLocAd:InvalidResult', 'Unsupported saved-trial schema.');
end
required = {'energy', 'time', 'sla', 'sta', 'param', 'fall_flag'};
for i = 1:numel(required)
    if ~isfield(result, required{i}) || isempty(result.(required{i})) || any(~isfinite(result.(required{i})), 'all')
        error('invLocAd:InvalidResult', 'Missing, empty or nonfinite result: %s', required{i});
    end
end
if size(result.param, 1) ~= 8 || numel(result.energy) ~= numel(result.time)
    error('invLocAd:InvalidResult', 'Unexpected controller or time-series dimensions.');
end
end

function restore_session(old_path, old_rng)
% Restore the MATLAB path and random number generator state.
path(old_path);
rng(old_rng);
end

function delete_if_present(file)
% Delete a file if it exists.
if isfile(file); delete(file); end
end
