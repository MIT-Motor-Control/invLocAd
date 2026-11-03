function summary = run_manuscript_sweep(output_dir, options)
% Run a grid of independent seeded trials and write summary.csv.
% Inputs:  output_dir - directory for sweep.mat, trial_NNNNNN.mat and summary.csv
%          options.speed_differences, options.learning_rates, options.energy_weights - grid
%          options.trials - trials per grid point
%          options.seed - seed of the first trial; trial i uses seed + i - 1
%          options.max_iterations - iteration limit per trial
% Outputs: summary - one row per trial: job, speed_difference, learning_rate, psi, trial,
%              iterations and fell
% Completed trials are reloaded on rerun; an existing sweep.mat must have the same options.
arguments
    output_dir (1,1) string
    options.speed_differences (1,:) double = [0.4 0.7 1.0]
    options.learning_rates (1,:) double = linspace(0.005, 0.05, 20)
    options.energy_weights (1,:) double = linspace(0, 1, 20)
    options.trials (1,1) double {mustBePositive, mustBeInteger} = 50
    options.seed (1,1) double {mustBeNonnegative, mustBeInteger} = 0
    options.max_iterations (1,1) double {mustBePositive} = Inf
end
validateattributes(options.speed_differences, {'double'}, {'nonempty','finite','positive','<',2});
validateattributes(options.learning_rates, {'double'}, {'nonempty','finite','nonnegative'});
validateattributes(options.energy_weights, {'double'}, {'nonempty','finite','>=',0,'<=',1});
[d, a, w, t] = ndgrid(options.speed_differences, options.learning_rates, options.energy_weights, 1:options.trials);
jobs = [d(:), a(:), w(:), t(:)];
if options.seed + size(jobs,1) - 1 > 2^32 - 1
    error('invLocAd:InvalidConfiguration', 'The sweep seeds exceed uint32.');
end
if ~isfolder(output_dir); mkdir(output_dir); end
manifest_file = fullfile(output_dir, 'sweep.mat');
if isfile(manifest_file)
    previous = load(manifest_file, 'options', 'jobs');
    if ~isequaln(previous.options, options) || ~isequal(previous.jobs, jobs)
        error('invLocAd:SweepMismatch', 'Existing sweep has a different grid or seed. Choose a new directory.');
    end
else
    save(manifest_file, 'options', 'jobs');
end
summary = table((1:size(jobs,1))', jobs(:,1), jobs(:,2), 1-jobs(:,3), jobs(:,4), ...
    zeros(size(jobs,1),1), zeros(size(jobs,1),1), ...
    'VariableNames', {'job', 'speed_difference', 'learning_rate', 'psi', 'trial', 'iterations', 'fell'});
for i = 1:size(jobs,1)
    file = fullfile(output_dir, sprintf('trial_%06d.mat', i));
    result = run_manuscript_trial(file, seed=options.seed+i-1, ...
        speed_difference=jobs(i,1), learning_rate=jobs(i,2), energy_weight=jobs(i,3), ...
        max_iterations=options.max_iterations);
    summary.iterations(i) = size(result.param, 2);
    summary.fell(i) = result.fall_flag(1);
end
writetable(summary, fullfile(output_dir, 'summary.csv'));
end
