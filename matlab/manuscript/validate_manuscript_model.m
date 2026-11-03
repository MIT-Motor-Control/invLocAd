function validate_manuscript_model(output_dir)
% Check step budgets, seeded repeatability, saved-trial reuse, mismatch rejection and sweep restart.
% Inputs:  output_dir - directory for validation trials (default: a new temporary folder)
if nargin < 1; output_dir = tempname; end
validate_protocol_step_budgets;
if ~isfolder(output_dir); mkdir(output_dir); end
single = run_manuscript_trial(fullfile(output_dir, 'single.mat'), seed=42, max_iterations=1);
assert(size(single.param, 2) == 1);
first = run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=42, max_iterations=8);
repeat = run_manuscript_trial(fullfile(output_dir, 'repeat.mat'), seed=42, max_iterations=8);
assert(isequaln(first.param, repeat.param));
assert(isequaln(first.energy, repeat.energy));
loaded = run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=42, max_iterations=8);
assert(isequaln(loaded.param, first.param));
different = run_manuscript_trial(fullfile(output_dir, 'different.mat'), seed=43, max_iterations=8);
assert(~isequaln(first.param, different.param));
try
    run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=43, max_iterations=8);
    error('invLocAd:ExpectedRejection', 'Mismatched checkpoint should fail.');
catch error_info
    assert(strcmp(error_info.identifier, 'invLocAd:CheckpointMismatch'));
end
sweep_dir = fullfile(output_dir, 'sweep');
first_sweep = run_manuscript_sweep(sweep_dir, speed_differences=0.7, learning_rates=0.02, energy_weights=0.5, trials=2, max_iterations=8);
second_sweep = run_manuscript_sweep(sweep_dir, speed_differences=0.7, learning_rates=0.02, energy_weights=0.5, trials=2, max_iterations=8);
assert(isequal(first_sweep, second_sweep));
disp('Manuscript engine: seeded repeatability, saved-trial reuse, mismatch rejection and sweep resume passed.');
end

function validate_protocol_step_budgets()
% Check that odd step requests are rounded up to complete step pairs before budgeting.
old_path = path;
restore_path = onCleanup(@() path(old_path));
engine_dir = fullfile(fileparts(mfilename('fullpath')), 'engine');
addpath(engine_dir, '-begin');
for requested_steps = [2 3]
    initial = struct('Learner', struct('numStepsPerIteration', requested_steps));
    protocol = loadProtocolParameters(initial);
    fixed_budget = loadHowLongParameters(initial);
    expected_steps = 2*ceil(requested_steps/2);
    for parameters = {protocol, fixed_budget}
        configured = parameters{1};
        assert(configured.Learner.numStepsPerIteration == expected_steps);
        assert(configured.numIterations == floor(configured.numStepsToLearn/expected_steps));
        assert(configured.numIterations*expected_steps <= configured.numStepsToLearn);
    end
end
end
