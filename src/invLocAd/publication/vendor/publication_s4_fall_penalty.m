function publication_s4_fall_penalty(data_dir, output_dir)
% Reproduce Supplementary Figure 4 and the comparison of models with and without a fall penalty.
% Model C selects learning rate, symmetry weight and fall weight per speed; model E fixes the
% fall weight at zero. Cached cell mean trajectories and participant mean trajectories are
% summarized by bounded exponential fits and models are selected by the minimum RMSE between
% fits. Each leave-one-subject-out fold selects on the remaining participants and scores the
% held-out participant; the noise floor is the RMSE between the training-group and held-out fits.
% Inputs:  data_dir   - prepared publication data folder containing extensions/s5 and
%                       extensions/experimental_csv/split_energy.csv
%          output_dir - folder for the output tables and figure
% Outputs: s4_energy_timeseries.csv, s4_selected_models.csv, s4_experimental_subjects.csv,
%          s4_loso_folds.csv, s4_loso_summary.csv, s4_loso_tests.csv, s4_mechanical_cells.csv,
%          s4_mechanical_weight_slices.csv, s4_recomputed_selection.csv and s4_fall_penalty.png
%          in output_dir
speeds = [0.4 0.7 1.0];
cohort = [8 9 9];
nt = 2700;
base = fullfile(data_dir, 'extensions', 's5');
config = jsondecode(fileread(fullfile(base, 'grid_config.json')));
summary = jsondecode(fileread(fullfile(base, 'analysis_summary.json')));
saved = summary.selection;
assert(isequal(config.speeds(:)', speeds), 'Unexpected fall-penalty speeds');
weights = config.fall_weights(:);
assert(numel(unique(weights)) == numel(weights) && any(weights == 0), ...
    'Fall-weight grid must have distinct values including zero');
[ids, condition_speed, raw] = read_split_energy( ...
    fullfile(data_dir, 'extensions', 'experimental_csv', 'split_energy.csv'), nt);
if ~isfolder(output_dir), mkdir(output_dir); end

% Experimental mean and SEM against the saved model C and E selections.
labels = {'C', 'fall_weight_free'; 'E', 'fall_weight_zero'};
experimental = cell(1, 3);
participants = cell(1, 3);
series = zeros(3 * nt, 7);
selected = zeros(6, 7);
subject_rows = cell(3, 1);
minutes = (0:nt-1) / 60;
figure_handle = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 10 3]);
layout = tiledlayout(figure_handle, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for c = 1:3
    rows = find(condition_speed == speeds(c));
    assert(numel(rows) == cohort(c), 'Unexpected energy cohort at %g m/s', speeds(c));
    em = zeros(numel(rows), nt);
    for i = 1:numel(rows)
        em(i, :) = energy_percent(raw(rows(i), :));
    end
    experimental{c} = em;
    participants{c} = ids(rows);
    [average, count] = nan_mean_rows(em);
    sem = nan_std_rows(em, average, count) ./ sqrt(sum(isfinite(em), 1));
    prediction = zeros(2, nt);
    speed_key = matlab.lang.makeValidName(sprintf('%.1f', speeds(c)));
    for k = 1:2
        choice = saved.(labels{k, 1}).per_speed.(speed_key);
        assert(k == 1 || choice.w == 0, ...
            'The selected model without a fall term has a nonzero fall weight');
        entry = read_cell(base, speeds(c), choice.lr, choice.sym, choice.w);
        successful = numel(entry.ok_seeds);
        assert(successful > 0 && isequal(size(entry.energy_pct), [successful nt]), ...
            'Fall-penalty cell has no valid complete energy trajectories');
        prediction(k, :) = nan_mean_rows(entry.energy_pct);
        assert(all(isfinite(prediction(k, :))), 'Selected model mean has nonfinite values');
        selected(2 * (c - 1) + k, :) = [speeds(c), k, choice.lr, choice.sym, choice.w, ...
            successful, numel(entry.failed_seeds)];
    end
    series((c - 1) * nt + (1:nt), :) = [repmat(speeds(c), nt, 1), (0:nt-1)', minutes', ...
        average', sem', prediction(1, :)', prediction(2, :)'];
    np = numel(rows);
    subject_rows{c} = {repmat(speeds(c), np * nt, 1), repelem(participants{c}, nt, 1), ...
        repmat((0:nt-1)', np, 1), reshape(em', [], 1)};
    plot_condition(nexttile(layout), minutes, average, sem, prediction, speeds(c), ...
        saved.C.per_speed.(speed_key).w, c == 1);
end
key = legend(nexttile(layout, 2), {'Experiment', 'Without fall', 'With fall'}, ...
    'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 8);
key.Layout.Tile = 'south';
title(layout, 'Supplementary Figure 4: models with and without a fall penalty', ...
    'FontSize', 10);
exportgraphics(figure_handle, fullfile(output_dir, 's4_fall_penalty.png'), 'Resolution', 300);
close(figure_handle);

% Refit every cached cell and record its mechanical summaries.
% Rows of keys are (speed, learning rate, symmetry weight, fall weight), fall weight fastest.
lrs = config.learning_rates(:);
syms = config.symmetry_weights(:);
[iw, is, il, iy] = ndgrid(1:numel(weights), 1:numel(syms), 1:numel(lrs), 1:numel(speeds));
keys = [speeds(iy(:))', lrs(il(:)), syms(is(:)), weights(iw(:))];
ncell = size(keys, 1);
curves = zeros(ncell, nt);
mechanical = zeros(ncell, 11);
for k = 1:ncell
    entry = read_cell(base, keys(k, 1), keys(k, 2), keys(k, 3), keys(k, 4));
    seeds = double([entry.ok_seeds; entry.failed_seeds]);
    assert(isequal(sort(seeds), sort(config.seeds(:))), ...
        'Fall-penalty cell has missing or duplicate seeds');
    n = numel(entry.ok_seeds);
    assert(n > 0 && isequal(size(entry.energy_pct), [n nt]) && ...
        numel(entry.mean_penalty) == n && numel(entry.min_com) == n, ...
        'Fall-penalty cell has invalid records');
    assert(all(isfinite(entry.energy_pct(:))) && all(isfinite(entry.mean_penalty)) && ...
        all(isfinite(entry.min_com)), 'Fall-penalty cell contains nonfinite values');
    curves(k, :) = fit_energy_exponential(nan_mean_rows(entry.energy_pct));
    mechanical(k, :) = [keys(k, :), n, numel(entry.failed_seeds), ...
        pairwise_sum(entry.mean_penalty) / n, min(entry.min_com), max(entry.min_com), ...
        min(entry.mean_penalty), max(entry.mean_penalty)];
end

% Group selections and leave-one-subject-out folds.
recomputed = zeros(6, 6);
folds = zeros(sum(cohort), 11);
fold_ids = strings(sum(cohort), 1);
row = 0;
for c = 1:3
    em = experimental{c};
    at_speed = keys(:, 1) == speeds(c);
    candidates = {find(at_speed), find(at_speed & keys(:, 4) == 0)};
    group = fit_energy_exponential(nan_mean_rows(em));
    for k = 1:2
        [error_value, chosen] = select_cell(curves, candidates{k}, group);
        recomputed(2 * (c - 1) + k, :) = [speeds(c), k, keys(chosen, 2:4), error_value];
    end
    for i = 1:size(em, 1)
        held = fit_energy_exponential(em(i, :));
        train = fit_energy_exponential(nan_mean_rows(em([1:i-1, i+1:end], :)));
        row = row + 1;
        fold_ids(row) = participants{c}(i);
        folds(row, 1:3) = [speeds(c), size(em, 1) - 1, rmse(train', held')];
        for k = 1:2
            [~, chosen] = select_cell(curves, candidates{k}, train);
            folds(row, 4 * k + (0:3)) = [rmse(curves(chosen, :)', held'), keys(chosen, 2:4)];
        end
    end
end

% Held-out RMSE summaries and paired tests of model C against model E.
summary_delta = strings(12, 1);
summary_values = zeros(12, 3);
summary_model = strings(12, 1);
tests = zeros(4, 7);
columns = [4 8 3];
model_names = ["C"; "E"; "noise_floor"];
for g = 1:4
    if g < 4
        in_group = folds(:, 1) == speeds(g);
        delta_text = number_text(speeds(g));
    else
        in_group = true(size(folds, 1), 1);
        delta_text = "pooled";
    end
    for k = 1:3
        values = folds(in_group, columns(k));
        summary_delta(3 * (g - 1) + k) = delta_text;
        summary_model(3 * (g - 1) + k) = model_names(k);
        summary_values(3 * (g - 1) + k, :) = [numel(values), ...
            pairwise_sum(values) / numel(values), numpy_median(values)];
    end
    difference = folds(in_group, 4) - folds(in_group, 8);
    nonzero = nnz(difference);
    p = 1;
    if nonzero > 0
        p = wilcoxon_p(difference);
    end
    tests(g, :) = [numel(difference), nonzero, numel(difference) - nonzero, ...
        pairwise_sum(difference) / numel(difference), numpy_median(difference), p, NaN];
end
% Holm step-down adjustment across the three speed conditions.
[sorted_p, order] = sort(tests(1:3, 6));
running = 0;
for i = 1:3
    running = max(running, min(1, (4 - i) * sorted_p(i)));
    tests(order(i), 7) = running;
end

% Saved selections must agree with the recomputed selections.
assert(isequal(selected(:, 1:5), recomputed(:, 1:5)), ...
    'Recomputed fall-penalty selection disagrees with the saved selection');

model_text = ["C"; "E"];
label_text = string(labels(:, 2));
write_csv(fullfile(output_dir, 's4_energy_timeseries.csv'), ...
    {'delta_mps', 'time_s', 'time_min', 'experiment_mean_pct', 'experiment_sem_pct', ...
    'model_C_fall_weight_free_pct', 'model_E_fall_weight_zero_pct'}, num2cell(series, 1));
write_csv(fullfile(output_dir, 's4_selected_models.csv'), ...
    {'delta_mps', 'model', 'label', 'learning_rate', 'symmetry_weight', 'fall_weight', ...
    'successful_seeds', 'failed_seeds'}, ...
    [{selected(:, 1), model_text(selected(:, 2)), label_text(selected(:, 2))}, ...
    num2cell(selected(:, 3:7), 1)]);
subjects = vertcat(subject_rows{:});
write_csv(fullfile(output_dir, 's4_experimental_subjects.csv'), ...
    {'delta_mps', 'subject_id', 'time_s', 'energy_change_pct'}, ...
    {vertcat(subjects{:, 1}), vertcat(subjects{:, 2}), vertcat(subjects{:, 3}), ...
    vertcat(subjects{:, 4})});
write_csv(fullfile(output_dir, 's4_loso_folds.csv'), ...
    {'delta_mps', 'subject_id', 'training_participants', 'noise_floor_rmse', 'rmse_C', ...
    'learning_rate_C', 'symmetry_weight_C', 'fall_weight_C', 'rmse_E', 'learning_rate_E', ...
    'symmetry_weight_E', 'fall_weight_E'}, ...
    [{folds(:, 1), fold_ids}, num2cell(folds(:, 2:11), 1)]);
write_csv(fullfile(output_dir, 's4_loso_summary.csv'), ...
    {'delta_mps', 'model', 'n', 'mean_rmse', 'median_rmse'}, ...
    [{summary_delta, summary_model}, num2cell(summary_values, 1)]);
write_csv(fullfile(output_dir, 's4_loso_tests.csv'), ...
    {'delta_mps', 'n', 'n_effective', 'n_zero_differences', 'mean_difference_C_minus_E', ...
    'median_difference_C_minus_E', 'wilcoxon_two_sided_p', 'method', ...
    'holm_p_within_three_conditions'}, ...
    [{[number_text(speeds(:)); "pooled"]}, num2cell(tests(:, 1:6), 1), ...
    {repmat("scipy.wilcoxon, two-sided, zero_method=wilcox, method=auto", 4, 1), ...
    [number_text(tests(1:3, 7)); "not_in_family"]}]);
write_csv(fullfile(output_dir, 's4_recomputed_selection.csv'), ...
    {'delta_mps', 'model', 'learning_rate', 'symmetry_weight', 'fall_weight', 'group_fit_rmse'}, ...
    [{recomputed(:, 1), model_text(recomputed(:, 2))}, num2cell(recomputed(:, 3:6), 1)]);
mechanical_names = {'delta_mps', 'learning_rate', 'symmetry_weight', 'fall_weight', ...
    'successful_seeds', 'failed_seeds', 'mean_penalty', 'min_com_height_over_seeds', ...
    'max_seed_min_com_height', 'min_seed_mean_penalty', 'max_seed_mean_penalty'};
write_csv(fullfile(output_dir, 's4_mechanical_cells.csv'), mechanical_names, ...
    num2cell(mechanical, 1));
% Hold the fitted learning rate and symmetry weight fixed while varying fall weight.
slices = zeros(0, 11);
for c = 1:3
    fit_c = recomputed(2 * c - 1, :);
    on_slice = all(mechanical(:, 1:3) == [fit_c(1), fit_c(3:4)], 2);
    slices = [slices; mechanical(on_slice, :)]; %#ok<AGROW>
end
write_csv(fullfile(output_dir, 's4_mechanical_weight_slices.csv'), mechanical_names, ...
    num2cell(slices, 1));
fprintf('S4 completed: %d cells refitted, %d folds, Holm p = %.16g/%.16g/%.16g\n', ...
    ncell, size(folds, 1), tests(1:3, 7));
end

function [ids, speed, values] = read_split_energy(path, nt)
% Read participant identifiers, belt speed differences and the 1-Hz energy samples.
% Inputs:  path - split_energy.csv with subject_id, speed_diff_ms and time columns t0, t1, ...
%          nt   - required number of time columns
% Outputs: ids    - participant identifiers (string column)
%          speed  - belt speed difference (m/s) per participant
%          values - participants x nt energy samples in time order
lines = readlines(path, 'EmptyLineRule', 'skip');
header = split(lines(1), ',')';
fields = split(lines(2:end), ',');
time_columns = find(matches(header, "t" + digitsPattern));
seconds = str2double(extractAfter(header(time_columns), 1));
assert(numel(time_columns) == nt && issorted(seconds), ...
    'Supplementary Figure 4 requires %d ordered experimental energy samples', nt);
ids = fields(:, header == "subject_id");
speed = str2double(fields(:, header == "speed_diff_ms"));
values = str2double(fields(:, time_columns));
end

function entry = read_cell(base, speed, lr, sym, weight)
% Read one cached fall-penalty cell and check that its grid coordinates match the request.
% Inputs:  base - extensions/s5 folder; speed, lr, sym, weight - grid coordinates
% Outputs: entry - read_npz struct with energy_pct converted to double
name = sprintf('sp%g_lr%.3e_sym%g_w%g.npz', speed, lr, sym, weight);
entry = read_npz(fullfile(base, 'cells', name));
expected = [speed lr sym weight];
stored = [entry.speed entry.learning_rate entry.symmetry_weight entry.fall_weight];
assert(all(abs(stored - expected) <= 1e-12 * abs(expected)), ...
    'Fall-penalty cell metadata differs: %s', name);
entry.energy_pct = double(entry.energy_pct);
end

function pct = energy_percent(raw)
% Smooth an energy signal and express it as percent change from its onset.
% The NaN-ignoring centered moving mean spans 181 samples in the interior and shrinks at the
% ends; the onset is the mean of the first 30 smoothed samples.
% Inputs:  raw - energy samples at 1 Hz (row vector)
% Outputs: pct - smoothed energy change (%) from onset
half_window = 90;
n = numel(raw);
finite = isfinite(raw);
filled = raw;
filled(~finite) = 0;
cs = [0, cumsum(filled)];
counts = [0, cumsum(finite)];
i = 0:n-1;
lo = max(0, i - half_window) + 1;
hi = min(n, i + half_window + 1) + 1;
k = counts(hi) - counts(lo);
smoothed = (cs(hi) - cs(lo)) ./ k;
smoothed(k == 0) = NaN;
onset_samples = smoothed(1:30)';
valid = ~isnan(onset_samples);
onset_samples(~valid) = 0;
onset = pairwise_sum(onset_samples) / nnz(valid);
assert(isfinite(onset) && abs(onset) >= 1e-6, 'Energy signal has an invalid smoothed onset');
pct = (smoothed - onset) / onset * 100;
end

function [average, count] = nan_mean_rows(x)
% Return the NaN-ignoring mean over rows, accumulated one row at a time.
% Inputs:  x - observations x samples
% Outputs: average - 1 x samples mean; count - number of non-NaN values per sample
total = zeros(1, size(x, 2));
count = zeros(1, size(x, 2));
for i = 1:size(x, 1)
    valid = ~isnan(x(i, :));
    values = x(i, :);
    values(~valid) = 0;
    total = total + values;
    count = count + valid;
end
average = total ./ count;
end

function deviation = nan_std_rows(x, average, count)
% Return the NaN-ignoring sample standard deviation (ddof = 1) over rows.
% Inputs:  x - observations x samples; average, count - outputs of nan_mean_rows(x)
% Outputs: deviation - 1 x samples standard deviation
total = zeros(1, size(x, 2));
for i = 1:size(x, 1)
    residual = x(i, :) - average;
    residual(isnan(x(i, :))) = 0;
    total = total + residual .* residual;
end
deviation = sqrt(total ./ (count - 1));
end

function total = pairwise_sum(x)
% Sum the columns of x in the pairwise order of numpy.sum, which differs from sequential
% summation in the last bits. Blocks of at most 128 values are summed with eight interleaved
% accumulators; longer columns are split in two at a multiple of eight.
n = size(x, 1);
if n < 8
    total = zeros(1, size(x, 2));
    for i = 1:n
        total = total + x(i, :);
    end
elseif n <= 128
    r = x(1:8, :);
    last = n - mod(n, 8);
    for i = 9:8:last
        r = r + x(i:i+7, :);
    end
    total = ((r(1, :) + r(2, :)) + (r(3, :) + r(4, :))) + ...
        ((r(5, :) + r(6, :)) + (r(7, :) + r(8, :)));
    for i = last+1:n
        total = total + x(i, :);
    end
else
    half = floor(n / 2);
    half = half - mod(half, 8);
    total = pairwise_sum(x(1:half, :)) + pairwise_sum(x(half+1:end, :));
end
end

function value = numpy_median(x)
% Return the median, averaging the two central values for an even count.
x = sort(x(:));
n = numel(x);
if mod(n, 2)
    value = x((n + 1) / 2);
else
    value = (x(n / 2) + x(n / 2 + 1)) / 2;
end
end

function value = rmse(first, second)
% Return the RMSE between the columns of first and the column second over finite samples.
difference = first - second;
finite = isfinite(difference);
difference(~finite) = 0;
value = sqrt(pairwise_sum(difference .^ 2) ./ sum(finite, 1));
end

function [error_value, chosen] = select_cell(curves, candidates, target)
% Return the smallest fit-to-fit RMSE among candidate cells and the first cell attaining it.
errors = rmse(curves(candidates, :)', target');
[error_value, index] = min(errors);
chosen = candidates(index);
end

function p = wilcoxon_p(d)
% Return the two-sided Wilcoxon signed-rank p-value of d, discarding zero differences.
% The exact null distribution is used when scipy.stats.wilcoxon (method 'auto') computes an
% exact or complete-enumeration p-value: n <= 13, or n <= 50 without zeros or tied magnitudes.
% Otherwise the normal approximation without continuity correction is used.
nonzero = d(d ~= 0);
tied = numel(unique(abs(nonzero))) < numel(nonzero);
if numel(d) <= 13 || (numel(d) <= 50 && ~tied && numel(nonzero) == numel(d))
    method = 'exact';
else
    method = 'approximate';
end
p = signrank(d, [], 'method', method);
end

function plot_condition(ax, minutes, average, sem, prediction, speed, weight, first)
% Draw one speed condition: experiment mean +/- SEM and the model C and E predictions.
% Inputs:  ax - target axes; minutes - time (min); average, sem - experiment mean and SEM (%)
%          prediction - model C (row 1) and model E (row 2) mean energy change (%)
%          speed - belt speed difference (m/s); weight - fitted fall weight of model C
%          first - true for the panel that carries the y-axis label
hold(ax, 'on');
fill(ax, [minutes, fliplr(minutes)], [average - sem, fliplr(average + sem)], ...
    [217 217 217] / 255, 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ax, minutes, average, 'Color', 'k', 'LineWidth', 1.8);
plot(ax, minutes, prediction(2, :), '-.', 'Color', [139 0 0] / 255, 'LineWidth', 1.7);
plot(ax, minutes, prediction(1, :), '-', 'Color', [0 0 205] / 255, 'LineWidth', 1.7);
hold(ax, 'off');
set(ax, 'XLim', [0 45], 'XTick', [0 15 30 45], 'FontSize', 9, 'Box', 'off', 'TickDir', 'out');
xlabel(ax, 'Time (min)');
title(ax, sprintf('%.1f m/s (fitted w = %g)', speed, weight), 'FontWeight', 'normal');
if first
    ylabel(ax, 'Energy change (%)');
end
end

function curve = fit_energy_exponential(values)
% Return the bounded least-squares fit of A + B exp(-t / tau) on the 2700-s grid.
% Only finite samples are fitted. The fit starts from (last value, first minus last value,
% 300 s) with bounds |A| <= 200, |B| <= 500 and 5 <= tau <= 8100 s.
% Inputs:  values - energy change (%) at 1 Hz, 2700 samples
% Outputs: curve  - fitted curve at t = 0, ..., 2699 s (row vector)
y = values(:);
assert(numel(y) == 2700, 'Exponential fitting requires 2700 one-Hz samples');
finite = isfinite(y);
assert(nnz(finite) >= 10, 'Exponential fitting requires at least ten finite samples');
t = (0:2699)';
tf = t(finite);
v = y(finite);
p = trf_least_squares(@(p) p(1) + p(2) * exp(-tf / p(3)) - v, ...
    [v(end); v(1) - v(end); 300], [-200; -500; 5], [200; 500; 8100], 10000);
curve = (p(1) + p(2) * exp(-t / p(3)))';
assert(all(isfinite(curve)), 'Exponential fitting produced nonfinite values');
end

% trf_least_squares and the helper functions below it are translated from SciPy
% (scipy/optimize/_lsq/trf.py and common.py) and are distributed under its license:
%
% Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions
% are met:
%
% 1. Redistributions of source code must retain the above copyright
%    notice, this list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above
%    copyright notice, this list of conditions and the following
%    disclaimer in the documentation and/or other materials provided
%    with the distribution.
%
% 3. Neither the name of the copyright holder nor the names of its
%    contributors may be used to endorse or promote products derived
%    from this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
% "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
% SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
% LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
% DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
% THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function x = trf_least_squares(fun, x0, lb, ub, max_nfev)
% Minimize 0.5 * ||fun(x)||^2 subject to lb <= x <= ub by the trust-region reflective method.
% The iteration (Branch, Coleman and Li, SIAM J Sci Comput 1999), the forward-difference
% Jacobian and the tolerances ftol = xtol = gtol = 1e-8 are those of
% scipy.optimize.least_squares (method 'trf', unit variable scaling, exact trust-region solver)
% as called by scipy.optimize.curve_fit. These tolerances stop the iteration short of the exact
% optimum, so the fitted curves depend on the iteration path and not only on the objective.
% Inputs:  fun      - residual vector as a function of the parameter column vector
%          x0       - initial parameters; lb, ub - finite bounds
%          max_nfev - maximum number of residual evaluations
% Outputs: x - parameters at termination
tol = 1e-8;
assert(all(x0 >= lb & x0 <= ub), 'Initial guess is outside of provided bounds');
x = strictly_feasible(x0, lb, ub, 1e-10);
f = fun(x);
assert(all(isfinite(f)), 'Residuals are not finite in the initial point');
nfev = 1;
J = forward_jacobian(fun, x, f, lb, ub);
[m, n] = size(J);
cost = 0.5 * (f' * f);
g = J' * f;
v = scaling_vector(x, g, lb, ub);
Delta = vnorm(x ./ sqrt(v));
if Delta == 0, Delta = 1; end
alpha = 0;
status = [];
while true
    [v, dv] = scaling_vector(x, g, lb, ub);
    g_norm = max(abs(g .* v));
    if g_norm < tol, status = 1; end
    if ~isempty(status) || nfev == max_nfev, break; end
    d = sqrt(v);
    diag_h = g .* dv;
    g_h = d .* g;
    J_h = J .* d';
    [U, s, V] = svd([J_h; diag(sqrt(diag_h))], 'econ');
    s = diag(s);
    uf = U' * [f; zeros(n, 1)];
    theta = max(0.995, 1 - g_norm);
    actual_reduction = -1;
    while actual_reduction <= 0 && nfev < max_nfev
        [p_h, alpha] = trust_region_step(n, m, uf, s, V, Delta, alpha);
        p = d .* p_h;
        [step, step_h, predicted_reduction] = select_step(x, J_h, diag_h, g_h, p, p_h, d, ...
            Delta, lb, ub, theta);
        x_new = strictly_feasible(x + step, lb, ub, 0);
        f_new = fun(x_new);
        nfev = nfev + 1;
        step_h_norm = vnorm(step_h);
        if ~all(isfinite(f_new))
            Delta = 0.25 * step_h_norm;
            continue
        end
        cost_new = 0.5 * (f_new' * f_new);
        actual_reduction = cost - cost_new;
        [Delta_new, ratio] = update_radius(Delta, actual_reduction, predicted_reduction, ...
            step_h_norm, step_h_norm > 0.95 * Delta);
        status = termination_status(actual_reduction, cost, vnorm(step), vnorm(x), ratio, tol);
        if ~isempty(status), break; end
        alpha = alpha * (Delta / Delta_new);
        Delta = Delta_new;
    end
    if actual_reduction > 0
        x = x_new;
        f = f_new;
        cost = cost_new;
        J = forward_jacobian(fun, x, f, lb, ub);
        g = J' * f;
    end
end
assert(~isempty(status), 'Optimal parameters not found: evaluation limit reached');
end

function value = vnorm(x)
% Return the Euclidean norm of a column vector.
value = sqrt(x' * x);
end

function J = forward_jacobian(fun, x, f0, lb, ub)
% Return the forward-difference Jacobian with relative step sqrt(eps), reversed or shortened
% where a step would leave the bounds.
h = sqrt(eps) * (2 * (x >= 0) - 1) .* max(1, abs(x));
lower_dist = x - lb;
upper_dist = ub - x;
violated = x + h < lb | x + h > ub;
fitting = abs(h) <= max(lower_dist, upper_dist);
h(violated & fitting) = -h(violated & fitting);
forward = upper_dist >= lower_dist & ~fitting;
h(forward) = upper_dist(forward);
backward = upper_dist < lower_dist & ~fitting;
h(backward) = -lower_dist(backward);
J = zeros(numel(f0), numel(x));
for i = 1:numel(x)
    x1 = x;
    x1(i) = x(i) + h(i);
    J(:, i) = (fun(x1) - f0) / (x1(i) - x(i));
end
end

function [v, dv] = scaling_vector(x, g, lb, ub)
% Return the Coleman-Li scaling vector and its derivative.
v = ones(size(x));
dv = zeros(size(x));
upper = g < 0 & isfinite(ub);
v(upper) = ub(upper) - x(upper);
dv(upper) = -1;
lower = g > 0 & isfinite(lb);
v(lower) = x(lower) - lb(lower);
dv(lower) = 1;
end

function x_new = strictly_feasible(x, lb, ub, rstep)
% Shift parameters on or within rstep of a bound into the interior of the feasible box.
% With rstep = 0 the shift is to the adjacent floating-point number.
if rstep == 0
    lower = x <= lb;
    upper = x >= ub;
else
    lower_dist = x - lb;
    upper_dist = ub - x;
    lower = lower_dist <= min(upper_dist, rstep * max(1, abs(lb)));
    upper = upper_dist <= min(lower_dist, rstep * max(1, abs(ub)));
end
lower = lower & ~upper;
x_new = x;
if rstep == 0
    for i = find(lower | upper)'
        if lower(i)
            x_new(i) = adjacent_double(lb(i), ub(i));
        else
            x_new(i) = adjacent_double(ub(i), lb(i));
        end
    end
else
    x_new(lower) = lb(lower) + rstep * max(1, abs(lb(lower)));
    x_new(upper) = ub(upper) - rstep * max(1, abs(ub(upper)));
end
tight = x_new < lb | x_new > ub;
x_new(tight) = 0.5 * (lb(tight) + ub(tight));
end

function y = adjacent_double(x, target)
% Return the double adjacent to x in the direction of target.
if x == 0
    y = sign(target) * realmin * eps;
    return
end
bits = typecast(x, 'int64');
if (target > x) == (x > 0)
    bits = bits + 1;
else
    bits = bits - 1;
end
y = typecast(bits, 'double');
end

function [p, alpha] = trust_region_step(n, m, uf, s, V, Delta, alpha)
% Solve the trust-region subproblem from the SVD of the augmented Jacobian.
% The Levenberg-Marquardt parameter is found by safeguarded Newton iterations
% (More, 1978; relative tolerance 0.01, at most 10 iterations).
% Inputs:  n, m  - numbers of parameters and residuals
%          uf    - U' * [f; 0]; s, V - singular values and right singular vectors
%          Delta - trust-region radius; alpha - initial Levenberg-Marquardt parameter
% Outputs: p - step in scaled variables; alpha - Levenberg-Marquardt parameter
suf = s .* uf;
full_rank = m >= n && s(end) > eps * m * s(1);
if full_rank
    p = -V * (uf ./ s);
    if vnorm(p) <= Delta
        alpha = 0;
        return
    end
end
alpha_upper = vnorm(suf) / Delta;
alpha_lower = 0;
if full_rank
    [phi, phi_prime] = secular(0, suf, s, Delta);
    alpha_lower = -phi / phi_prime;
end
if ~full_rank && alpha == 0
    alpha = max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper));
end
for it = 1:10
    if alpha < alpha_lower || alpha > alpha_upper
        alpha = max(0.001 * alpha_upper, sqrt(alpha_lower * alpha_upper));
    end
    [phi, phi_prime] = secular(alpha, suf, s, Delta);
    if phi < 0
        alpha_upper = alpha;
    end
    ratio = phi / phi_prime;
    alpha_lower = max(alpha_lower, alpha - ratio);
    alpha = alpha - (phi + Delta) * ratio / Delta;
    if abs(phi) < 0.01 * Delta
        break
    end
end
p = -V * (suf ./ (s .^ 2 + alpha));
p = p * (Delta / vnorm(p));
end

function [phi, phi_prime] = secular(alpha, suf, s, Delta)
% Return the regularized step norm minus Delta and its derivative with respect to alpha.
denom = s .^ 2 + alpha;
p_norm = vnorm(suf ./ denom);
phi = p_norm - Delta;
phi_prime = -sum(suf .^ 2 ./ denom .^ 3) / p_norm;
end

function [step, step_h, reduction] = select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, ...
    lb, ub, theta)
% Choose among the trust-region step, its reflection at the first bound hit and the scaled
% gradient step, keeping the iterate strictly feasible.
% Inputs:  x - current parameters; J_h, g_h, diag_h - scaled Jacobian, gradient and
%          Coleman-Li diagonal; p, p_h - trust-region step in original and scaled variables;
%          d - scaling vector; Delta - trust-region radius; lb, ub - bounds;
%          theta - fraction of the step to a bound that is taken
% Outputs: step, step_h - selected step in original and scaled variables
%          reduction    - predicted reduction of the quadratic model
if all(x + p >= lb & x + p <= ub)
    step = p;
    step_h = p_h;
    reduction = -quadratic_value(J_h, g_h, p_h, diag_h);
    return
end
[p_stride, hits] = step_to_bound(x, p, lb, ub);
r_h = p_h;
r_h(hits ~= 0) = -r_h(hits ~= 0);
r = d .* r_h;
p = p * p_stride;
p_h = p_h * p_stride;
[~, to_tr] = intersect_radius(p_h, r_h, Delta);
to_bound = step_to_bound(x + p, r, lb, ub);
r_stride = min(to_bound, to_tr);
if r_stride > 0
    r_stride_l = (1 - theta) * p_stride / r_stride;
    if r_stride == to_bound
        r_stride_u = theta * to_bound;
    else
        r_stride_u = to_tr;
    end
else
    r_stride_l = 0;
    r_stride_u = -1;
end
if r_stride_l <= r_stride_u
    [a, b, c] = quadratic_1d(J_h, g_h, r_h, diag_h, p_h);
    [r_stride, r_value] = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c);
    r_h = r_h * r_stride + p_h;
    r = r_h .* d;
else
    r_value = Inf;
end
p = p * theta;
p_h = p_h * theta;
p_value = quadratic_value(J_h, g_h, p_h, diag_h);
ag_h = -g_h;
ag = d .* ag_h;
to_tr = Delta / vnorm(ag_h);
to_bound = step_to_bound(x, ag, lb, ub);
if to_bound < to_tr
    ag_stride = theta * to_bound;
else
    ag_stride = to_tr;
end
[a, b] = quadratic_1d(J_h, g_h, ag_h, diag_h, []);
[ag_stride, ag_value] = minimize_quadratic_1d(a, b, 0, ag_stride, 0);
ag_h = ag_h * ag_stride;
ag = ag * ag_stride;
if p_value < r_value && p_value < ag_value
    step = p; step_h = p_h; reduction = -p_value;
elseif r_value < p_value && r_value < ag_value
    step = r; step_h = r_h; reduction = -r_value;
else
    step = ag; step_h = ag_h; reduction = -ag_value;
end
end

function [min_step, hits] = step_to_bound(x, s, lb, ub)
% Return the largest multiple of s that keeps x + t*s within bounds and the bounds hit.
nonzero = s ~= 0;
steps = Inf(size(x));
steps(nonzero) = max((lb(nonzero) - x(nonzero)) ./ s(nonzero), ...
    (ub(nonzero) - x(nonzero)) ./ s(nonzero));
min_step = min(steps);
hits = (steps == min_step) .* sign(s);
end

function [t1, t2] = intersect_radius(x, s, Delta)
% Return the two roots t1 <= t2 of ||x + t*s|| = Delta.
a = s' * s;
b = x' * s;
c = x' * x - Delta ^ 2;
d = sqrt(b * b - a * c);
if b < 0 || (b == 0 && 1 / b < 0)
    d = -d;
end
q = -(b + d);
t1 = q / a;
t2 = c / q;
if t1 >= t2
    [t1, t2] = deal(t2, t1);
end
end

function [a, b, c] = quadratic_1d(J, g, s, diag_h, s0)
% Return coefficients of the quadratic model along s0 + t*s as a*t^2 + b*t + c.
v = J * s;
a = v' * v + (s .* diag_h)' * s;
a = a * 0.5;
b = g' * s;
c = 0;
if ~isempty(s0)
    u = J * s0;
    b = b + u' * v;
    c = 0.5 * (u' * u) + g' * s0;
    b = b + (s0 .* diag_h)' * s;
    c = c + 0.5 * ((s0 .* diag_h)' * s0);
end
end

function [t_min, y_min] = minimize_quadratic_1d(a, b, lb, ub, c)
% Return the minimizer and minimum of a*t^2 + b*t + c on [lb, ub].
t = [lb, ub];
if a ~= 0
    extremum = -0.5 * b / a;
    if lb < extremum && extremum < ub
        t(end + 1) = extremum;
    end
end
y = t .* (a * t + b) + c;
[y_min, index] = min(y);
t_min = t(index);
end

function value = quadratic_value(J, g, s, diag_h)
% Return the quadratic model 0.5 * s'(J'J + diag)s + g's.
Js = J * s;
q = Js' * Js + (s .* diag_h)' * s;
value = 0.5 * q + s' * g;
end

function [Delta, ratio] = update_radius(Delta, actual, predicted, step_norm, bound_hit)
% Update the trust-region radius from the ratio of actual to predicted reduction.
if predicted > 0
    ratio = actual / predicted;
elseif predicted == actual && actual == 0
    ratio = 1;
else
    ratio = 0;
end
if ratio < 0.25
    Delta = 0.25 * step_norm;
elseif ratio > 0.75 && bound_hit
    Delta = Delta * 2;
end
end

function status = termination_status(dF, F, dx_norm, x_norm, ratio, tol)
% Return the termination code for the ftol (2), xtol (3) or both (4) criteria, or [] to continue.
ftol_satisfied = dF < tol * F && ratio > 0.25;
xtol_satisfied = dx_norm < tol * (tol + x_norm);
if ftol_satisfied && xtol_satisfied
    status = 4;
elseif ftol_satisfied
    status = 2;
elseif xtol_satisfied
    status = 3;
else
    status = [];
end
end

function text = number_text(x)
% Return the shortest decimal text, at most 17 significant digits, that reads back as x.
x = x(:);
text = compose("%.15g", x);
for digits = [16 17]
    inexact = str2double(text) ~= x;
    if ~any(inexact), break; end
    text(inexact) = compose("%." + digits + "g", x(inexact));
end
end

function write_csv(path, names, columns)
% Write a CSV file whose columns are numeric vectors or string arrays.
% Text containing commas or quotes is quoted; numbers are written at full precision.
cells = cell(1, numel(columns));
for k = 1:numel(columns)
    values = columns{k};
    if isnumeric(values) || islogical(values)
        cells{k} = number_text(double(values));
    else
        values = string(values(:));
        quoted = contains(values, [",", """", newline, char(13)]);
        values(quoted) = """" + replace(values(quoted), """", """""") + """";
        cells{k} = values;
    end
end
stream = fopen(path, 'w');
closer = onCleanup(@() fclose(stream));
fprintf(stream, '%s\n', [strjoin(string(names), ","); join([cells{:}], ",", 2)]);
end
