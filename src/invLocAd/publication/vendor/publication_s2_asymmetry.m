function publication_s2_asymmetry(data_dir, output_dir)
% Recompute the Supplementary Figure 2 step length and step time asymmetry tables and figures.
% Inputs:  data_dir   - prepared publication data folder containing extensions/s2 with
%                       per_subject_temporal.csv, analysis_summary.json,
%                       group_trajectories.npz and model_temporal.npz
%          output_dir - folder for the output CSV and PNG files
% Outputs: s2_adapted.csv, s2_participants.csv, s2_sta_statistics.csv,
%          s2_symmetry_sweep.csv, s2_timeseries.csv, s2_adapted.png and
%          s2_timeseries.png in output_dir
speeds = [0.4 0.7 1.0];
folder = fullfile(data_dir, 'extensions', 's2');
names = {'per_subject_temporal.csv', 'analysis_summary.json', ...
    'group_trajectories.npz', 'model_temporal.npz'};
for k = 1:numel(names)
    if ~isfile(fullfile(folder, names{k}))
        error('publication_s2_asymmetry:MissingInput', ...
            'Required publication input is missing: extensions/s2/%s', names{k});
    end
end
% Participant summaries are kept as text so that s2_participants.csv reproduces the source.
options = detectImportOptions(fullfile(folder, names{1}), 'Delimiter', ',', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
options = setvartype(options, 'string');
participants = readtable(fullfile(folder, names{1}), options);
if height(participants) == 0
    error('publication_s2_asymmetry:EmptyInput', 'Empty source table: %s', names{1});
end
summary = jsondecode(fileread(fullfile(folder, names{2})));
group = read_npz(fullfile(folder, names{3}));
model = read_npz(fullfile(folder, names{4}));
if ~isfolder(output_dir), mkdir(output_dir); end
speed_of = str2double(participants.speed_diff_ms);

% Adapted asymmetry: participant mean with a Student-t 95% CI, against the model prediction.
metrics = {'sla', 'sla_adapted_fp', 'Step length asymmetry'; ...
    'sta', 'sta_adapted', 'Step time asymmetry'};
nrows = size(metrics, 1)*numel(speeds);
adapted = table(strings(nrows, 1), zeros(nrows, 1), zeros(nrows, 1), zeros(nrows, 1), ...
    zeros(nrows, 1), zeros(nrows, 1), zeros(nrows, 1), zeros(nrows, 1), zeros(nrows, 1), ...
    'VariableNames', {'metric', 'delta_mps', 'n', 'experimental_mean', ...
    'experimental_ci95_low', 'experimental_ci95_high', 'model_mean', ...
    'model_learning_rate', 'model_symmetry_weight'});
max_reference_error = 0;
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 6.5 5], 'Color', 'w');
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for m = 1:size(metrics, 1)
    metric = metrics{m, 1};
    [means, ci_low, ci_high, predictions] = deal(zeros(1, numel(speeds)));
    for s = 1:numel(speeds)
        values = str2double(participants.(metrics{m, 2})(speed_of == speeds(s)));
        if numel(values) < 2 || ~all(isfinite(values))
            error('publication_s2_asymmetry:MissingValues', ...
                'Supplementary Figure 2 %s: missing participant values at %.1f m/s', ...
                upper(metric), speeds(s));
        end
        n = numel(values);
        means(s) = mean(values);
        half = tinv(0.975, n - 1)*std(values)/sqrt(n);
        ci_low(s) = means(s) - half;
        ci_high(s) = means(s) + half;
        key = matlab.lang.makeValidName(sprintf('%.1f', speeds(s)));
        if strcmp(metric, 'sla')
            reference = summary.statistics.sla_by_condition.(key);
            saved = [reference.adapted_mean; reference.adapted_ci95(:)];
        else
            reference = summary.statistics.by_condition.(key);
            saved = [reference.sta_adapted_mean; reference.sta_adapted_ci95(:)];
        end
        deviation = max(abs([means(s); ci_low(s); ci_high(s)] - saved));
        max_reference_error = max(max_reference_error, deviation);
        if deviation > 1e-10
            error('publication_s2_asymmetry:Reference', ...
                'Supplementary Figure 2 statistics do not match the original analysis cache');
        end
        suffix = speed_suffix(speeds(s));
        predictions(s) = model.(sprintf('scalar_%s_adapted_%s', metric, suffix));
        row = (m - 1)*numel(speeds) + s;
        adapted.metric(row) = metric;
        adapted{row, 2:end} = [speeds(s), n, means(s), ci_low(s), ci_high(s), predictions(s), ...
            model.(['scalar_learning_rate_' suffix]), model.(['scalar_symmetry_weight_' suffix])];
    end
    lo = min([ci_low, predictions]);
    hi = max([ci_high, predictions]);
    pad = 0.12*(hi - lo);
    if pad == 0, pad = 0.01; end
    ax = nexttile(layout);
    hold(ax, 'on');
    errorbar(ax, speeds, means, means - ci_low, ci_high - means, 'o-', 'Color', 0.35*[1 1 1], ...
        'MarkerFaceColor', 0.35*[1 1 1], 'LineWidth', 1.6, 'CapSize', 6);
    style_adapted_axes(ax, speeds, metrics{m, 3}, 'Experimental data', [lo - pad, hi + pad]);
    ax = nexttile(layout);
    hold(ax, 'on');
    plot(ax, speeds, predictions, 'o--', 'Color', hex_color('#000099'), ...
        'MarkerFaceColor', hex_color('#000099'), 'LineWidth', 1.6);
    if strcmp(metric, 'sla')
        weights = [0 0.5];
        handles = gobjects(1, numel(weights));
        % Model sweeps at other symmetry weights, drawn at 30% opacity on a white background.
        faded = 0.3*hex_color('#000099') + 0.7;
        for w = 1:numel(weights)
            sweep = arrayfun(@(speed) summary.model_symmetry_sweep.( ...
                sweep_key(speed, weights(w))).sla_adapted, speeds);
            handles(w) = plot(ax, speeds, sweep, '--', 'Color', faded, 'LineWidth', 1.5);
        end
        legend(ax, handles, compose('Symmetry weight %g', weights), 'Box', 'off', ...
            'FontSize', 7, 'Location', 'northeast', 'AutoUpdate', 'off');
    end
    style_adapted_axes(ax, speeds, metrics{m, 3}, 'Model', [lo - pad, hi + pad]);
end
write_csv(adapted, output_dir, 's2_adapted');
write_csv(participants, output_dir, 's2_participants');
exportgraphics(fig, fullfile(output_dir, 's2_adapted.png'), 'Resolution', 300);
close(fig);

% Pooled STA tests: regression of adapted STA on speed difference and paired adapted - early.
x = speed_of;
adapted_sta = str2double(participants.sta_adapted);
early_sta = str2double(participants.sta_early);
n = numel(x);
[slope, slope_p] = linear_regression(x, adapted_sta);
change = adapted_sta - early_sta;
half = tinv(0.975, n - 1)*std(change)/sqrt(n);
paired_t = mean(change)/(std(change)/sqrt(n));
paired_p = 2*tcdf(abs(paired_t), n - 1, 'upper');
if ~is_close(slope_p, summary.statistics.sta_vs_speed.p)
    error('publication_s2_asymmetry:Reference', ...
        'Supplementary Figure 2 STA regression differs from the original source');
end
if ~is_close(paired_p, summary.statistics.sta_adapted_vs_early.p)
    error('publication_s2_asymmetry:Reference', ...
        'Supplementary Figure 2 STA adaptation test differs from the original source');
end
statistics = table(["adapted_STA_vs_speed"; "adapted_STA_minus_early_STA"], [n; n], ...
    ["linear regression"; "paired t test"], [slope; mean(change)], ...
    [NaN; mean(change) - half], [NaN; mean(change) + half], [slope_p; paired_p], ...
    'VariableNames', {'comparison', 'n', 'method', 'estimate', 'ci95_low', 'ci95_high', ...
    'p_two_sided'});
write_csv(statistics, output_dir, 's2_sta_statistics');

% Saved model sweep over the symmetry weight.
weights = [0 0.5 1];
[speed_grid, weight_grid] = meshgrid(speeds, weights);
sweep = arrayfun(@(speed, weight) summary.model_symmetry_sweep.(sweep_key(speed, weight)), ...
    speed_grid(:), weight_grid(:));
symmetry_sweep = [table(speed_grid(:), weight_grid(:), ...
    'VariableNames', {'delta_mps', 'symmetry_weight'}), struct2table(sweep)];
write_csv(symmetry_sweep, output_dir, 's2_symmetry_sweep');

% Binned time courses: experimental mean with a mean +/- 1.96 SEM band, and model mean.
colors = [hex_color('#2C7BB6'); hex_color('#FDAE61'); hex_color('#D7191C')];
sources = {'experiment', 'model'};
titles = {'Experimental data', 'Model'};
blocks = cell(size(metrics, 1)*numel(sources)*numel(speeds), 1);
fig = figure('Visible', 'off', 'Units', 'inches', 'Position', [0 0 7 5], 'Color', 'w');
layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
axes_grid = gobjects(size(metrics, 1), numel(sources));
for m = 1:size(metrics, 1)
    metric = metrics{m, 1};
    limits = zeros(1, 0);
    for c = 1:numel(sources)
        z = group;
        if c == 2, z = model; end
        edges = z.edges;
        minutes = (edges(1:end-1) + edges(2:end))/120;
        [series, sem] = deal(cell(1, numel(speeds)));
        for s = 1:numel(speeds)
            suffix = speed_suffix(speeds(s));
            if c == 1
                series{s} = z.(sprintf('%s_%s_mean', metric, suffix));
                sem{s} = z.(sprintf('%s_%s_sem', metric, suffix));
            else
                series{s} = z.(sprintf('%s_mean_%s', metric, suffix));
                sem{s} = NaN(size(minutes));
            end
            if ~isequal(size(series{s}), size(minutes)) || ~all(isfinite(series{s}))
                error('publication_s2_asymmetry:Trajectory', ...
                    'Supplementary Figure 2 trajectory cache has invalid dimensions or values');
            end
        end
        ax = nexttile(layout);
        axes_grid(m, c) = ax;
        hold(ax, 'on');
        % Bands are drawn before all lines so that no band covers a mean trajectory.
        if c == 1
            for s = 1:numel(speeds)
                band = [series{s} - 1.96*sem{s}, series{s} + 1.96*sem{s}];
                fill(ax, [minutes; flipud(minutes)], [band(:, 1); flipud(band(:, 2))], ...
                    colors(s, :), 'FaceAlpha', 0.18, 'EdgeColor', colors(s, :), ...
                    'EdgeAlpha', 0.18, 'LineWidth', 1);
                limits = [limits, min(band(:, 1)), max(band(:, 2))]; %#ok<AGROW>
            end
        else
            limits = [limits, cellfun(@min, series), cellfun(@max, series)]; %#ok<AGROW>
        end
        handles = gobjects(1, numel(speeds));
        style = '-';
        if c == 2, style = '--'; end
        for s = 1:numel(speeds)
            handles(s) = plot(ax, minutes, series{s}, style, 'Color', colors(s, :), ...
                'LineWidth', 1.5);
            bins = numel(minutes);
            blocks{((m - 1)*numel(sources) + c - 1)*numel(speeds) + s} = table( ...
                repmat(string(metric), bins, 1), repmat(string(sources{c}), bins, 1), ...
                repmat(speeds(s), bins, 1), edges(1:end-1), edges(2:end), minutes, ...
                series{s}, sem{s}, 'VariableNames', {'metric', 'source', 'delta_mps', ...
                'bin_start_s', 'bin_end_s', 'time_min', 'mean', 'sem'});
        end
        if m == 1 && c == 1
            legend(ax, handles, compose('%.1f m/s', speeds), 'Box', 'off', 'FontSize', 7, ...
                'Location', 'southeast', 'AutoUpdate', 'off');
        end
        style_axes(ax, 'Time (min)', metrics{m, 3}, titles{c});
        xlim(ax, padded_range(minutes, 0.05));
    end
    pad = 0.06*(max(limits) - min(limits));
    for c = 1:numel(sources)
        ylim(axes_grid(m, c), [min(limits) - pad, max(limits) + pad]);
        uniform_tick_decimals(axes_grid(m, c));
    end
end
write_csv(vertcat(blocks{:}), output_dir, 's2_timeseries');
exportgraphics(fig, fullfile(output_dir, 's2_timeseries.png'), 'Resolution', 300);
close(fig);
fprintf(['S2 completed: %d participants, maximum deviation of adapted statistics from ' ...
    'the analysis summary %.3g\n'], height(participants), max_reference_error);
end

function [slope, p] = linear_regression(x, y)
% Return the least-squares slope of y on x and its two-sided p-value.
% Inputs:  x, y  - paired observations (column vectors)
% Outputs: slope - least-squares slope
%          p     - two-sided p-value of the slope t test with n - 2 degrees of freedom
dx = x - mean(x);
dy = y - mean(y);
slope = sum(dx.*dy)/sum(dx.^2);
r = sum(dx.*dy)/sqrt(sum(dx.^2)*sum(dy.^2));
df = numel(x) - 2;
t = r*sqrt(df/((1 - r)*(1 + r)));
p = 2*tcdf(abs(t), df, 'upper');
end

function tf = is_close(value, reference)
% Return true when value agrees with reference to 1e-8 + 1e-10*|reference|; NaN never agrees.
tf = abs(value - reference) <= 1e-8 + 1e-10*abs(reference);
end

function suffix = speed_suffix(speed)
% Return the speed label used in the NumPy array names, e.g. 0_4 for 0.4 m/s.
suffix = strrep(sprintf('%.1f', speed), '.', '_');
end

function key = sweep_key(speed, weight)
% Return the decoded JSON field name of one model symmetry-sweep entry.
key = matlab.lang.makeValidName(sprintf('speed%.1f_psi%.1f', speed, weight));
end

function rgb = hex_color(hex)
% Return the RGB triplet in [0, 1] of a #RRGGBB color.
rgb = reshape(hex2dec(reshape(hex(2:7), 2, 3)'), 1, 3)/255;
end

function limits = padded_range(values, margin)
% Return the data range of values widened by margin times its width on each side.
span = max(values) - min(values);
limits = [min(values) - margin*span, max(values) + margin*span];
end

function style_axes(ax, x_label, y_label, title_text)
% Apply the shared axis style, labels and title.
set(ax, 'Box', 'off', 'TickDir', 'out', 'FontSize', 9, 'LineWidth', 0.8, ...
    'XColor', 'k', 'YColor', 'k', 'TitleFontWeight', 'normal', ...
    'TitleFontSizeMultiplier', 1.2, 'LabelFontSizeMultiplier', 1);
xlabel(ax, x_label);
ylabel(ax, y_label);
title(ax, title_text);
end

function style_adapted_axes(ax, speeds, y_label, title_text, y_limits)
% Style one adapted-asymmetry panel with a zero reference line and speed ticks.
yline(ax, 0, 'Color', 0.8*[1 1 1], 'LineWidth', 0.6, 'Layer', 'bottom');
style_axes(ax, 'Speed difference (m/s)', y_label, title_text);
set(ax, 'XTick', speeds, 'XTickLabel', compose('%.1f', speeds));
xlim(ax, padded_range(speeds, 0.05));
ylim(ax, y_limits);
uniform_tick_decimals(ax);
end

function uniform_tick_decimals(ax)
% Label all y ticks with the number of decimals set by the tick spacing (0.00, 0.02, ...).
decimals = max(0, ceil(-log10(min(diff(ax.YTick))) - 1e-9));
ax.YAxis.TickLabelFormat = sprintf('%%.%df', decimals);
end

function write_csv(T, output_dir, name)
% Write table T to output_dir/name.csv with numbers in shortest round-trip decimal text.
for k = 1:width(T)
    if isnumeric(T.(k)), T.(k) = number_text(T.(k)); end
end
writetable(T, fullfile(output_dir, [name '.csv']), 'QuoteStrings', 'minimal');
end

function text = number_text(values)
% Return the shortest %.15g to %.17g text that reads back to each value; NaN gives "".
text = strings(size(values));
for k = 1:numel(values)
    if isnan(values(k)), continue; end
    for digits = 15:17
        text(k) = sprintf('%.*g', digits, values(k));
        if str2double(text(k)) == values(k), break; end
    end
end
end
