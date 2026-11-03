function publication_s1_grid_sensitivity(grid_path, output_dir)
% Repeat the leave-one-subject-out comparison on progressively relaxed parameter grids.
% Inputs:  grid_path  - MAT file with rmse_matrix and exp_subj_num, as in publication_s1_loso
%          output_dir - folder for the output CSV files
% Outputs: comment1_grid_sensitivity_summary.csv and _persubject.csv in output_dir
rng(0);
R = 200;
S = load(grid_path, 'rmse_matrix', 'exp_subj_num');
rm = S.rmse_matrix; nsub = S.exp_subj_num;
[nC,~,nA,nP] = size(rm);
assert(isequal(nsub(:)', [8 9 9]), 'Expected 8/9/9 participants');
pairs = [];
for c=1:nC
    for s=1:nsub(c), pairs=[pairs; c s]; end %#ok<AGROW>
end
Np = size(pairs,1);
fc = zeros(nA,nP);
for i=1:Np
    fc = fc + isfinite(squeeze(rm(pairs(i,1),pairs(i,2),:,:)));
end
% A grid cell is admissible when at least thrs participants have a finite RMSE there.
thrs = [26 17 8];
summary = zeros(numel(thrs), 9);
subject_rows = zeros(numel(thrs)*Np, 5);
for t=1:numel(thrs)
    gbase = fc >= thrs(t);
    [csM, cszM, better, pv] = loso_relaxed(rm, nsub, pairs, gbase, R);
    summary(t,:) = [thrs(t), sum(gbase(:)), median(csM,'omitnan'), ...
        median(cszM,'omitnan'), median(cszM,'omitnan')/median(csM,'omitnan'), ...
        better, pv, R, 0];
    subject_rows((t-1)*Np+(1:Np),:) = [repmat(thrs(t),Np,1), pairs, csM, cszM];
    fprintf('Grid sensitivity: threshold=%d cells=%d medians=%.16g/%.16g ratio=%.16g p=%.16g\n', ...
        thrs(t),sum(gbase(:)),summary(t,3),summary(t,4),summary(t,5),pv);
end
if ~isfolder(output_dir), mkdir(output_dir); end
writetable(array2table(summary,'VariableNames', ...
    {'minimum_finite_subjects','admissible_cells','median_condition_specific', ...
     'median_common_size_matched','median_ratio_common_over_specific', ...
     'pct_subjects_better','wilcoxon_two_sided_matlab_signrank_default', ...
     'subsample_repeats','rng_seed'}), ...
    fullfile(output_dir,'comment1_grid_sensitivity_summary.csv'));
writetable(array2table(subject_rows,'VariableNames', ...
    {'minimum_finite_subjects','condition','subject','cond_specific','common_size_matched'}), ...
    fullfile(output_dir,'comment1_grid_sensitivity_persubject.csv'));
end

function [heldCS, heldCMs, better, pv] = loso_relaxed(rm, nsub, pairs, gbase, R)
% Return leave-one-subject-out RMSEs, fitting only admissible cells where the
% held-out participant's RMSE is finite.
% Inputs:  rm, nsub - RMSE grid and participants per condition
%          pairs    - (condition, participant) row for each participant
%          gbase    - admissible grid cells
%          R        - number of random size-matched shared pools per participant
% Outputs: heldCS, heldCMs - held-out RMSE of the condition-specific and size-matched
%                            shared fits
%          better         - percentage of participants with heldCS < heldCMs
%          pv             - two-sided signrank p-value
Np = size(pairs,1);
heldCS = nan(Np,1); heldCMs = nan(Np,1);
for i=1:Np
    c=pairs(i,1); s=pairs(i,2);
    gi = gbase & isfinite(squeeze(rm(c,s,:,:)));
    tr = pairs(pairs(:,1)==c & ~(pairs(:,2)==s), :);
    [bi,bj] = argmin_grid(meanmasked(rm, tr, gi));
    heldCS(i) = rm(c,s,bi,bj);
    K = nsub(c)-1; acc = nan(R,1);
    poolIdx = find(~(pairs(:,1)==c & pairs(:,2)==s));
    for r=1:R
        sel = pairs(poolIdx(randperm(numel(poolIdx), K)), :);
        [bi,bj] = argmin_grid(meanmasked(rm, sel, gi));
        acc(r) = rm(c,s,bi,bj);
    end
    heldCMs(i) = mean(acc,'omitnan');
end
better = mean(heldCS < heldCMs)*100;
pv = signrank(heldCS, heldCMs);
end

function M = meanmasked(rm, rows, gmask)
% Return the NaN-ignoring mean RMSE grid over (condition, participant) rows, NaN outside gmask.
[nA,nP] = size(gmask);
A = nan(size(rows,1), nA, nP);
for k=1:size(rows,1), A(k,:,:) = squeeze(rm(rows(k,1), rows(k,2), :, :)); end
M = squeeze(mean(A, 1, 'omitnan'));
M(~gmask) = NaN;
end

function [bi,bj] = argmin_grid(M)
% Return the row and column indices of the smallest value in M.
[~,idx] = min(M(:)); [bi,bj] = ind2sub(size(M), idx);
end
