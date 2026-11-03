function publication_s1_loso(grid_path, output_dir)
% Compare condition-specific and shared fits using leave-one-subject-out validation.
% Inputs:  grid_path  - MAT file with rmse_matrix (energy RMSE, condition x
%                       participant x learning_pool x sym_cost_pool) and
%                       exp_subj_num (participants per condition)
%          output_dir - folder for the output CSV files
% Outputs: comment1_loso_cv_stats.csv, _test.csv and _persubject.csv in output_dir
rng(0);
S = load(grid_path, 'rmse_matrix', 'exp_subj_num');
rm = S.rmse_matrix;
nsub = S.exp_subj_num;
[nC,~,nA,nP] = size(rm);
assert(isequal(nsub(:)', [8 9 9]), 'Expected 8/9/9 participants');
R = 200;
pairs = [];
for c=1:nC
    for s=1:nsub(c)
        pairs=[pairs; c s]; %#ok<AGROW>
    end
end
Np = size(pairs,1);
% Fit only on grid cells that are finite for every participant.
gmask = true(nA,nP);
for i=1:Np
    g = squeeze(rm(pairs(i,1),pairs(i,2),:,:));
    gmask = gmask & isfinite(g);
end
assert(any(gmask(:)), 'No grid cell is finite for every participant');
bestcell = @(rows) argmin_grid(meanmasked(rm, rows, gmask));
% Held-out RMSE of the condition-specific fit (heldCS), the shared fit on all other
% participants (heldCM) and the shared fit averaged over R random pools of the
% condition-specific training size (heldCMs).
heldCS = nan(Np,1); heldCM = nan(Np,1); heldCMs = nan(Np,1);
for i=1:Np
    c=pairs(i,1); s=pairs(i,2);
    tr = pairs(pairs(:,1)==c & ~(pairs(:,1)==c & pairs(:,2)==s), :);
    [bi,bj] = bestcell(tr);
    heldCS(i) = rm(c,s,bi,bj);
    tr = pairs(~(pairs(:,1)==c & pairs(:,2)==s), :);
    [bi,bj] = bestcell(tr);
    heldCM(i) = rm(c,s,bi,bj);
    K = nsub(c)-1; acc = nan(R,1);
    poolIdx = find(~(pairs(:,1)==c & pairs(:,2)==s));
    for r=1:R
        sel = pairs(poolIdx(randperm(numel(poolIdx), K)), :);
        [bi,bj] = bestcell(sel);
        acc(r) = rm(c,s,bi,bj);
    end
    heldCMs(i) = mean(acc,'omitnan');
end
pv2 = signrank(heldCS, heldCMs);
better = mean(heldCS < heldCMs)*100;
d = heldCMs - heldCS; d = d(isfinite(d) & d~=0);
r = tiedrank(abs(d)); Rpos = sum(r(d>0)); Rneg = sum(r(d<0));
rbc = (Rpos - Rneg) / sum(r);
if ~isfolder(output_dir), mkdir(output_dir); end
out = fullfile(output_dir, 'comment1_loso_cv');
models = {'condition_specific';'common_full_pool';'common_size_matched'};
Tstat = table(models, ...
    [mean(heldCS,'omitnan');mean(heldCM,'omitnan');mean(heldCMs,'omitnan')], ...
    [median(heldCS,'omitnan');median(heldCM,'omitnan');median(heldCMs,'omitnan')], ...
    'VariableNames', {'model','mean_heldout_RMSE','median_heldout_RMSE'});
writetable(Tstat, [out '_stats.csv']);
Ttest = table({'avg'}, pv2, rbc, better, R, 0, ...
    'VariableNames', {'grid','wilcoxon_p_condspec_vs_commonsize','rank_biserial', ...
    'pct_subj_condspec_better','subsample_repeats','rng_seed'});
writetable(Ttest, [out '_test.csv']);
Tsubj = table(pairs(:,1), pairs(:,2), heldCS, heldCM, heldCMs, ...
    'VariableNames', {'condition','subject','cond_specific','common_full','common_size_matched'});
writetable(Tsubj, [out '_persubject.csv']);
fprintf('S1 completed: %d participants, %d/%d finite cells, signrank p=%.16g\n', ...
    Np, sum(gmask(:)), nA*nP, pv2);
end

function M = meanmasked(rm, rows, gmask)
% Return the mean RMSE grid over (condition, participant) rows, NaN outside gmask.
[nA,nP] = size(gmask);
acc = zeros(nA,nP);
for k=1:size(rows,1)
    acc = acc + squeeze(rm(rows(k,1),rows(k,2),:,:));
end
M = acc/size(rows,1);
M(~gmask) = NaN;
end

function [bi,bj] = argmin_grid(M)
% Return the row and column indices of the smallest value in M.
[~,idx] = min(M(:));
[bi,bj] = ind2sub(size(M), idx);
end
