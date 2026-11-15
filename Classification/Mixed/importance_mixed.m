% importance_mixed.m : Permutation importance of the representative scenario in Mixed-year validation
function S = importance_mixed(seeds, comb)

%% Settings
if nargin < 1, seeds = 1:5; end
if nargin < 2, comb = 'geo'; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
n_rep = 20;
rdir = out_dir(cfg, 'mixed');
cdir = out_dir(cfg, 'mixed', 'checkpoints');
run_log = start_run(cfg, 'mixed', sprintf('importance_mixed_%s', comb));

%% Data and feature map
sc = scenario(cfg, cfg.rep_scenario);
[ds23, ch_names] = load_store(cfg, '2023', sc);
F23 = extract_features(ds23, sc.stages);
F24 = extract_features(load_store(cfg, '2024', sc), sc.stages);
n_ang = numel(sc.angles);
cpa = numel(ch_names) / n_ang;
[col_ch, col_st, col_stat, stat_names] = feature_map(numel(sc.stages), numel(ch_names));
col_ang = ceil(col_ch / cpa);
col_idx = mod(col_ch - 1, cpa) + 1;
idx_names = regexprep(ch_names(1:cpa), '_0$', '');
ang_names = {'0', '120', '240'};
grp = struct('type', {}, 'name', {}, 'cols', {});
for i = 1:cpa, grp(end+1) = struct('type', 'index', 'name', idx_names{i}, 'cols', find(col_idx == i)); end
for p = 1:numel(sc.stages), grp(end+1) = struct('type', 'stage', 'name', sprintf('stage %d', p), 'cols', find(col_st == p)); end
for a = 1:n_ang, grp(end+1) = struct('type', 'angle', 'name', ang_names{a}, 'cols', find(col_ang == a)); end
for s = 1:numel(stat_names), grp(end+1) = struct('type', 'stat', 'name', stat_names{s}, 'cols', find(col_stat == s)); end
G = numel(grp);

%% Hyperparameters from the nested runs
[jo, js] = ndgrid(1:P, seeds);
jo = jo(:); js = js(:);
J = numel(jo);
best = cell(J, 1);
for j = 1:J
    f = fullfile(cdir, sprintf('checkpoint_mixed_%s_k%d_s%d_f%d.mat', comb, cfg.rep_scenario, js(j), jo(j)));
    if exist(f, 'file') ~= 2, error('Run nested_mixed first (missing %s)', f); end
    R0 = checkpoint('load', f);
    best{j} = R0.best;
end
fprintf('Importance Mixed (%s) | %s | seeds %s | %d refits\n', comb, sc.label, mat2str(seeds), J);
D = progress('start', J);

%% Refit and permutation importance (rows permuted within each year of the test set)
acc_j = zeros(J, 1); drop_j = zeros(J, G);
parfor j = 1:J
    [M, Xte, te_lab] = fit_mixed(best{j}, V, P, F23, F24, comb, setdiff(1:P, jo(j)), jo(j));
    acc0 = mean(predict(M.lda, Xte(:, M.sel)) == te_lab) * 100;
    acc_j(j) = acc0;
    rs = RandStream('mt19937ar', 'Seed', 7 + jo(j) + 100 * js(j));
    dj = zeros(1, G);
    for g = 1:G
        a = zeros(n_rep, 1);
        for r = 1:n_rep
            Xp = Xte;
            c = grp(g).cols;
            Xp(1:V, c) = Xte(randperm(rs, V), c);
            Xp(V+1:end, c) = Xte(V + randperm(rs, V), c);
            a(r) = mean(predict(M.lda, Xp(:, M.sel)) == te_lab) * 100;
        end
        dj(g) = acc0 - mean(a);
    end
    drop_j(j, :) = dj;
    progress('tick', D, 1);
end

%% Summary
T_perm = table({grp.type}', {grp.name}', mean(drop_j, 1)', std(drop_j, 0, 1)', ...
               'VariableNames', {'Group', 'Name', 'AccDrop', 'AccDropSD'});
fprintf('\nRefit accuracy: %.2f (should equal nested_mixed %s)\n', mean(acc_j), comb);
for tt = {'index', 'stage', 'angle', 'stat'}
    fprintf('\nPermutation importance by %s (accuracy drop, pt)\n', tt{1});
    disp(sortrows(T_perm(strcmp(T_perm.Group, tt{1}), 2:4), 'AccDrop', 'descend'));
end

%% Save
S = struct('comb', comb, 'seeds', seeds, 'T_perm', T_perm, 'drop_j', drop_j, 'acc_j', acc_j);
base = fullfile(rdir, sprintf('importance_mixed_%s', comb));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T_perm, [base '.xlsx'], 'Sheet', 'Permutation');
end

%% Helper: final model of one outer fold (same steps as fit_eval in mixed_fold.m)
function [M, Xte, te_lab] = fit_mixed(opt, V, P, F23, F24, comb, fit_p, test_p)
[tr_idx, tr_lab] = plant_idx(V, P, fit_p);
[te_idx, te_lab1] = plant_idx(V, P, test_p);
te_lab = [te_lab1; te_lab1];
A = F23(tr_idx, :); B = F24(tr_idx, :);
mA = mean(A, 1); sA = std(A, 0, 1); sA(sA < eps) = 1;
mB = mean(B, 1); sB = std(B, 0, 1); sB(sB < eps) = 1;
ZA = (A - mA) ./ sA; ZB = (B - mB) ./ sB;
switch comb
    case 'geo',    fs = sqrt(anova_f(A, tr_lab, V) .* anova_f(B, tr_lab, V));
    case 'arith',  fs = (anova_f(A, tr_lab, V) + anova_f(B, tr_lab, V)) / 2;
    case 'min',    fs = min(anova_f(A, tr_lab, V), anova_f(B, tr_lab, V));
    case 'pooled', fs = anova_f([ZA; ZB], [tr_lab; tr_lab], V);
end
[~, ord] = sort(fs, 'descend');
sel = ord(1:min(opt.num_feats, numel(fs)));
lda = fitcdiscr([ZA(:, sel); ZB(:, sel)], [tr_lab; tr_lab], 'DiscrimType', 'linear', ...
                'Gamma', opt.gamma, 'FillCoeffs', 'off');
Xte = [(F23(te_idx, :) - mA) ./ sA; (F24(te_idx, :) - mB) ./ sB];
M = struct('sel', sel, 'lda', lda);
end
