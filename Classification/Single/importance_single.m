% importance_single.m : Feature importance, univariate analysis and confused cultivar pairs (representative scenario)
function S = importance_single(year_str, seeds)

%% Settings
if nargin < 1, year_str = '2023'; end
if nargin < 2, seeds = 1:5; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
n_rep = 20;
rdir = out_dir(cfg, 'single');
cdir = out_dir(cfg, 'single', 'checkpoints');
run_log = start_run(cfg, 'single', sprintf('importance_single_%s', year_str));

%% Data and feature map
sc = scenario(cfg, cfg.rep_scenario);
[ds, ch_names] = load_store(cfg, year_str, sc);
Fall = extract_features(ds, sc.stages);
NF = size(Fall, 2);
n_ang = numel(sc.angles);
cpa = numel(ch_names) / n_ang;
[col_ch, col_st, col_stat, stat_names] = feature_map(numel(sc.stages), numel(ch_names));
col_ang = ceil(col_ch / cpa);
col_idx = mod(col_ch - 1, cpa) + 1;
idx_names = regexprep(ch_names(1:cpa), '_0$', '');
ang_names = {'0', '120', '240'};
st_lab = @(c) sprintf('stage %d', col_st(c));
st_dif = @(c) sprintf('stage %d-%d', col_st(c) - 1, col_st(c));
feat_name = cell(NF, 1);
for c = 1:NF
    if col_stat(c) == 7, sl = st_dif(c); else, sl = st_lab(c); end
    feat_name{c} = sprintf('%s_%s | %s | %s', idx_names{col_idx(c)}, ang_names{col_ang(c)}, sl, stat_names{col_stat(c)});
end
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
    f = fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, cfg.rep_scenario, js(j), jo(j)));
    if exist(f, 'file') ~= 2, error('Run nested_single first (missing %s)', f); end
    R0 = checkpoint('load', f);
    best{j} = R0.best;
end
fprintf('Importance %s | %s | seeds %s | %d refits\n', year_str, sc.label, mat2str(seeds), J);
D = progress('start', J);

%% Refit and permutation importance
acc_j = zeros(J, 1); drop_j = zeros(J, G); pred_j = cell(J, 1); top_j = zeros(J, 50);
parfor j = 1:J
    [tr_idx, tr_lab] = plant_idx(V, P, setdiff(1:P, jo(j)));
    [te_idx, te_lab] = plant_idx(V, P, jo(j));
    M = pipe_fit(Fall(tr_idx, :), tr_lab, best{j}, V);
    Fte = Fall(te_idx, :);
    pred = pipe_predict(M, Fte);
    acc0 = mean(pred == te_lab) * 100;
    acc_j(j) = acc0;
    pred_j{j} = pred;
    [~, ord] = sort(anova_f(Fall(tr_idx, :), tr_lab, V), 'descend');
    top_j(j, :) = ord(1:50);
    rs = RandStream('mt19937ar', 'Seed', 7 + jo(j) + 100 * js(j));
    dj = zeros(1, G);
    for g = 1:G
        a = zeros(n_rep, 1);
        for r = 1:n_rep
            Fp = Fte;
            Fp(:, grp(g).cols) = Fte(randperm(rs, V), grp(g).cols);
            a(r) = mean(pipe_predict(M, Fp) == te_lab) * 100;
        end
        dj(g) = acc0 - mean(a);
    end
    drop_j(j, :) = dj;
    progress('tick', D, 1);
end

%% Permutation summary
T_perm = table({grp.type}', {grp.name}', mean(drop_j, 1)', std(drop_j, 0, 1)', ...
               'VariableNames', {'Group', 'Name', 'AccDrop', 'AccDropSD'});
fprintf('\nRefit accuracy: %.2f (should equal nested_single)\n', mean(acc_j));
for tt = {'index', 'stage', 'angle', 'stat'}
    fprintf('\nPermutation importance by %s (accuracy drop, pt)\n', tt{1});
    disp(sortrows(T_perm(strcmp(T_perm.Group, tt{1}), 2:4), 'AccDrop', 'descend'));
end
tf = top_j(arrayfun(@(o) find(jo == o, 1), 1:P), :);
ov = zeros(P);
for a = 1:P, for b = 1:P, ov(a, b) = numel(intersect(tf(a, :), tf(b, :))) / 50; end, end
ov_mean = mean(ov(~eye(P)));
fprintf('Top-50 ANOVA overlap between the %d training sets: %.2f\n', P, ov_mean);

%% Univariate ANOVA (4 independent plants, descriptive)
y_all = repelem((1:V)', P);
[Fu, pu] = anova_f(Fall, y_all, V);
m = numel(pu); [ps, o] = sort(pu(:));
qs = min(1, flipud(cummin(flipud(ps .* m ./ (1:m)'))));
q = zeros(1, m); q(o) = qs;
[~, ou] = sort(Fu, 'descend');
T_uni = table(feat_name(ou(1:20)), Fu(ou(1:20))', pu(ou(1:20))', q(ou(1:20))', 'VariableNames', {'Feature', 'F', 'p', 'q_BH'});
fprintf('\nUnivariate ANOVA: %d of %d features with q < 0.05\n', sum(q < 0.05), NF);
disp(T_uni);
med_idx = arrayfun(@(i) median(Fu(col_idx == i)), (1:cpa)');
T_uni_idx = sortrows(table(idx_names', med_idx, 'VariableNames', {'Index', 'MedianF'}), 'MedianF', 'descend');

%% Confusion
Cm = zeros(V);
for j = 1:J
    pr = pred_j{j};
    for c = 1:V, Cm(c, pr(c)) = Cm(c, pr(c)) + 1; end
end
Cs = triu(Cm + Cm', 1);
[cv, ci] = sort(Cs(:), 'descend');
[pa_i, pb_i] = ind2sub([V V], ci(1:10));
n_test = numel(seeds) * P;
T_conf = table(pa_i, pb_i, Cm(sub2ind([V V], pa_i, pb_i)), Cm(sub2ind([V V], pb_i, pa_i)), cv(1:10), cv(1:10) / (2 * n_test), ...
               'VariableNames', {'CultivarA', 'CultivarB', 'A_as_B', 'B_as_A', 'Total', 'Rate'});
err_cv = 1 - diag(Cm) / n_test;
fprintf('\nMost confused pairs (per cultivar: %d plants x %d BO seeds = %d predictions)\n', P, numel(seeds), n_test);
disp(T_conf);
fprintf('Never misclassified: %d of %d | error rate > 50%%: %d\n', sum(err_cv == 0), V, sum(err_cv > 0.5));

%% Example pair
ca = pa_i(1); cb = pb_i(1);
ra = (ca - 1) * P + (1:P); rb = (cb - 1) * P + (1:P);
xa = Fall(ra, :); xb = Fall(rb, :);
diff_ab = mean(xa, 1) - mean(xb, 1);
d = diff_ab ./ (sqrt((var(xa, 0, 1) + var(xb, 0, 1)) / 2) + eps);
[~, od] = sort(abs(d), 'descend');
T_pair = table(feat_name(od(1:10)), mean(xa(:, od(1:10)), 1)', mean(xb(:, od(1:10)), 1)', d(od(1:10))', ...
               'VariableNames', {'Feature', sprintf('Mean_%d', ca), sprintf('Mean_%d', cb), 'CohenD'});
fprintf('\nExample pair %d vs %d (4 vs 4 plants)\n', ca, cb);
disp(T_pair);

%% Figures
fdir = out_dir(cfg, 'single', 'figures');
Si = sortrows(T_perm(strcmp(T_perm.Group, 'index'), :), 'AccDrop', 'ascend');
fig = figure('Visible', 'off', 'Position', [100 100 700 650]);
barh(Si.AccDrop); hold on;
errorbar(Si.AccDrop, (1:height(Si))', Si.AccDropSD, 'horizontal', 'k.', 'LineStyle', 'none');
set(gca, 'YTick', 1:height(Si), 'YTickLabel', Si.Name, 'TickLabelInterpreter', 'none');
xlabel('Accuracy drop when permuted (pt)'); title(sprintf('%s permutation importance by index', year_str));
exportgraphics(fig, fullfile(fdir, sprintf('importance_index_%s.png', year_str)), 'Resolution', 200); close(fig);
ch1 = col_ch(od(1)); days = [sc.stages{:}];
fig = figure('Visible', 'off', 'Position', [100 100 700 400]); hold on;
for r = ra, plot(days, ds(r, :, ch1), '-', 'Color', [0.85 0.33 0.10]); end
for r = rb, plot(days, ds(r, :, ch1), '-', 'Color', [0 0.45 0.74]); end
xline(25.5, ':'); xline(39.5, ':');
xlabel('Time index'); ylabel(ch_names{ch1}, 'Interpreter', 'none');
title(sprintf('%s cultivar %d (orange) vs %d (blue)', year_str, ca, cb));
exportgraphics(fig, fullfile(fdir, sprintf('pair_%s_%d_%d.png', year_str, ca, cb)), 'Resolution', 200); close(fig);

%% Save
S = struct('year', year_str, 'seeds', seeds, 'T_perm', T_perm, 'T_uni', T_uni, 'T_uni_idx', T_uni_idx, ...
           'T_conf', T_conf, 'T_pair', T_pair, 'Cm', Cm, 'err_cv', err_cv, 'Fu', Fu, 'pu', pu, 'q', q, ...
           'feat_name', {feat_name}, 'drop_j', drop_j, 'acc_j', acc_j, 'ov_mean', ov_mean);
base = fullfile(rdir, sprintf('importance_single_%s', year_str));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T_perm, [base '.xlsx'], 'Sheet', 'Permutation');
writetable(T_uni, [base '.xlsx'], 'Sheet', 'Univariate_top20');
writetable(T_uni_idx, [base '.xlsx'], 'Sheet', 'Univariate_by_index');
writetable(T_conf, [base '.xlsx'], 'Sheet', 'Confused_pairs');
writetable(T_pair, [base '.xlsx'], 'Sheet', 'Pair_example');
end
