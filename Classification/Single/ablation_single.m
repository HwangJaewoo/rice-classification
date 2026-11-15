% ablation_single.m : Ablation of the components of the pipeline on the representative scenario
function S = ablation_single(year_str, seeds, conds, tag)

%% Settings
if nargin < 1, year_str = '2023'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, conds = {'full', 'noanova', 'gamma0', 'pca', 'aug'}; end
if nargin < 4, tag = ''; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
aug = struct('n_aug', 10, 'noise', 0.003, 'scale', 0.005, 'seed', 42);
rdir = out_dir(cfg, 'single');
cdir = out_dir(cfg, 'single', 'checkpoints');
run_log = start_run(cfg, 'single', sprintf('ablation_single_%s%s', year_str, tag));

%% Data
sc = scenario(cfg, cfg.rep_scenario);
ds = load_store(cfg, year_str, sc);
Fall = extract_features(ds, sc.stages);
fprintf('Ablation %s | %s | seeds %s | %s\n', year_str, sc.label, mat2str(seeds), strjoin(conds, ', '));

%% Jobs
K = numel(conds);
[jo, js, jk] = ndgrid(1:P, seeds, 1:K);
jo = jo(:); js = js(:); jk = jk(:);
J = numel(jo);
files = cell(J, 1);
for j = 1:J
    if strcmp(conds{jk(j)}, 'full')
        files{j} = fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, cfg.rep_scenario, js(j), jo(j)));
    else
        files{j} = fullfile(cdir, sprintf('checkpoint_abl_%s_%s_s%d_f%d.mat', year_str, conds{jk(j)}, js(j), jo(j)));
    end
end
done = cellfun(@(f) exist(f, 'file') == 2, files);
if any(done), fprintf('Resume: %d of %d jobs done\n', sum(done), J); end
D = progress('start', sum(~done) * cfg.n_bo);

%% Run
R = cell(J, 1);
t0 = tic;
parfor j = 1:J
    warning('off', 'all');
    if exist(files{j}, 'file') == 2
        R{j} = checkpoint('load', files{j});
    else
        if strcmp(conds{jk(j)}, 'full')
            Rj = single_fold(cfg, Fall, jo(j), js(j), D);
        else
            Rj = abl_fold(conds{jk(j)}, cfg, ds, Fall, sc.stages, aug, jo(j), js(j), D);
        end
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(K, ns, P); f1 = zeros(K, ns); inner = zeros(K, ns);
for k = 1:K
    for b = 1:ns
        yt = []; yp = [];
        for o = 1:P
            Rj = R{jk == k & js == seeds(b) & jo == o};
            acc(k, b, o) = Rj.acc; inner(k, b) = inner(k, b) + Rj.inner / P;
            yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
        end
        f1(k, b) = macro_f1(yt, yp, V);
    end
end
seed_mean = mean(acc, 3);
T = table(conds(:), mean(seed_mean, 2), std(seed_mean, 0, 2), mean(std(acc, 0, 3), 2), mean(f1, 2), mean(inner, 2), ...
          'VariableNames', {'Condition', 'Mean', 'SeedSD', 'FoldSD', 'MacroF1', 'InnerErr'});
disp(T);
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('year', year_str, 'seeds', seeds, 'conds', {conds}, 'acc', acc, 'f1', f1, 'inner', inner, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('ablation_single_%s%s', year_str, tag));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Summary');
SM = array2table(seed_mean, 'VariableNames', compose('Seed%d', seeds));
writetable([T(:, 1), SM], [base '.xlsx'], 'Sheet', 'Seeds');
end

%% Functions
function R = abl_fold(cond, cfg, ds, Fall, stages, aug, o, seed_r, D)
rng(o + 1000 * (seed_r - 1), 'twister');
V = cfg.V; P = cfg.P; NF = size(Fall, 2);
train_p = setdiff(1:P, o);
vars = optimizableVariable.empty;
if ~strcmp(cond, 'noanova'), vars(end+1) = optimizableVariable('num_feats', [1, NF], 'Type', 'integer', 'Transform', 'log'); end
if strcmp(cond, 'pca'), vars(end+1) = optimizableVariable('cve_target', [80, 99.9], 'Type', 'real'); end
if ~strcmp(cond, 'gamma0'), vars(end+1) = optimizableVariable('gamma', cfg.gamma_range, 'Type', 'real'); end
r = bayesopt(@inner_error, vars, 'MaxObjectiveEvaluations', cfg.n_bo, 'AcquisitionFunctionName', cfg.bo_acq, ...
             'IsObjectiveDeterministic', true, 'Verbose', 0, 'PlotFcn', []);
[tr_idx, tr_lab] = plant_idx(V, P, train_p);
[te_idx, te_lab] = plant_idx(V, P, o);
[acc, pred] = fit_eval(r.XAtMinObjective, tr_idx, tr_lab, te_idx, te_lab);
R = struct('acc', acc, 'pred', pred, 'te_lab', te_lab, 'best', r.XAtMinObjective, 'inner', r.MinObjective);

    function err = inner_error(x)
        accs = zeros(numel(train_p), 1);
        for j = 1:numel(train_p)
            [a_idx, a_lab] = plant_idx(V, P, setdiff(train_p, train_p(j)));
            [v_idx, v_lab] = plant_idx(V, P, train_p(j));
            accs(j) = fit_eval(x, a_idx, a_lab, v_idx, v_lab);
        end
        progress('tick', D, 1);
        err = 100 - mean(accs);
    end

    function [acc, pred] = fit_eval(opt, tr_idx, tr_lab, te_idx, te_lab)
        pred = zeros(size(te_lab)); acc = 0;
        try
            if strncmp(cond, 'aug', 3)
                ap = aug_params(cond, aug);
                rs = RandStream('mt19937ar', 'Seed', ap.seed);
                X = repmat(ds(tr_idx, :, :), ap.n_aug, 1, 1);
                n = size(X, 1);
                nz = ap.noise * std(X, 0, 2) .* randn(rs, size(X));
                s = (1 - ap.scale) + (2 * ap.scale) * rand(rs, n, 1, 1);
                Ftr = extract_features((X + nz) .* s, stages);
                ytr = repmat(tr_lab, ap.n_aug, 1);
            else
                Ftr = Fall(tr_idx, :); ytr = tr_lab;
            end
            if strcmp(cond, 'noanova')
                sel = 1:NF;
            else
                [~, ord] = sort(anova_f(Ftr, ytr, V), 'descend');
                sel = ord(1:min(opt.num_feats, NF));
            end
            mu = mean(Ftr(:, sel), 1); sg = std(Ftr(:, sel), 0, 1); sg(sg < eps) = 1;
            Xtr = (Ftr(:, sel) - mu) ./ sg;
            Xte = (Fall(te_idx, sel) - mu) ./ sg;
            if strcmp(cond, 'pca')
                [coeff, score, ~, ~, explained] = pca(Xtr, 'Economy', true);
                npc = find(cumsum(explained) >= opt.cve_target, 1);
                if isempty(npc), npc = numel(explained); end
                Xtr = score(:, 1:npc); Xte = Xte * coeff(:, 1:npc);
            end
            if strcmp(cond, 'gamma0'), g = 0; else, g = opt.gamma; end
            pred = predict(fitcdiscr(Xtr, ytr, 'DiscrimType', 'linear', 'Gamma', g, 'FillCoeffs', 'off'), Xte);
            acc = mean(pred == te_lab) * 100;
        catch ME
            if ~strcmp(cond, 'gamma0'), fprintf('!! %s fit failed: %s\n', cond, ME.message); end
        end
    end
end

function a = aug_params(cond, a)
switch cond
    case 'aug'
    case 'aug_noise0',   a.noise = 0;
    case 'aug_noise001', a.noise = 0.001;
    case 'aug_noise01',  a.noise = 0.01;
    case 'aug_scale0',   a.scale = 0;
    case 'aug_scale02',  a.scale = 0.02;
    case 'aug_n1',       a.n_aug = 1;
    case 'aug_n5',       a.n_aug = 5;
    case 'aug_n25',      a.n_aug = 25;
    otherwise, error('Unknown condition: %s', cond);
end
end
