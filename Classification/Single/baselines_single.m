% baselines_single.m : Classifier comparison under the same nested splits and preprocessing (ANOVA -> standardization)
function S = baselines_single(year_str, seeds, clfs)

%% Settings
if nargin < 1, year_str = '2023'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, clfs = {'LDA', 'MNR', 'LOGIT', 'NC', 'SVM', 'RF', 'PLSDA'}; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
all_clfs = {'LDA', 'MNR', 'LOGIT', 'NC', 'SVM', 'RF', 'PLSDA'};
n_bo_rf = 30; n_trees = 100;
rdir = out_dir(cfg, 'single');
cdir = out_dir(cfg, 'single', 'checkpoints');
run_log = start_run(cfg, 'single', sprintf('baselines_single_%s', year_str));

%% Data
sc = scenario(cfg, cfg.rep_scenario);
Fall = extract_features(load_store(cfg, year_str, sc), sc.stages);
NF = size(Fall, 2);
fprintf('Baselines %s | %s | seeds %s | %s\n', year_str, sc.label, mat2str(seeds), strjoin(clfs, ', '));

%% Jobs
K = numel(clfs);
[jo, js, jk] = ndgrid(1:P, seeds, 1:K);
jo = jo(:); js = js(:); jk = jk(:);
J = numel(jo);
files = cell(J, 1); nb = zeros(J, 1);
for j = 1:J
    if strcmp(clfs{jk(j)}, 'LDA')
        files{j} = fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, cfg.rep_scenario, js(j), jo(j)));
    else
        files{j} = fullfile(cdir, sprintf('checkpoint_base_%s_%s_s%d_f%d.mat', year_str, clfs{jk(j)}, js(j), jo(j)));
    end
    nb(j) = cfg.n_bo; if strcmp(clfs{jk(j)}, 'RF'), nb(j) = n_bo_rf; end
end
done = cellfun(@(f) exist(f, 'file') == 2, files);
if any(done), fprintf('Resume: %d of %d jobs done\n', sum(done), J); end
D = progress('start', sum(nb(~done)));

%% Run
R = cell(J, 1);
t0 = tic;
parfor j = 1:J
    warning('off', 'all');
    if exist(files{j}, 'file') == 2
        R{j} = checkpoint('load', files{j});
    else
        name = clfs{jk(j)};
        if strcmp(name, 'LDA')
            Rj = single_fold(cfg, Fall, jo(j), js(j), D);
        else
            Rj = base_fold(name, find(strcmp(all_clfs, name)), cfg, Fall, NF, jo(j), js(j), nb(j), n_trees, D);
        end
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(K, ns, P); f1 = zeros(K, ns); nfail = zeros(K, 1);
for k = 1:K
    for b = 1:ns
        yt = []; yp = [];
        for o = 1:P
            Rj = R{jk == k & js == seeds(b) & jo == o};
            acc(k, b, o) = Rj.acc;
            yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
            if isfield(Rj, 'nfail'), nfail(k) = nfail(k) + Rj.nfail; end
        end
        f1(k, b) = macro_f1(yt, yp, V);
    end
end
seed_mean = mean(acc, 3);
T = table(clfs(:), mean(seed_mean, 2), std(seed_mean, 0, 2), mean(std(acc, 0, 3), 2), mean(f1, 2), nfail, ...
          'VariableNames', {'Classifier', 'Mean', 'SeedSD', 'FoldSD', 'MacroF1', 'InnerFailures'});
T = sortrows(T, 'Mean', 'descend');
disp(T);
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('year', year_str, 'seeds', seeds, 'clfs', {clfs}, 'acc', acc, 'f1', f1, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('baselines_single_%s', year_str));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Summary');
FA = table(repelem(clfs(:), ns), repmat(seeds(:), K, 1), reshape(permute(acc, [2 1 3]), [], P), ...
           'VariableNames', {'Classifier', 'Seed', 'Folds'});
writetable(splitvars(FA), [base '.xlsx'], 'Sheet', 'Folds');
end

%% Functions
function R = base_fold(name, cid, cfg, Fall, NF, o, seed_r, nb, n_trees, D)
rng(100 * cid + o + 1000 * (seed_r - 1), 'twister');
V = cfg.V; P = cfg.P;
train_p = setdiff(1:P, o);
nfail = 0;
r = bayesopt(@inner_error, make_bvars(name, NF), 'MaxObjectiveEvaluations', nb, 'AcquisitionFunctionName', cfg.bo_acq, ...
             'IsObjectiveDeterministic', ismember(name, {'NC', 'MNR', 'PLSDA'}), 'Verbose', 0, 'PlotFcn', []);
[tr_idx, tr_lab] = plant_idx(V, P, train_p);
[te_idx, te_lab] = plant_idx(V, P, o);
[acc, pred] = fit_eval(r.XAtMinObjective, tr_idx, tr_lab, te_idx, te_lab);
R = struct('acc', acc, 'pred', pred, 'te_lab', te_lab, 'best', r.XAtMinObjective, 'inner', r.MinObjective, 'nfail', nfail);

    function err = inner_error(x)
        accs = zeros(numel(train_p), 1);
        for j = 1:numel(train_p)
            [a_idx, a_lab] = plant_idx(V, P, setdiff(train_p, train_p(j)));
            [v_idx, v_lab] = plant_idx(V, P, train_p(j));
            [accs(j), ~, bad] = fit_eval(x, a_idx, a_lab, v_idx, v_lab);
            nfail = nfail + bad;
        end
        progress('tick', D, 1);
        err = 100 - mean(accs);
    end

    function [acc, pred, bad] = fit_eval(opt, tr_idx, tr_lab, te_idx, te_lab)
        pred = zeros(size(te_lab)); acc = 0; bad = 0;
        try
            Ftr = Fall(tr_idx, :);
            fs = anova_f(Ftr, tr_lab, V);
            [~, ord] = sort(fs, 'descend');
            sel = ord(1:min(opt.num_feats, NF));
            mu = mean(Ftr(:, sel), 1); sg = std(Ftr(:, sel), 0, 1); sg(sg < eps) = 1;
            Xtr = (Ftr(:, sel) - mu) ./ sg;
            Xte = (Fall(te_idx, sel) - mu) ./ sg;
            switch name
                case 'NC'
                    mc = zeros(V, size(Xtr, 2));
                    for c = 1:V, mc(c, :) = mean(Xtr(tr_lab == c, :), 1); end
                    [~, pred] = min(pdist2(Xte, mc), [], 2);
                case {'SVM', 'LOGIT'}
                    lrn = 'svm'; if strcmp(name, 'LOGIT'), lrn = 'logistic'; end
                    t = templateLinear('Learner', lrn, 'Lambda', opt.lambda, 'Regularization', 'ridge');
                    pred = predict(fitcecoc(Xtr, tr_lab, 'Learners', t, 'Coding', 'onevsone'), Xte);
                case 'MNR'
                    W = mnr_fit(Xtr, tr_lab, V, opt.lambda);
                    [~, pred] = max([Xte, ones(size(Xte, 1), 1)] * W, [], 2);
                case 'RF'
                    t = templateTree('MinLeafSize', opt.leaf, 'NumVariablesToSample', max(1, round(opt.mtry * size(Xtr, 2))));
                    pred = predict(fitcensemble(Xtr, tr_lab, 'Method', 'Bag', 'NumLearningCycles', n_trees, 'Learners', t), Xte);
                case 'PLSDA'
                    nc = min([opt.ncomp, size(Xtr, 2), size(Xtr, 1) - 1]);
                    Yd = full(sparse((1:size(Xtr, 1))', tr_lab, 1, size(Xtr, 1), V));
                    [~, ~, ~, ~, beta] = plsregress(Xtr, Yd, nc);
                    [~, pred] = max([ones(size(Xte, 1), 1), Xte] * beta, [], 2);
            end
            acc = mean(pred(:) == te_lab(:)) * 100;
        catch
            bad = 1;
        end
    end
end

function v = make_bvars(name, NF)
nf = optimizableVariable('num_feats', [1, NF], 'Type', 'integer', 'Transform', 'log');
switch name
    case 'NC', v = nf;
    case {'SVM', 'LOGIT', 'MNR'}, v = [nf, optimizableVariable('lambda', [1e-6, 1], 'Type', 'real', 'Transform', 'log')];
    case 'RF', v = [nf, optimizableVariable('leaf', [1, 20], 'Type', 'integer'), ...
                    optimizableVariable('mtry', [0.01, 1], 'Type', 'real', 'Transform', 'log')];
    case 'PLSDA', v = [nf, optimizableVariable('ncomp', [1, 200], 'Type', 'integer', 'Transform', 'log')];
end
end

function W = mnr_fit(X, y, V, lambda)
[n, p] = size(X);
Xb = [X, ones(n, 1)];
Y = full(sparse((1:n)', y(:), 1, n, V));
w = zeros((p + 1) * V, 1);
[f, g] = mnr_obj(w, Xb, Y, lambda, p, V);
S = zeros(numel(w), 0); Yh = S;
for it = 1:500
    q = g; k = size(S, 2); al = zeros(k, 1); rho = zeros(k, 1);
    for i = k:-1:1
        rho(i) = 1 / (Yh(:, i)' * S(:, i));
        al(i) = rho(i) * (S(:, i)' * q);
        q = q - al(i) * Yh(:, i);
    end
    if k > 0, q = q * (S(:, k)' * Yh(:, k)) / (Yh(:, k)' * Yh(:, k)); else, q = q / max(norm(g), 1); end
    for i = 1:k
        q = q + S(:, i) * (al(i) - rho(i) * (Yh(:, i)' * q));
    end
    d = -q; gd = g' * d;
    if gd >= 0, d = -g; gd = -(g' * g); end
    t = 1;
    while true
        wn = w + t * d;
        [fn, gn] = mnr_obj(wn, Xb, Y, lambda, p, V);
        if fn <= f + 1e-4 * t * gd || t < 1e-10, break; end
        t = t / 2;
    end
    s = wn - w; yv = gn - g;
    conv = abs(f - fn) < 1e-9 * max(1, abs(f)) || norm(gn, inf) < 1e-6;
    w = wn; f = fn; g = gn;
    if conv, break; end
    if s' * yv > 1e-10
        S = [S, s]; Yh = [Yh, yv];
        if size(S, 2) > 10, S(:, 1) = []; Yh(:, 1) = []; end
    end
end
W = reshape(w, p + 1, V);
end

function [f, g] = mnr_obj(w, Xb, Y, lambda, p, V)
W = reshape(w, p + 1, V);
Z = Xb * W;
Z = Z - max(Z, [], 2);
E = exp(Z);
Ss = sum(E, 2);
f = -sum(sum(Y .* (Z - log(Ss)))) / size(Xb, 1) + lambda / 2 * sum(sum(W(1:p, :).^2));
G = Xb' * (E ./ Ss - Y) / size(Xb, 1);
G(1:p, :) = G(1:p, :) + lambda * W(1:p, :);
g = G(:);
end
