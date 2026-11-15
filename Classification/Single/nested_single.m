% nested_single.m : Single-year nested evaluation of one scenario over several seeds
function S = nested_single(year_str, seeds, k)

%% Settings
if nargin < 1, year_str = '2023'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, k = 27; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'single');
cdir = out_dir(cfg, 'single', 'checkpoints');
run_log = start_run(cfg, 'single', sprintf('nested_single_%s_k%d', year_str, k));

%% Data
sc = scenario(cfg, k);
ds = load_store(cfg, year_str, sc);
Fall = extract_features(ds, sc.stages);
fprintf('Single %s | scenario %d (%s) | seeds %s | features %d\n', year_str, k, sc.label, mat2str(seeds), size(Fall, 2));

%% Jobs
[jo, js] = meshgrid(1:P, seeds);
jo = jo(:); js = js(:);
J = numel(jo);
ck = @(j) fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, k, js(j), jo(j)));
done = arrayfun(@(j) exist(ck(j), 'file') == 2, 1:J);
if any(done), fprintf('Resume: %d of %d jobs done\n', sum(done), J); end
D = progress('start', sum(~done) * cfg.n_bo);

%% Run
R = cell(J, 1);
t0 = tic;
parfor j = 1:J
    warning('off', 'all');
    f = fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, k, js(j), jo(j)));
    if exist(f, 'file') == 2
        R{j} = checkpoint('load', f);
    else
        Rj = single_fold(cfg, Fall, jo(j), js(j), D);
        checkpoint('save', f, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(ns, P); f1 = zeros(ns, 1); inner = zeros(ns, P);
best = cell(ns, P);
for a = 1:ns
    yt = []; yp = [];
    for o = 1:P
        Rj = R{js == seeds(a) & jo == o};
        acc(a, o) = Rj.acc; inner(a, o) = Rj.inner; best{a, o} = Rj.best;
        yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
        if all(Rj.pred == 0), fprintf('!! seed %d fold %d: outer fit failed\n', seeds(a), o); end
    end
    f1(a) = macro_f1(yt, yp, V);
end
seed_mean = mean(acc, 2);
T = table(seeds(:), acc(:, 1), acc(:, 2), acc(:, 3), acc(:, 4), seed_mean, std(acc, 0, 2), f1, mean(inner, 2), ...
          'VariableNames', {'Seed', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Mean', 'FoldSD', 'MacroF1', 'InnerErr'});
disp(T);
bt = best.';
B = vertcat(bt{:});
B = [table(repelem(seeds(:), P), repmat((1:P)', ns, 1), 'VariableNames', {'Seed', 'Fold'}), B];
disp(B);
fprintf('Single %s scenario %d: %.2f (seed SD %.2f, mean fold SD %.2f, macro-F1 %.3f) | %.1f min\n', ...
        year_str, k, mean(seed_mean), std(seed_mean), mean(T.FoldSD), mean(f1), toc(t0) / 60);

%% Save
S = struct('year', year_str, 'k', k, 'label', sc.label, 'seeds', seeds, 'acc', acc, ...
           'f1', f1, 'inner', inner, 'T', T, 'B', B, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('nested_single_%s_k%d', year_str, k));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Seeds');
writetable(B, [base '.xlsx'], 'Sheet', 'Hyperparameters');
end
