% nested_mixed.m : Mixed-year nested evaluation (train and test on both years) of one scenario over several seeds
function S = nested_mixed(seeds, k, comb)

%% Settings
if nargin < 1, seeds = 1:5; end
if nargin < 2, k = 27; end
if nargin < 3, comb = 'geo'; end
if ~ismember(comb, {'geo', 'arith', 'min', 'pooled'}), error('comb must be geo, arith, min or pooled'); end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'mixed');
cdir = out_dir(cfg, 'mixed', 'checkpoints');
run_log = start_run(cfg, 'mixed', sprintf('nested_mixed_%s_k%d', comb, k));

%% Data
sc = scenario(cfg, k);
F23 = extract_features(load_store(cfg, '2023', sc), sc.stages);
F24 = extract_features(load_store(cfg, '2024', sc), sc.stages);
fprintf('Mixed | scenario %d (%s) | F combination %s | seeds %s | features %d\n', k, sc.label, comb, mat2str(seeds), size(F23, 2));

%% Jobs
[jo, js] = ndgrid(1:P, seeds);
jo = jo(:); js = js(:);
J = numel(jo);
files = arrayfun(@(j) fullfile(cdir, sprintf('checkpoint_mixed_%s_k%d_s%d_f%d.mat', comb, k, js(j), jo(j))), ...
                 (1:J)', 'UniformOutput', false);
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
        Rj = mixed_fold(cfg, F23, F24, comb, jo(j), js(j), D);
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(ns, P); a23 = acc; a24 = acc; f1 = zeros(ns, 1); inner = acc;
best = cell(ns, P);
for a = 1:ns
    yt = []; yp = [];
    for o = 1:P
        Rj = R{js == seeds(a) & jo == o};
        acc(a, o) = Rj.acc; a23(a, o) = Rj.acc23; a24(a, o) = Rj.acc24;
        inner(a, o) = Rj.inner; best{a, o} = Rj.best;
        yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
        if Rj.failed, fprintf('!! seed %d fold %d: outer fit failed\n', seeds(a), o); end
    end
    f1(a) = macro_f1(yt, yp, V);
end
T = table(seeds(:), acc(:, 1), acc(:, 2), acc(:, 3), acc(:, 4), mean(acc, 2), std(acc, 0, 2), ...
          mean(a23, 2), mean(a24, 2), f1, mean(inner, 2), ...
          'VariableNames', {'Seed', 'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Mean', 'FoldSD', 'Part2023', 'Part2024', 'MacroF1', 'InnerErr'});
disp(T);
bt = best.';
B = vertcat(bt{:});
B = [table(repelem(seeds(:), P), repmat((1:P)', ns, 1), 'VariableNames', {'Seed', 'Fold'}), B];
disp(B);
fprintf('Mixed %s scenario %d: %.2f (seed SD %.2f, mean fold SD %.2f, 2023 part %.2f, 2024 part %.2f, macro-F1 %.3f) | %.1f min\n', ...
        comb, k, mean(T.Mean), std(T.Mean), mean(T.FoldSD), mean(T.Part2023), mean(T.Part2024), mean(f1), toc(t0) / 60);

%% Save
S = struct('k', k, 'comb', comb, 'label', sc.label, 'seeds', seeds, 'acc', acc, 'acc23', a23, 'acc24', a24, ...
           'f1', f1, 'inner', inner, 'T', T, 'B', B, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('nested_mixed_%s_k%d', comb, k));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Seeds');
writetable(B, [base '.xlsx'], 'Sheet', 'Hyperparameters');
end
