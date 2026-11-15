% batch_single.m : Single-year nested evaluation of all 27 scenarios (Table S6 of the SI)
function S = batch_single(year_str, seeds, ks)

%% Settings
if nargin < 1, year_str = '2023'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, ks = 1:27; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'single');
cdir = out_dir(cfg, 'single', 'checkpoints');
run_log = start_run(cfg, 'single', sprintf('batch_single_%s', year_str));

%% Data
nk = numel(ks);
Fk = cell(nk, 1); sc = cell(nk, 1);
for a = 1:nk
    sc{a} = scenario(cfg, ks(a));
    Fk{a} = extract_features(load_store(cfg, year_str, sc{a}), sc{a}.stages);
end
fprintf('Single batch %s | %d scenarios | seeds %s | features %d-%d\n', year_str, nk, mat2str(seeds), ...
        min(cellfun(@(F) size(F, 2), Fk)), max(cellfun(@(F) size(F, 2), Fk)));

%% Jobs
[jo, js, ja] = ndgrid(1:P, seeds, 1:nk);
jo = jo(:); js = js(:); ja = ja(:);
J = numel(jo);
fname = @(j) fullfile(cdir, sprintf('checkpoint_single_%s_k%d_s%d_f%d.mat', year_str, ks(ja(j)), js(j), jo(j)));
files = arrayfun(fname, 1:J, 'UniformOutput', false);
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
        Rj = single_fold(cfg, Fk{ja(j)}, jo(j), js(j), D);
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(nk, ns, P); f1 = zeros(nk, ns);
for a = 1:nk
    for b = 1:ns
        yt = []; yp = [];
        for o = 1:P
            Rj = R{ja == a & js == seeds(b) & jo == o};
            acc(a, b, o) = Rj.acc;
            yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
        end
        f1(a, b) = macro_f1(yt, yp, V);
    end
end
seed_mean = mean(acc, 3);
fold_sd = std(acc, 0, 3);
T = table(ks(:), cellfun(@(s) s.stage_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.index_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.angle_name, sc, 'UniformOutput', false), ...
          mean(seed_mean, 2), std(seed_mean, 0, 2), mean(fold_sd, 2), mean(f1, 2), ...
          'VariableNames', {'Scenario', 'Stages', 'Indices', 'Angles', 'Mean', 'SeedSD', 'FoldSD', 'MacroF1'});
disp(T);
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('year', year_str, 'ks', ks, 'seeds', seeds, 'acc', acc, 'f1', f1, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('batch_single_%s', year_str));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Summary');
SM = array2table(seed_mean, 'VariableNames', compose('Seed%d', seeds));
writetable([T(:, 1:4), SM], [base '.xlsx'], 'Sheet', 'Seeds');
end
