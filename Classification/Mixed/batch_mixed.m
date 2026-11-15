% batch_mixed.m : Mixed-year nested evaluation of all 27 scenarios (Table S6 of the SI)
function S = batch_mixed(seeds, ks, comb)

%% Settings
if nargin < 1, seeds = 1:5; end
if nargin < 2, ks = 1:27; end
if nargin < 3, comb = 'geo'; end
if ~ismember(comb, {'geo', 'arith', 'min', 'pooled'}), error('comb must be geo, arith, min or pooled'); end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'mixed');
cdir = out_dir(cfg, 'mixed', 'checkpoints');
run_log = start_run(cfg, 'mixed', sprintf('batch_mixed_%s', comb));

%% Data
nk = numel(ks);
F23 = cell(nk, 1); F24 = cell(nk, 1); sc = cell(nk, 1);
for a = 1:nk
    sc{a} = scenario(cfg, ks(a));
    F23{a} = extract_features(load_store(cfg, '2023', sc{a}), sc{a}.stages);
    F24{a} = extract_features(load_store(cfg, '2024', sc{a}), sc{a}.stages);
end
fprintf('Mixed batch | %d scenarios | F combination %s | seeds %s\n', nk, comb, mat2str(seeds));

%% Jobs
[jo, js, ja] = ndgrid(1:P, seeds, 1:nk);
jo = jo(:); js = js(:); ja = ja(:);
J = numel(jo);
files = arrayfun(@(j) fullfile(cdir, sprintf('checkpoint_mixed_%s_k%d_s%d_f%d.mat', comb, ks(ja(j)), js(j), jo(j))), ...
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
        Rj = mixed_fold(cfg, F23{ja(j)}, F24{ja(j)}, comb, jo(j), js(j), D);
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
ns = numel(seeds);
acc = zeros(nk, ns, P); a23 = acc; a24 = acc; f1 = zeros(nk, ns);
for a = 1:nk
    for b = 1:ns
        yt = []; yp = [];
        for o = 1:P
            Rj = R{ja == a & js == seeds(b) & jo == o};
            acc(a, b, o) = Rj.acc; a23(a, b, o) = Rj.acc23; a24(a, b, o) = Rj.acc24;
            yt = [yt; Rj.te_lab]; yp = [yp; Rj.pred];
            if Rj.failed, fprintf('!! scenario %d seed %d fold %d: outer fit failed\n', ks(a), seeds(b), o); end
        end
        f1(a, b) = macro_f1(yt, yp, V);
    end
end
seed_mean = mean(acc, 3);
T = table(ks(:), cellfun(@(s) s.stage_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.index_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.angle_name, sc, 'UniformOutput', false), ...
          mean(seed_mean, 2), std(seed_mean, 0, 2), mean(std(acc, 0, 3), 2), ...
          mean(mean(a23, 3), 2), mean(mean(a24, 3), 2), mean(f1, 2), ...
          'VariableNames', {'Scenario', 'Stages', 'Indices', 'Angles', 'Mean', 'SeedSD', 'FoldSD', 'Part2023', 'Part2024', 'MacroF1'});
disp(T);
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('ks', ks, 'comb', comb, 'seeds', seeds, 'acc', acc, 'acc23', a23, 'acc24', a24, 'f1', f1, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('batch_mixed_%s', comb));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Summary');
SM = array2table(seed_mean, 'VariableNames', compose('Seed%d', seeds));
writetable([T(:, 1:4), SM], [base '.xlsx'], 'Sheet', 'Seeds');
end
