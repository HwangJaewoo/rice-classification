% batch_cross.m : Cross-year evaluation of all 27 scenarios (Table S6 of the SI)
function S = batch_cross(dir_str, seeds, ks, modes)

%% Settings
if nargin < 1, dir_str = '23to24'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, ks = 1:27; end
if nargin < 4, modes = {'source', 'target'}; end
modes = cellstr(modes);
if ~all(ismember(modes, {'source', 'target'})), error('modes must be source and/or target'); end
switch dir_str
    case '23to24', src = '2023'; tgt = '2024';
    case '24to23', src = '2024'; tgt = '2023';
    otherwise, error('dir_str must be ''23to24'' or ''24to23''');
end
addpath('../Common');
cfg = config();
V = cfg.V;
rdir = out_dir(cfg, 'cross');
cdir = out_dir(cfg, 'cross', 'checkpoints');
run_log = start_run(cfg, 'cross', sprintf('batch_cross_%s', dir_str));

%% Data
nk = numel(ks);
FS = cell(nk, 1); FT = cell(nk, 1); sc = cell(nk, 1);
for a = 1:nk
    sc{a} = scenario(cfg, ks(a));
    FS{a} = extract_features(load_store(cfg, src, sc{a}), sc{a}.stages);
    FT{a} = extract_features(load_store(cfg, tgt, sc{a}), sc{a}.stages);
end
fprintf('Cross batch %s -> %s | %d scenarios | modes %s | seeds %s\n', src, tgt, nk, strjoin(modes, ','), mat2str(seeds));

%% Jobs
[js, jm, ja] = ndgrid(seeds, 1:numel(modes), 1:nk);
js = js(:); jm = jm(:); ja = ja(:);
J = numel(js);
files = arrayfun(@(j) fullfile(cdir, sprintf('checkpoint_cross_%s_%s_k%d_s%d.mat', dir_str, modes{jm(j)}, ks(ja(j)), js(j))), ...
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
        Rj = cross_fold(cfg, FS{ja(j)}, FT{ja(j)}, modes{jm(j)}, js(j), D);
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
M = numel(modes); ns = numel(seeds);
acc = zeros(nk, M, ns); psd = acc; f1 = acc;
for a = 1:nk
    for m = 1:M
        for b = 1:ns
            Rj = R{ja == a & jm == m & js == seeds(b)};
            acc(a, m, b) = Rj.acc; psd(a, m, b) = std(Rj.plant_acc);
            if Rj.failed, fprintf('!! scenario %d mode %s seed %d: fit failed\n', ks(a), modes{m}, seeds(b)); end
            f1(a, m, b) = macro_f1(Rj.te_lab, Rj.pred, V);
        end
    end
end
T = table(ks(:), cellfun(@(s) s.stage_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.index_name, sc, 'UniformOutput', false), ...
          cellfun(@(s) s.angle_name, sc, 'UniformOutput', false), ...
          'VariableNames', {'Scenario', 'Stages', 'Indices', 'Angles'});
for m = 1:M
    T.(sprintf('%s_Mean', modes{m}))    = mean(acc(:, m, :), 3);
    T.(sprintf('%s_SeedSD', modes{m}))  = std(acc(:, m, :), 0, 3);
    T.(sprintf('%s_PlantSD', modes{m})) = mean(psd(:, m, :), 3);
    T.(sprintf('%s_MacroF1', modes{m})) = mean(f1(:, m, :), 3);
end
disp(T);
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('dir', dir_str, 'ks', ks, 'modes', {modes}, 'seeds', seeds, 'acc', acc, 'psd', psd, 'f1', f1, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('batch_cross_%s', dir_str));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Summary');
end
