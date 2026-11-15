% nested_cross.m : Cross-year evaluation of one scenario, modes source, target, source_minmax, target_minmax (see cross_fold.m)
function S = nested_cross(dir_str, seeds, k, modes)

%% Settings
if nargin < 1, dir_str = '23to24'; end
if nargin < 2, seeds = 1:5; end
if nargin < 3, k = 27; end
if nargin < 4, modes = {'source', 'target'}; end
modes = cellstr(modes);
tag = ''; if ~isequal(sort(modes), {'source', 'target'}), tag = ['_' strjoin(modes, '_')]; end
if ~all(ismember(modes, {'source', 'target', 'source_minmax', 'target_minmax'})), error('unknown mode'); end
switch dir_str
    case '23to24', src = '2023'; tgt = '2024';
    case '24to23', src = '2024'; tgt = '2023';
    otherwise, error('dir_str must be ''23to24'' or ''24to23''');
end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'cross');
cdir = out_dir(cfg, 'cross', 'checkpoints');
run_log = start_run(cfg, 'cross', sprintf('nested_cross_%s_k%d%s', dir_str, k, tag));

%% Data
sc = scenario(cfg, k);
XS = load_store(cfg, src, sc);
XT = load_store(cfg, tgt, sc);
FS = extract_features(XS, sc.stages);
FT = extract_features(XT, sc.stages);
stages = sc.stages;
fprintf('Cross %s -> %s | scenario %d (%s) | modes %s | seeds %s | features %d\n', ...
        src, tgt, k, sc.label, strjoin(modes, ','), mat2str(seeds), size(FS, 2));

%% Jobs
[js, jm] = ndgrid(seeds, 1:numel(modes));
js = js(:); jm = jm(:);
J = numel(js);
files = arrayfun(@(j) fullfile(cdir, sprintf('checkpoint_cross_%s_%s_k%d_s%d.mat', dir_str, modes{jm(j)}, k, js(j))), ...
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
        Rj = cross_fold(cfg, FS, FT, modes{jm(j)}, js(j), D, XS, XT, stages);
        checkpoint('save', files{j}, Rj);
        R{j} = Rj;
    end
end

%% Summary
rows = {};
for m = 1:numel(modes)
    for a = 1:numel(seeds)
        Rj = R{jm == m & js == seeds(a)};
        if Rj.failed, fprintf('!! mode %s seed %d: fit failed\n', modes{m}, seeds(a)); end
        rows(end+1, :) = {modes{m}, seeds(a), Rj.acc, Rj.plant_acc(1), Rj.plant_acc(2), Rj.plant_acc(3), ...
                          Rj.plant_acc(4), std(Rj.plant_acc), macro_f1(Rj.te_lab, Rj.pred, V), Rj.inner, ...
                          Rj.best.num_feats, Rj.best.gamma}; %#ok<AGROW>
    end
end
T = cell2table(rows, 'VariableNames', {'Mode', 'Seed', 'Acc', 'Plant1', 'Plant2', 'Plant3', 'Plant4', ...
                                       'PlantSD', 'MacroF1', 'InnerErr', 'num_feats', 'gamma'});
disp(T);
for m = 1:numel(modes)
    Tm = T(strcmp(T.Mode, modes{m}), :);
    fprintf('Cross %s -> %s mode %s: %.2f (seed SD %.2f, mean plant SD %.2f, macro-F1 %.3f)\n', ...
            src, tgt, modes{m}, mean(Tm.Acc), std(Tm.Acc), mean(Tm.PlantSD), mean(Tm.MacroF1));
end
fprintf('Elapsed %.1f min\n', toc(t0) / 60);

%% Save
S = struct('dir', dir_str, 'k', k, 'label', sc.label, 'modes', {modes}, 'seeds', seeds, 'T', T, 'cfg', cfg);
S.R = R;
base = fullfile(rdir, sprintf('nested_cross_%s_k%d%s', dir_str, k, tag));
save([base '.mat'], '-struct', 'S');
if exist([base '.xlsx'], 'file'), delete([base '.xlsx']); end
writetable(T, [base '.xlsx'], 'Sheet', 'Seeds');
end
