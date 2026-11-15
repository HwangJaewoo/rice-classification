% export_predictions.m : Predictions for every test plant of the saved nested results in one CSV file
%% Columns: Scheme, Method, Direction, Seed, Fold, Year, Cultivar, Plant, Predicted
%% Accuracies, macro-F1 and confusion matrices can be recomputed from this file.
function T = export_predictions()

%% Setup
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P; k = cfg.rep_scenario;
rdir = out_dir(cfg, 'analysis');
run_log = start_run(cfg, 'analysis', 'export_predictions');
Tn = readtable(fullfile(cfg.data_root, '2023', 'r_2023.csv'), 'TextType', 'string');
names = strtrim(Tn{1:P:end, 1});
rows = {};

%% Single-year (job order as in nested_single.m)
for y = {'2023', '2024'}
    f = fullfile(cfg.results_root, 'single', sprintf('nested_single_%s_k%d.mat', y{1}, k));
    if ~exist(f, 'file'), continue; end
    S = load(f, 'R', 'seeds');
    [jo, js] = meshgrid(1:P, S.seeds); jo = jo(:); js = js(:);
    for j = 1:numel(S.R)
        rows = add(rows, 'Single', 'LDA', '', js(j), jo(j), repmat(string(y{1}), V, 1), ...
                   S.R{j}.te_lab, repmat(jo(j), V, 1), S.R{j}.pred, names);
    end
end

%% Baselines, single-year (job order as in baselines_single.m)
for y = {'2023', '2024'}
    f = fullfile(cfg.results_root, 'single', sprintf('baselines_single_%s.mat', y{1}));
    if ~exist(f, 'file'), continue; end
    S = load(f, 'R', 'seeds', 'clfs');
    [jo, js, jk] = ndgrid(1:P, S.seeds, 1:numel(S.clfs)); jo = jo(:); js = js(:); jk = jk(:);
    for j = 1:numel(S.R)
        if strcmp(S.clfs{jk(j)}, 'LDA'), continue; end
        rows = add(rows, 'Single', S.clfs{jk(j)}, '', js(j), jo(j), repmat(string(y{1}), V, 1), ...
                   S.R{j}.te_lab, repmat(jo(j), V, 1), S.R{j}.pred, names);
    end
end

%% Mixed-year (job order as in nested_mixed.m; test rows = 2023 plants, then 2024 plants)
f = fullfile(cfg.results_root, 'mixed', sprintf('nested_mixed_geo_k%d.mat', k));
if exist(f, 'file')
    S = load(f, 'R', 'seeds');
    [jo, js] = ndgrid(1:P, S.seeds); jo = jo(:); js = js(:);
    yr = [repmat("2023", V, 1); repmat("2024", V, 1)];
    for j = 1:numel(S.R)
        rows = add(rows, 'Mixed', 'LDA', '', js(j), jo(j), yr, S.R{j}.te_lab, repmat(jo(j), 2 * V, 1), S.R{j}.pred, names);
    end
end

%% Cross-year (job order as in nested_cross.m; test rows = plant 1 of all cultivars, then plant 2, ...)
for d = {'23to24', '24to23'}
    tgt = "2024"; if strcmp(d{1}, '24to23'), tgt = "2023"; end
    for tag = {'', '_source_minmax', '_target_minmax'}
        f = fullfile(cfg.results_root, 'cross', sprintf('nested_cross_%s_k%d%s.mat', d{1}, k, tag{1}));
        if ~exist(f, 'file'), continue; end
        S = load(f, 'R', 'seeds', 'modes');
        [js, jm] = ndgrid(S.seeds, 1:numel(S.modes)); js = js(:); jm = jm(:);
        pl = repelem((1:P)', V);
        for j = 1:numel(S.R)
            rows = add(rows, 'Cross', ['LDA_' S.modes{jm(j)}], d{1}, js(j), 0, repmat(tgt, V * P, 1), ...
                       S.R{j}.te_lab, pl, S.R{j}.pred, names);
        end
    end
end
if isempty(rows), error('No saved results found under %s', cfg.results_root); end

%% Save
T = vertcat(rows{:});
f = fullfile(rdir, 'predictions_all.csv');
writetable(T, f);
fprintf('%d predictions written to %s\n', height(T), f);
end

function rows = add(rows, scheme, method, direction, seed, fold, year, lab, plant, pred, names)
n = numel(lab);
pname = repmat("", n, 1);
ok = pred >= 1 & pred <= numel(names);
pname(ok) = names(pred(ok));
rows{end+1} = table(repmat(string(scheme), n, 1), repmat(string(method), n, 1), repmat(string(direction), n, 1), ...
                    repmat(seed, n, 1), repmat(fold, n, 1), year(:), names(lab(:)), plant(:), pname, ...
                    'VariableNames', {'Scheme', 'Method', 'Direction', 'Seed', 'Fold', 'Year', 'Cultivar', 'Plant', 'Predicted'});
end
