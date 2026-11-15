% index_redundancy.m : Redundancy of the color and morphological features (correlation and effective dimension)
function T = index_redundancy()

%% Settings
addpath('../Common');
cfg = config();
rdir = out_dir(cfg, 'analysis');
run_log = start_run(cfg, 'analysis', 'index_redundancy');
sc = scenario(cfg, cfg.rep_scenario);
color_set = [cfg.color_idx, {'ExG', 'GLI', 'NGRDI'}];

%% Analyse
rows = {};
for y = {'2023', '2024'}
    [ds, ch_names] = load_store(cfg, y{1}, sc);
    F = extract_features(ds, sc.stages);
    n_ang = numel(sc.angles);
    cpa = numel(ch_names) / n_ang;
    [col_ch, ~, col_stat] = feature_map(numel(sc.stages), numel(ch_names));
    base = regexprep(ch_names(1:cpa), '_\d+$', '');
    is_color_ch = ismember(base, color_set);
    col_color = is_color_ch(mod(col_ch - 1, cpa) + 1);
    col_color = col_color(:);
    for g = {'Color', 'Morphology'}
        in_g = col_color == strcmp(g{1}, 'Color');
        for f = {'stage means', 'all features'}
            sel = in_g;
            if strcmp(f{1}, 'stage means'), sel = sel & col_stat == 1; end
            X = F(:, sel);
            X = X(:, std(X, 0, 1) > eps);
            R = corrcoef(X);
            lam = sort(eig(R), 'descend');
            lam = max(lam, 0);
            eff = sum(lam)^2 / sum(lam.^2);
            n90 = find(cumsum(lam) / sum(lam) >= 0.9, 1);
            r_off = abs(R(~eye(size(R))));
            rows(end+1, :) = {y{1}, g{1}, f{1}, size(X, 2), mean(r_off), median(r_off), eff, n90}; %#ok<AGROW>
        end
    end
end
T = cell2table(rows, 'VariableNames', {'Year', 'Group', 'Features', 'nFeatures', 'MeanAbsR', 'MedianAbsR', 'EffectiveDim', 'nPC90'});
disp(T);

%% Save
xls = fullfile(rdir, 'index_redundancy.xlsx');
if exist(xls, 'file'), delete(xls); end
writetable(T, xls, 'Sheet', 'Summary');
fprintf('Saved %s\n', xls);
end
