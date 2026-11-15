% year_stability.m : Between-year stability of each index (cultivar x year variance components, cross-year correlation, year shift)
function year_stability(k)

%% Settings
if nargin < 1, k = 27; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'analysis');
run_log = start_run(cfg, 'analysis', 'year_stability');

%% Data
sc = scenario(cfg, k);
[d23, ch] = load_store(cfg, '2023', sc);
d24 = load_store(cfg, '2024', sc);
stages = cfg.stage_cases{3};
n_ch = numel(ch);
color_set = [cfg.color_idx, {'ExG', 'GLI', 'NGRDI'}];
fprintf('Scenario %d (%s) | channels %d | stages %d\n', k, sc.label, n_ch, numel(stages));

%% Per channel and stage
rows = cell(n_ch * numel(stages), 14);
r = 0;
for c = 1:n_ch
    u = find(ch{c} == '_', 1, 'last');
    base = ch{c}(1:u - 1); ang = ch{c}(u + 1:end);
    if ismember(base, color_set), typ = 'Color'; else, typ = 'Morphology'; end
    for s = 1:numel(stages)
        y = cat(3, reshape(mean(d23(:, stages{s}, c), 2), P, V), reshape(mean(d24(:, stages{s}, c), 2), P, V));
        st = stats(y);
        r = r + 1;
        rows(r, :) = {base, typ, ang, s, st.icc(1), st.icc(2), st.r, st.rho, st.shift, ...
                      st.pG, st.pGY, st.pE, st.H2, st.F_G};
    end
end
T = cell2table(rows, 'VariableNames', {'Index', 'Type', 'Angle', 'Stage', 'ICC2023', 'ICC2024', ...
    'CrossYearR', 'CrossYearRho', 'YearShiftSD', 'VarCultivarPct', 'VarCultivarYearPct', 'VarResidualPct', ...
    'H2', 'F_Cultivar'});

%% Summary
G = findgroups(T.Type, T.Stage);
[gt, gs] = deal(splitapply(@(x) x(1), T.Type, G), splitapply(@(x) x(1), T.Stage, G));
S = table(gt, gs, 'VariableNames', {'Type', 'Stage'});
for v = {'ICC2023', 'ICC2024', 'CrossYearR', 'YearShiftSD', 'VarCultivarPct', 'VarCultivarYearPct', 'VarResidualPct', 'H2'}
    S.(v{1}) = splitapply(@median, abs(T.(v{1})), G);
end
Gt = findgroups(T.Type);
S2 = table(splitapply(@(x) x(1), T.Type, Gt), 'VariableNames', {'Type'});
for v = {'ICC2023', 'ICC2024', 'CrossYearR', 'YearShiftSD', 'VarCultivarPct', 'VarCultivarYearPct', 'VarResidualPct', 'H2'}
    S2.(v{1}) = splitapply(@median, abs(T.(v{1})), Gt);
end
disp(S2); disp(S);

%% Save
xls = fullfile(rdir, sprintf('year_stability_k%d.xlsx', k));
if exist(xls, 'file'), delete(xls); end
writetable(S2, xls, 'Sheet', 'ByType');
writetable(S, xls, 'Sheet', 'ByTypeStage');
writetable(sortrows(T, 'H2', 'descend'), xls, 'Sheet', 'All');
save(fullfile(rdir, sprintf('year_stability_k%d.mat', k)), 'T', 'S', 'S2');
fprintf('Saved %s\n', xls);
end

%% Helpers
function st = stats(y)
[P, V, Y] = size(y);
m = mean(y, 'all');
mv = mean(y, [1 3]);
mt = mean(y, [1 2]);
mvt = mean(y, 1);
ss_g  = P * Y * sum((mv - m) .^ 2, 'all');
ss_gy = P * sum((mvt - mv - mt + m) .^ 2, 'all');
ss_e  = sum((y - mvt) .^ 2, 'all');
ms_g = ss_g / (V - 1); ms_gy = ss_gy / (V - 1); ms_e = ss_e / (V * Y * (P - 1));
s_e = ms_e;
s_gy = max((ms_gy - ms_e) / P, 0);
s_g = max((ms_g - ms_gy) / (P * Y), 0);
tot = s_g + s_gy + s_e;
st.pG = 100 * s_g / tot; st.pGY = 100 * s_gy / tot; st.pE = 100 * s_e / tot;
st.H2 = s_g / (s_g + s_gy / Y + s_e / (P * Y));
st.F_G = ms_g / ms_gy;
st.icc = zeros(1, Y);
cm = squeeze(mean(y, 1));
for t = 1:Y
    msb = P * sum((cm(:, t) - mean(cm(:, t))) .^ 2) / (V - 1);
    msw = sum((y(:, :, t) - cm(:, t).') .^ 2, 'all') / (V * (P - 1));
    st.icc(t) = (msb - msw) / (msb + (P - 1) * msw);
end
st.r = corr(cm(:, 1), cm(:, 2));
st.rho = corr(cm(:, 1), cm(:, 2), 'Type', 'Spearman');
st.shift = (mean(cm(:, 2)) - mean(cm(:, 1))) / sqrt((var(cm(:, 1)) + var(cm(:, 2))) / 2);
end
