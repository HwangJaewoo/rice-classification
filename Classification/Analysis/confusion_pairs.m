% confusion_pairs.m : Confusion matrices, per-cultivar recall and most confused cultivar pairs from saved nested results
function out = confusion_pairs(n_top)

%% Settings
if nargin < 1, n_top = 20; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
rdir = out_dir(cfg, 'analysis');
run_log = start_run(cfg, 'analysis', 'confusion_pairs');

T = readtable(fullfile(cfg.data_root, '2023', 'r_2023.csv'), 'TextType', 'string');
names = strtrim(T{1:P:end, 1});
if numel(names) ~= V, error('Expected %d cultivar names, found %d', V, numel(names)); end

%% Collect predictions
sets = {};
for y = {'2023', '2024'}
    f = fullfile(cfg.results_root, 'single', sprintf('nested_single_%s_k%d.mat', y{1}, cfg.rep_scenario));
    if exist(f, 'file')
        S = load(f, 'R');
        [yt, yp] = pool(S.R, true(numel(S.R), 1), []);
        sets(end+1, :) = {sprintf('Single_%s', y{1}), yt, yp}; %#ok<AGROW>
    end
end
f = fullfile(cfg.results_root, 'mixed', sprintf('nested_mixed_geo_k%d.mat', cfg.rep_scenario));
if exist(f, 'file')
    S = load(f, 'R');
    sel = true(numel(S.R), 1);
    [yt, yp] = pool(S.R, sel, []);
    sets(end+1, :) = {'Mixed', yt, yp};
    [yt, yp] = pool(S.R, sel, 1:V);
    sets(end+1, :) = {'Mixed_2023part', yt, yp};
    [yt, yp] = pool(S.R, sel, V + (1:V));
    sets(end+1, :) = {'Mixed_2024part', yt, yp};
end
for d = {'23to24', '24to23'}
    f = fullfile(cfg.results_root, 'cross', sprintf('nested_cross_%s_k%d.mat', d{1}, cfg.rep_scenario));
    if exist(f, 'file')
        S = load(f, 'R', 'modes', 'seeds');
        [~, jm] = ndgrid(S.seeds, 1:numel(S.modes));
        for m = 1:numel(S.modes)
            [yt, yp] = pool(S.R, jm(:) == m, []);
            sets(end+1, :) = {sprintf('Cross_%s_%s', d{1}, S.modes{m}), yt, yp}; %#ok<AGROW>
        end
    end
end
if isempty(sets), error('No saved results found under %s', cfg.results_root); end

%% Analyse
xls = fullfile(rdir, sprintf('confusion_pairs_k%d.xlsx', cfg.rep_scenario));
if exist(xls, 'file'), delete(xls); end
summary = cell(size(sets, 1), 5);
out = struct();
for s = 1:size(sets, 1)
    [tag, yt, yp] = sets{s, :};
    C = confusionmat(yt, yp, 'Order', 1:V);
    n_true = sum(C, 2);
    recall = diag(C) ./ max(n_true, 1);
    W = C + C.';
    W(1:V+1:end) = 0;
    [i, j] = find(triu(W) > 0);
    cnt = W(sub2ind([V V], i, j));
    rate = cnt ./ (n_true(i) + n_true(j));
    [~, ord] = sortrows([cnt, rate], [-1 -2]);
    ord = ord(1:min(n_top, numel(ord)));
    Tp = table(names(i(ord)), names(j(ord)), C(sub2ind([V V], i(ord), j(ord))), C(sub2ind([V V], j(ord), i(ord))), ...
               cnt(ord), rate(ord) * 100, recall(i(ord)) * 100, recall(j(ord)) * 100, ...
               'VariableNames', {'CultivarA', 'CultivarB', 'A_as_B', 'B_as_A', 'Total', 'RatePct', 'RecallA', 'RecallB'});
    Tr = sortrows(table(names, recall * 100, n_true, 'VariableNames', {'Cultivar', 'RecallPct', 'nTest'}), 'RecallPct');
    sh = matlab.lang.makeValidName(tag);
    writetable(Tp, xls, 'Sheet', [sh '_pairs']);
    writetable(Tr, xls, 'Sheet', [sh '_recall']);
    top_share = sum(cnt(ord)) / max(sum(W(:)) / 2, 1) * 100;
    summary(s, :) = {tag, mean(yt == yp) * 100, sum(recall == 0), sum(recall == 1), top_share};
    out.(sh) = struct('C', C, 'pairs', Tp, 'recall', Tr);
    save_matrix(C, recall, tag, fullfile(rdir, sprintf('confusion_%s_k%d.png', sh, cfg.rep_scenario)));
    fprintf('%-18s acc %6.2f | top-%d pairs %5.1f%% of errors | worst pair %s / %s (%d)\n', ...
            tag, mean(yt == yp) * 100, numel(ord), top_share, Tp.CultivarA(1), Tp.CultivarB(1), Tp.Total(1));
end
Ts = cell2table(summary, 'VariableNames', {'Setting', 'AccPct', 'nRecall0', 'nRecall100', 'TopPairsShareOfErrorsPct'});
writetable(Ts, xls, 'Sheet', 'Summary');
save(fullfile(rdir, sprintf('confusion_pairs_k%d.mat', cfg.rep_scenario)), 'out', 'names', 'Ts');
fprintf('Saved %s\n', xls);
if nargout == 0, clear out; end
end

%% Helpers
function [yt, yp] = pool(R, sel, rows)
yt = []; yp = [];
for j = find(sel(:))'
    t = R{j}.te_lab; p = R{j}.pred;
    if ~isempty(rows), t = t(rows); p = p(rows); end
    yt = [yt; t(:)]; yp = [yp; p(:)]; %#ok<AGROW>
end
keep = yp >= 1;
yt = yt(keep); yp = yp(keep);
end

function save_matrix(C, recall, tag, f)
fig = figure('Visible', 'off', 'Position', [100 100 760 680]);
Cn = C ./ max(sum(C, 2), 1);
imagesc(Cn, [0 1]); axis square; colormap(flipud(gray)); colorbar;
xlabel('Predicted cultivar'); ylabel('True cultivar');
title(sprintf('%s (mean recall %.1f%%)', strrep(tag, '_', ' '), mean(recall) * 100));
exportgraphics(fig, f, 'Resolution', 200);
close(fig);
end
