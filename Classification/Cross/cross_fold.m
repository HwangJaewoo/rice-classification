% cross_fold.m : Cross-year evaluation of one mode (train on all plants of the source year, test on the target year)
%%   source        : target year standardized with the statistics of the source (training) year
%%   target        : target year standardized with its own statistics (no labels)
%%   source_minmax : min-max scaling of the index time series with the minimum and maximum of the
%%                   fitting plants, applied to both years, then as source
%%   target_minmax : min-max scaling as above, then as target
function R = cross_fold(cfg, FS, FT, mode, seed_r, D, XS, XT, stages)

%% Inner folds on the source year only
rng(1000 * (seed_r - 1) + 1, 'twister');
V = cfg.V; P = cfg.P;
inner = cell(P, 1);
for p = 1:P
    [tr_idx, tr_lab] = plant_idx(V, P, setdiff(1:P, p));
    [va_idx, va_lab] = plant_idx(V, P, p);
    if endsWith(mode, '_minmax')
        [Ftr, Fva] = minmax_feats(XS(tr_idx, :, :), XS(va_idx, :, :), stages);
    else
        Ftr = FS(tr_idx, :); Fva = FS(va_idx, :);
    end
    inner{p} = struct('Ftr', Ftr, 'ytr', tr_lab, 'Fva', Fva, 'yva', va_lab);
end
obj = @(x) inner_error(x, V, inner, mode, D);
r = bayesopt(obj, make_vars(size(FS, 2), cfg), 'MaxObjectiveEvaluations', cfg.n_bo, ...
             'AcquisitionFunctionName', cfg.bo_acq, 'IsObjectiveDeterministic', true, 'Verbose', 0, 'PlotFcn', []);

%% Outer test on the target year
[tr_idx, tr_lab] = plant_idx(V, P, 1:P);
[te_idx, te_lab] = plant_idx(V, P, 1:P);
if endsWith(mode, '_minmax')
    [Ftr, Fte] = minmax_feats(XS(tr_idx, :, :), XT(te_idx, :, :), stages);
else
    Ftr = FS(tr_idx, :); Fte = FT(te_idx, :);
end
pred = fit_eval(r.XAtMinObjective, V, Ftr, tr_lab, Fte, mode);
plant_acc = arrayfun(@(o) mean(pred((o - 1) * V + (1:V)) == te_lab((o - 1) * V + (1:V))) * 100, 1:P);
R = struct('acc', mean(pred == te_lab) * 100, 'plant_acc', plant_acc, 'pred', pred, 'te_lab', te_lab, ...
           'best', r.XAtMinObjective, 'inner', r.MinObjective, 'failed', all(pred == 0));
end

function err = inner_error(x, V, inner, mode, D)
accs = zeros(numel(inner), 1);
for p = 1:numel(inner)
    pred = fit_eval(x, V, inner{p}.Ftr, inner{p}.ytr, inner{p}.Fva, mode);
    accs(p) = mean(pred == inner{p}.yva) * 100;
end
progress('tick', D, 1);
err = 100 - mean(accs);
end

function pred = fit_eval(opt, V, Ftr, ytr, Fev, mode)
pred = zeros(size(Fev, 1), 1);
try
    M = pipe_fit(Ftr, ytr, opt, V);
    if startsWith(mode, 'target')
        mu = mean(Fev(:, M.sel), 1); sg = std(Fev(:, M.sel), 0, 1); sg(sg < eps) = 1;
        pred = pipe_predict(M, Fev, mu, sg);
    else
        pred = pipe_predict(M, Fev);
    end
catch ME
    fprintf('!! fit failed (%s): %s\n', mode, ME.message);
end
end

function [Ftr, Fev] = minmax_feats(Xtr, Xev, stages)
%% Minimum and maximum per channel over the fitting plants and all time points
gmin = min(min(Xtr, [], 1), [], 2);
gmax = max(max(Xtr, [], 1), [], 2);
Ftr = extract_features((Xtr - gmin) ./ (gmax - gmin + eps), stages);
Fev = extract_features((Xev - gmin) ./ (gmax - gmin + eps), stages);
end
