% single_fold.m : One outer fold of single-year nested leave-one-plant-out
function R = single_fold(cfg, Fall, o, seed_r, D)

%% Inner optimization
rng(o + 1000 * (seed_r - 1), 'twister');
V = cfg.V; P = cfg.P;
train_p = setdiff(1:P, o);
obj = @(x) inner_error(x, cfg, Fall, train_p, D);
r = bayesopt(obj, make_vars(size(Fall, 2), cfg), 'MaxObjectiveEvaluations', cfg.n_bo, ...
             'AcquisitionFunctionName', cfg.bo_acq, 'IsObjectiveDeterministic', true, 'Verbose', 0, 'PlotFcn', []);

%% Outer test
[tr_idx, tr_lab] = plant_idx(V, P, train_p);
[te_idx, te_lab] = plant_idx(V, P, o);
[acc, pred] = fit_eval(r.XAtMinObjective, V, Fall, tr_idx, tr_lab, te_idx, te_lab);
R = struct('acc', acc, 'pred', pred, 'te_lab', te_lab, 'best', r.XAtMinObjective, 'inner', r.MinObjective, 'failed', all(pred == 0));
end

function err = inner_error(x, cfg, Fall, train_p, D)
accs = zeros(numel(train_p), 1);
for j = 1:numel(train_p)
    [tr_idx, tr_lab] = plant_idx(cfg.V, cfg.P, setdiff(train_p, train_p(j)));
    [va_idx, va_lab] = plant_idx(cfg.V, cfg.P, train_p(j));
    accs(j) = fit_eval(x, cfg.V, Fall, tr_idx, tr_lab, va_idx, va_lab);
end
progress('tick', D, 1);
err = 100 - mean(accs);
end

function [acc, pred] = fit_eval(opt, V, Fall, tr_idx, tr_lab, te_idx, te_lab)
pred = zeros(size(te_lab));
acc = 0;
try
    M = pipe_fit(Fall(tr_idx, :), tr_lab, opt, V);
    pred = pipe_predict(M, Fall(te_idx, :));
    acc = mean(pred == te_lab) * 100;
catch ME
    fprintf('!! fit failed: %s\n', ME.message);
end
end
