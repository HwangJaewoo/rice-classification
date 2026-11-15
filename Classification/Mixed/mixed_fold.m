% mixed_fold.m : One outer fold of mixed-year nested leave-one-plant-out (test = plant o of both years)
function R = mixed_fold(cfg, F23, F24, comb, o, seed_r, D)

%% Inner optimization
rng(o + 1000 * (seed_r - 1), 'twister');
V = cfg.V; P = cfg.P;
train_p = setdiff(1:P, o);
obj = @(x) inner_error(x, cfg, F23, F24, comb, train_p, D);
r = bayesopt(obj, make_vars(size(F23, 2), cfg), 'MaxObjectiveEvaluations', cfg.n_bo, ...
             'AcquisitionFunctionName', cfg.bo_acq, 'IsObjectiveDeterministic', true, 'Verbose', 0, 'PlotFcn', []);

%% Outer test
[pred, te_lab] = fit_eval(r.XAtMinObjective, V, P, F23, F24, comb, train_p, o);
hit = pred == te_lab;
R = struct('acc', mean(hit) * 100, 'acc23', mean(hit(1:V)) * 100, 'acc24', mean(hit(V+1:end)) * 100, ...
           'pred', pred, 'te_lab', te_lab, 'best', r.XAtMinObjective, 'inner', r.MinObjective, 'failed', all(pred == 0));
end

function err = inner_error(x, cfg, F23, F24, comb, train_p, D)
accs = zeros(numel(train_p), 1);
for j = 1:numel(train_p)
    [pred, lab] = fit_eval(x, cfg.V, cfg.P, F23, F24, comb, setdiff(train_p, train_p(j)), train_p(j));
    accs(j) = mean(pred == lab) * 100;
end
progress('tick', D, 1);
err = 100 - mean(accs);
end

function [pred, te_lab] = fit_eval(opt, V, P, F23, F24, comb, fit_p, test_p)
[tr_idx, tr_lab] = plant_idx(V, P, fit_p);
[te_idx, te_lab1] = plant_idx(V, P, test_p);
te_lab = [te_lab1; te_lab1];
pred = zeros(size(te_lab));
try
    A = F23(tr_idx, :); B = F24(tr_idx, :);
    mA = mean(A, 1); sA = std(A, 0, 1); sA(sA < eps) = 1;
    mB = mean(B, 1); sB = std(B, 0, 1); sB(sB < eps) = 1;
    ZA = (A - mA) ./ sA; ZB = (B - mB) ./ sB;
    switch comb
        case 'geo',    fs = sqrt(anova_f(A, tr_lab, V) .* anova_f(B, tr_lab, V));
        case 'arith',  fs = (anova_f(A, tr_lab, V) + anova_f(B, tr_lab, V)) / 2;
        case 'min',    fs = min(anova_f(A, tr_lab, V), anova_f(B, tr_lab, V));
        case 'pooled', fs = anova_f([ZA; ZB], [tr_lab; tr_lab], V);
    end
    [~, ord] = sort(fs, 'descend');
    sel = ord(1:min(opt.num_feats, numel(fs)));
    lda = fitcdiscr([ZA(:, sel); ZB(:, sel)], [tr_lab; tr_lab], 'DiscrimType', 'linear', ...
                    'Gamma', opt.gamma, 'FillCoeffs', 'off');
    Xte = [(F23(te_idx, sel) - mA(sel)) ./ sA(sel); (F24(te_idx, sel) - mB(sel)) ./ sB(sel)];
    pred = predict(lda, Xte);
catch ME
    fprintf('!! fit failed: %s\n', ME.message);
end
end
