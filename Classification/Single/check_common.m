% check_common.m : Quick check of the data path and the common functions (no optimization)
function check_common(year_str)

%% Setup
if nargin < 1, year_str = '2023'; end
addpath('../Common');
cfg = config();
V = cfg.V; P = cfg.P;
run_log = start_run(cfg, 'single', sprintf('check_common_%s', year_str));

%% Feature counts
for k = [1 2 3 27]
    sc = scenario(cfg, k);
    [ds, ch] = load_store(cfg, year_str, sc);
    NF = size(extract_features(ds(1, :, :), sc.stages), 2);
    fprintf('Scenario %2d (%s): %d channels, %d features\n', k, sc.label, numel(ch), NF);
end

%% One fold, fixed hyperparameters
sc = scenario(cfg, cfg.rep_scenario);
Fall = extract_features(load_store(cfg, year_str, sc), sc.stages);
[te_idx, te_lab] = plant_idx(V, P, 1);
[tr_idx, tr_lab] = plant_idx(V, P, 2:4);
M = pipe_fit(Fall(tr_idx, :), tr_lab, struct('num_feats', 400, 'gamma', 0.6), V);
pred = pipe_predict(M, Fall(te_idx, :));
fprintf('Fold 1 (num_feats 400, gamma 0.6): acc %.2f%%, macro-F1 %.3f\n', mean(pred == te_lab) * 100, macro_f1(te_lab, pred, V));
end
