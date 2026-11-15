% run_mixed.m : All mixed-year experiments of the revised manuscript, in order
clc;

%% Representative scenario, geometric mean of yearly ANOVA F, 5 seeds
nested_mixed(1:5, 27, 'geo');

%% Table S6 (SI): all 27 scenarios, 5 seeds (scenario 27 reuses the checkpoints above)
batch_mixed(1:5, 1:27, 'geo');

%% F-score combination (R1-11): arithmetic mean, minimum, pooled
nested_mixed(1:5, 27, 'arith');
nested_mixed(1:5, 27, 'min');
nested_mixed(1:5, 27, 'pooled');

%% Permutation importance of the representative scenario (geometric mean)
importance_mixed(1:5, 'geo');
