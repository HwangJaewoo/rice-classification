% run_single.m : All single-year experiments of the revised manuscript, in order
clc;

%% Check common functions
check_common('2023');

%% Representative scenario (All / Stage 1-3 / 0,120,240), 5 seeds
nested_single('2023', 1:5);
nested_single('2024', 1:5);

%% Table S6 (SI): all 27 scenarios, 5 seeds (scenario 27 reuses the checkpoints above)
batch_single('2023', 1:5);
batch_single('2024', 1:5);

%% Baselines: 7 classifiers, same splits and preprocessing (LDA reuses the representative run)
baselines_single('2023', 1:5);
baselines_single('2024', 1:5);

%% Ablation: -ANOVA, gamma = 0, +PCA, +augmentation
ablation_single('2023', 1:5);
ablation_single('2024', 1:5);

%% Feature importance, univariate ANOVA, confused cultivar pairs
importance_single('2023', 1:5);
importance_single('2024', 1:5);

%% Augmentation sensitivity (R1-12): one parameter changed at a time from the original setting
sens = {'aug_noise0', 'aug_noise001', 'aug_noise01', 'aug_scale0', 'aug_scale02', 'aug_n1', 'aug_n5', 'aug_n25'};
ablation_single('2023', 1:5, sens, '_augsens');
ablation_single('2024', 1:5, sens, '_augsens');
