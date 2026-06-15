clear; clc;
warning('off', 'all'); 

num_varieties = 97; 
% base_path_23 = input your 2023 dataset path
% base_path_24 = input your 2024 dataset path 
base_path_23 = '../../data/2023/';
base_path_24 = '../../data/2024/';


fixed_aug = 10;
fixed_noise = 0.003;
fixed_scale = 0.005;

stage_cases = {{1:25}, {1:25, 26:39}, {1:25, 26:39, 40:61}}; 
stage_names = {'Stage 1', 'Stage 1-2', 'Stage 1-3'};

Color_idx = {'r', 'g', 'b', 'h', 's', 'v', 'y', 'cb', 'cr', 'l_lab', 'a_lab', 'b_lab'};
Morphology_idx = {'width', 'diagonal_h', 'area', 'vertical_h', 'Peri', 'FSEP', 'a1', 'b1', 'c1', 'd1', 'F_std'};
index_cases = {Color_idx, Morphology_idx, [Color_idx, Morphology_idx]};
index_names = {'Color', 'Morphology', 'All'};

angle_cases = {{''}, {'', '_120'}, {'', '_120', '_240'}};
angle_names = {'Single(0)', 'Multi(0,120)', 'Multi(0,120,240)'};

[g1, g2, g3] = meshgrid(1:3, 1:3, 1:3);
scenarios = [g1(:), g2(:), g3(:)];
num_scenarios = size(scenarios, 1);

results_cell = cell(num_scenarios, 9);

gen_pinvA = @(st) cellfun(@(p) (([ (1:length(p))'.^2, (1:length(p))', ones(length(p), 1) ]' * [ (1:length(p))'.^2, (1:length(p))', ones(length(p), 1) ]) \ [ (1:length(p))'.^2, (1:length(p))', ones(length(p), 1) ]'), st, 'UniformOutput', false);


for s_idx = 1:num_scenarios
    sprintf("Progress : %.2f", 100*s_idx/27)
    close all
    st_idx = scenarios(s_idx, 1);
    id_idx = scenarios(s_idx, 2);
    an_idx = scenarios(s_idx, 3);
    
    clear wrapper_framework; 
    
    curr_stages = stage_cases{st_idx};
    curr_indices = index_cases{id_idx};
    curr_angles = angle_cases{an_idx};
    
    num_days = length([curr_stages{:}]);
    num_multi_idx = length(curr_indices);
    num_derived = 4;
    ch_per_angle = num_multi_idx + num_derived;
    
    pinvA23 = gen_pinvA(curr_stages);
    pinvA24 = gen_pinvA(curr_stages);

    data_23 = load_data(base_path_23, '2023', num_days, [curr_stages{:}], curr_angles, curr_indices, ch_per_angle, num_multi_idx);
    data_24 = load_data(base_path_24, '2024', num_days, [curr_stages{:}], curr_angles, curr_indices, ch_per_angle, num_multi_idx);

    data_store = zeros(num_varieties * 8, num_days, ch_per_angle * length(curr_angles));
    for i = 1:num_varieties
        target_idx = (i-1)*8 + (1:8);
        data_store(target_idx(1:4), :, :) = data_23((i-1)*4 + (1:4), :, :);
        data_store(target_idx(5:8), :, :) = data_24((i-1)*4 + (1:4), :, :);
    end

    idx_w = find(strcmp(curr_indices, 'width'));
    idx_vh = find(strcmp(curr_indices, 'vertical_h'));

    vars = [
        optimizableVariable('num_feats', [100, 600], 'Type', 'integer'), ...
        optimizableVariable('cve_target', [85, 99.9], 'Type', 'real'), ...
        optimizableVariable('gamma', [0.01, 0.5], 'Type', 'real')
    ];


    obj_func = @(x) print_error(x, data_store, num_varieties, ...
                    curr_stages, pinvA23, curr_stages, pinvA24, fixed_aug, fixed_noise, fixed_scale);
    
    results_opt = bayesopt(obj_func, vars, 'MaxObjectiveEvaluations', 50, ...
                           'Verbose', 0);


    best_p = results_opt.XAtMinObjective;
    [~, f_top1, ~, f_npca] = logic_framework(best_p, data_store, num_varieties, ...
                                            curr_stages, pinvA23, curr_stages, pinvA24, fixed_aug, fixed_noise, fixed_scale);

    results_cell(s_idx, :) = {s_idx, stage_names{st_idx}, index_names{id_idx}, angle_names{an_idx}, ...
                              best_p.num_feats, f_npca, best_p.cve_target, ...
                              best_p.gamma, f_top1};
end

T = cell2table(results_cell, 'VariableNames', {'Scnenario', 'Stages', 'Indices', 'Angles', 'Num_Feats', 'Num_PCs', 'tau_CVE', 'Gamma', 'Accuracy'});
writetable(T, 'Cross_Year_Validation_Results_Wo_Min-Max_Scaling.xlsx');
warning('on', 'all'); 


function err = print_error(varargin)
err = wrapper_framework(varargin{:});
end
