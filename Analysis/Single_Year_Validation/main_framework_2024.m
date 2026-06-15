clear; clc;

num_varieties = 97; 
% base_path = input your dataset path

fixed_aug = 10;
fixed_noise = 0.003;
fixed_scale = 0.005;

stage_cases = {{1:25}, {1:25, 26:39}, {1:25, 26:39, 40:61}};
stage_names = {'Stage 1', 'Stage 1-2', 'Stage 1-3'};

color_idx = {'r', 'g', 'b', 'h', 's', 'v', 'y', 'cb', 'cr', 'l_lab', 'a_lab', 'b_lab'};
geo_idx = {'width', 'diagonal_h', 'area', 'vertical_h', 'Peri', 'FSEP', 'a1', 'b1', 'c1', 'd1', 'F_std'};
index_cases = {color_idx, geo_idx, [color_idx, geo_idx]};
index_names = {'Color', 'Geo', 'All'};

angle_cases = {{''}, {'', '_120'}, {'', '_120', '_240'}};
angle_names = {'Single(0)', 'Multi(0,120)', 'Multi(0,120,240)'};

[g1, g2, g3] = meshgrid(1:3, 1:3, 1:3);
scenarios = [g1(:), g2(:), g3(:)];
num_scenarios = size(scenarios, 1);

results_cell = cell(num_scenarios, 9);

for s_idx = 1:num_scenarios
    st_idx = scenarios(s_idx, 1);
    id_idx = scenarios(s_idx, 2);
    an_idx = scenarios(s_idx, 3);
        
    stages = stage_cases{st_idx};
    multi_view_indices = index_cases{id_idx};
    angles = angle_cases{an_idx};
    
    all_days = [stages{:}];
    num_total_days = length(all_days);
    num_multi = length(multi_view_indices);
    num_derived = 4; 
    ch_per_angle = num_multi + num_derived;
    total_channels = ch_per_angle * length(angles);
    
    pinvA_list = cell(length(stages), 1);
    for p = 1:length(stages)
        t_tmp = (1:length(stages{p}))';
        M = [t_tmp.^2, t_tmp, ones(length(t_tmp), 1)];
        pinvA_list{p} = (M' * M) \ M';
    end

    data_store = zeros(388, num_total_days, total_channels);
    idx_w = find(strcmp(multi_view_indices, 'width'));
    idx_vh = find(strcmp(multi_view_indices, 'vertical_h'));

    for a_idx = 1:length(angles)
        ang = angles{a_idx}; 
        ch_offset = (a_idx-1) * ch_per_angle;
        raw_tmp = zeros(388, num_total_days, num_multi);
        
        for b_idx = 1:num_multi
            path = sprintf('%s%s_2024%s.csv', base_path, multi_view_indices{b_idx}, ang);
            if exist(path, 'file')
                temp = readmatrix(path);
                val = temp(:, 2 + all_days);
                raw_tmp(:, :, b_idx) = movmean(val, 3, 2); 
            end
        end
        data_store(:,:,ch_offset + (1:num_multi)) = raw_tmp;
        
        if num_multi >= 3 && id_idx ~= 2 
            r = raw_tmp(:,:,1); g = raw_tmp(:,:,2); b = raw_tmp(:,:,3);
            data_store(:,:,ch_offset+num_multi+1) = 2*g - r - b; 
            data_store(:,:,ch_offset+num_multi+2) = (2*g - r - b) ./ (2*g + r + b + 1e-9); 
            data_store(:,:,ch_offset+num_multi+3) = (g - r) ./ (g + r + 1e-9); 
        end

        if ~isempty(idx_w) && ~isempty(idx_vh)
            data_store(:,:,ch_offset+num_multi+4) = raw_tmp(:,:,idx_w) ./ (raw_tmp(:,:,idx_vh) + 1e-6); 
        end
    end

    clear wrapper_framework; 
    
    vars = [
        optimizableVariable('num_feats', [100, 600], 'Type', 'integer'), ...
        optimizableVariable('cve_target', [85, 99.9], 'Type', 'real'), ...
        optimizableVariable('gamma', [0.01, 0.4], 'Type', 'real')
    ];

    obj_func = @(x) wrapper_framework(x, data_store, num_varieties, ...
                    stages, pinvA_list, fixed_aug, ...
                    fixed_noise, fixed_scale);

    results_opt = bayesopt(obj_func, vars, ...
        'MaxObjectiveEvaluations', 50, ... 
        'Verbose', 0, 'PlotFcn', []);
    
    best_p = results_opt.XAtMinObjective;
    [~, f_top1, ~, f_npca] = logic_framework(best_p, data_store, ...
                                 num_varieties, stages, pinvA_list, fixed_aug, ...
                                 fixed_noise, fixed_scale);

    results_cell(s_idx, :) = {s_idx, stage_names{st_idx}, index_names{id_idx}, angle_names{an_idx}, ...
                              best_p.num_feats, f_npca, best_p.cve_target, ...
                              best_p.gamma, f_top1};
end

% T = cell2table(results_cell, 'VariableNames', {'Scnenario', 'Stages', 'Indices', 'Angles', 'Num_Feats', 'Num_PCs', 'tau_CVE', 'Gamma', 'Accuracy'});
% writetable(T, 'Single_Year_Validation_Results(2024).xlsx');

