%% Load data
input_dir_path=cd;
table_R = readtable(fullfile(input_dir_path, "r_timeseries.csv"), detectImportOptions(fullfile(input_dir_path, "r_timeseries.csv"), 'Encoding', 'CP949', 'ReadVariableNames', true));
table_G = readtable(fullfile(input_dir_path, "g_timeseries.csv"), detectImportOptions(fullfile(input_dir_path, "g_timeseries.csv"), 'Encoding', 'CP949', 'ReadVariableNames', true));
table_B = readtable(fullfile(input_dir_path, "b_timeseries.csv"), detectImportOptions(fullfile(input_dir_path, "b_timeseries.csv"), 'Encoding', 'CP949', 'ReadVariableNames', true));

table_R = table2array(table_R(:, 3:end));
table_G = table2array(table_G(:, 3:end));
table_B = table2array(table_B(:, 3:end));

r=load_table(table_R);
g=load_table(table_G);
b=load_table(table_B);


close all
rng(42)

%% Hyperparameters
corr_threshold=0.8;
max_K=10; 
kruskal_rate = 0.4;

%% Stack  
% coordinate=[]; 
% num_unique_coordinate=[];

%% 8 color indices
% color_index= r;
% color_index= g;
% color_index= b;
% color_index= (r)./(r+g+b);
% color_index= (g)./(r+g+b);
% color_index= (b)./(r+g+b);
% color_index= (r-g)./(r+g+b);
% color_index= (g-r)./(g+r);

%% Choose the stages
stage1_idx = 1:3;
stage2_idx = 4:6;
stage3_idx = 7:9;
stage4_idx = 10:12;


%% Feautre Selection & Clustering Method
f=extract_features(color_index);
features=f(:, stage1_idx);
[reduced_features, reduced_idx]=remove_correlated_features(features, corr_threshold); 


[sil_scores, possible_K, ~, best_K] = calculate_silhouette(reduced_features, max_K);
[label, ~] = kmeans(reduced_features, best_K, 'Replicates', 20);

pval=KruskalWallis(reduced_features, label);
reduced_idx = find(pval < 0.05);
reduced_features = reduced_features(:, reduced_idx);

[sil_scores, possible_K, ~, best_K] = calculate_silhouette(reduced_features, max_K);
[label, ~] = kmeans(reduced_features, best_K, 'Replicates', 20);

[coeff, ~, ~, ~, explained] = pca(reduced_features);
cumExplained = cumsum(explained);  
min_features = find(cumExplained >= 90, 1); 
importance_score = sum(abs(coeff(:, 1:min_features)), 2); 
[~, sorted_idx] = sort(importance_score, 'descend');
selected_idx = sorted_idx(1:min_features);
selected_features = reduced_features(:, selected_idx);

[~, possible_K_new, ~, ~] = calculate_silhouette(selected_features, max_K);
reduced_features=selected_features;
possible_K=possible_K_new;


final_possible_K =[];

for K=possible_K
    [label, ~]=kmeans(reduced_features, K, 'Replicates', 20);
    num_data = []; 
    for i=1:K
        num_data=[num_data, sum(label==i)];
    end
    if min(num_data)/max(num_data) >= kruskal_rate
        final_possible_K = [final_possible_K, K];
    end
end

if ~isempty(final_possible_K)
    [final_label, ~] = kmeans(reduced_features, max(final_possible_K), 'Replicates', 20);
    % [~, ~, final_label] = unique(final_label, 'stable');
    coordinate = [coordinate, final_label];
    ID = unique(coordinate, 'rows');
    sprintf("Number of unique coordinates : %d", length(ID(:, 1)))
    num_unique_coordinate = [num_unique_coordinate, length(ID(:, 1))];
    disp(num_unique_coordinate);
end

function avg=load_table(data)
    avg = [];
    index=1:4:97*4;
    for i=1:97
        idx=index(i);
        avg_data = mean(data(idx+0:idx+3, :), 1); 
        avg = [avg; avg_data]; 
    end
end

function features=extract_features(data)

thd1 = 25;
thd2 = 39;
stage1=data(:, 1:thd1);
stage2=data(:, thd1+1:thd2);
stage3=data(:, thd2+1:end);
stage4=data(:, 1:end);
features = []; features1=[]; features2=[]; features3=[]; features4=[];
for i=1:97
    y1=stage1(i, :);
    y2=stage2(i, :);
    y3=stage3(i, :);
    y4=stage4(i, :);
    features1 = [features1; [max(y1), min(y1), std(y1)]];
    features2 = [features2; [max(y2), min(y2), std(y2)]];
    features3 = [features3; [max(y3), min(y3), std(y3)]];
    features4 = [features4; [max(y4), min(y4), std(y4)]];
end
features=[features1, features2, features3, features4];
for i=1:length(features(1,:))
    features(:, i)=rescale(features(:, i), 0, 1);
end
end


function [filtered_features, idx]= remove_correlated_features(features, threshold)
    R = corrcoef(features);
    mask = triu(true(size(R)), 1);
    [row, col] = find(abs(R) > threshold & mask);

    remove_idx = [];
    for i = 1:length(row)
        if ~ismember(col(i), remove_idx) && ~ismember(row(i), remove_idx)
            remove_idx(end+1) = col(i);
        end
    end
    remove_idx = unique(remove_idx);

    filtered_features = features;
    filtered_features(:, remove_idx) = [];
    idx=setdiff(1:length(features(1,:)), remove_idx);
end


function [sil_scores, possible_K, max_sil, best_K] = calculate_silhouette(features, num_cluster)
    possible_K = 1:num_cluster;             
    sil_scores = zeros(1, num_cluster);      

    for k = 2:num_cluster
        temp_idx = kmeans(features, k, 'Replicates', 20);
        sil_scores(k) = mean(silhouette(features, temp_idx));
    end

    possible_K = possible_K(sil_scores > 0.5);
    possible_K = setdiff(possible_K, 1);

    [max_sil, best_K] = max(sil_scores);
end



function pvals=KruskalWallis(features, label)

    [~, num_features] = size(features);
    pvals = zeros(1, num_features);
    for i = 1:num_features
        [p, tbl, stats] = kruskalwallis(features(:,i), label, 'off');
        pvals(i) = p;
    end

    format long
    disp('--- Statistically significant features (p < 0.05) ---')
    
    for i = 1:num_features
        fprintf("Feature %d: p = %.4f\n", i, pvals(i));
    end
end

