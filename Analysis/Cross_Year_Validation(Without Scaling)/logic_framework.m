function [error, accuracy, cve_val, n_pca_actual] = logic_framework(opt_vars, data_store, num_varieties, stages23, pinvA23, stages24, pinvA24, num_aug, noise_level, scale_range)

    rng(42, 'twister'); 
    num_feats = opt_vars.num_feats;
    cve_target = opt_vars.cve_target;
    gamma = opt_vars.gamma;
    
    [~, num_days, num_channels] = size(data_store);
    base_labels = repelem((1:num_varieties)', 4); 

    data_store_calibrated = data_store; 

    train_idx = zeros(num_varieties * 4, 1);
    test_idx = zeros(num_varieties * 4, 1);
    for v = 1:num_varieties
        base = (v-1)*8;
        train_idx((v-1)*4 + (1:4)) = base + (1:4); 
        test_idx((v-1)*4 + (1:4)) = base + (5:8);  
    end
    
    train_store = data_store_calibrated(train_idx, :, :);
    test_store = data_store_calibrated(test_idx, :, :);

    aug_train_d = repmat(train_store, num_aug, 1, 1);
    n_samples_aug = size(aug_train_d, 1);
    
    n_d = noise_level * std(aug_train_d, 0, 2) .* randn(n_samples_aug, num_days, num_channels);
    sc = (1 - scale_range) + (2 * scale_range) * rand(n_samples_aug, 1, 1);
    aug_train_d = (aug_train_d + n_d) .* sc;
    
    train_features = extract_features(aug_train_d, stages23, pinvA23);
    train_labels = repmat(base_labels, num_aug, 1);
    test_features = extract_features(test_store, stages24, pinvA24);
    test_labels = base_labels;

    n_samples_tr = size(train_features, 1);
    overall_mean = mean(train_features, 1);
    ss_b = zeros(1, size(train_features, 2)); 
    ss_w = zeros(1, size(train_features, 2));
    
    for k = 1:num_varieties
        c_idx = (train_labels == k);
        c_data = train_features(c_idx, :);
        if size(c_data, 1) < 2, continue; end
        ss_b = ss_b + size(c_data, 1) * (mean(c_data, 1) - overall_mean).^2;
        ss_w = ss_w + sum((c_data - mean(c_data, 1)).^2, 1);
    end
    f_scores = (ss_b / (num_varieties - 1)) ./ (ss_w / (n_samples_tr - num_varieties) + eps);
    [~, s_idx] = sort(f_scores, 'descend');
    sel_idx = s_idx(1:min(num_feats, length(f_scores)));
    
    train_x = train_features(:, sel_idx); 
    test_x = test_features(:, sel_idx);
    train_x = zscore(train_x); 
    test_x = zscore(test_x);   

    [coeff, score, ~, ~, explained] = pca(train_x, 'Economy', true);
    cve_cum = cumsum(explained);
    n_pca_actual = find(cve_cum >= cve_target, 1);
    if isempty(n_pca_actual), n_pca_actual = length(explained); end
    
    score_red = score(:, 1:n_pca_actual);
    coeff_red = coeff(:, 1:n_pca_actual);

    try
        lda_model = fitcdiscr(score_red, train_labels, 'DiscrimType', 'linear', 'Gamma', gamma);
        [pred, ~] = predict(lda_model, test_x * coeff_red);
        
        accuracy = sum(pred == test_labels) / length(test_labels) * 100;
        error = 100 - accuracy; 
    catch
        error = 100; accuracy = 0;
    end
    cve_val = cve_cum(n_pca_actual);
end