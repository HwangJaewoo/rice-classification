function [error, accuracy, cve_val, n_pca_actual] = logic_framework(opt_vars, data_store, num_varieties, phases, pinvA_list, num_aug, noise_level, scale_range)

    rng(42, 'twister'); 
    num_feats = opt_vars.num_feats;
    cve_target = opt_vars.cve_target;
    gamma = opt_vars.gamma;
    
    [~, num_days, num_channels] = size(data_store);
    
    test_indices = zeros(num_varieties, 1);
    train_indices_total = zeros(num_varieties * 3, 1);
    for v = 1:num_varieties
        idx = (v-1)*4 + (1:4);
        t_idx_local = randi(4);
        test_indices(v) = idx(t_idx_local);
        train_indices_total((v-1)*3+1 : v*3) = idx(setdiff(1:4, t_idx_local));
    end

    test_raw = data_store(test_indices, :, :);
    test_features = extract_features(test_raw, phases, pinvA_list);
    test_labels = (1:num_varieties)';

    train_raw = data_store(train_indices_total, :, :);
    train_labels_raw = repelem((1:num_varieties)', 3);

    aug_train_d = repmat(train_raw, num_aug, 1, 1);
    n_samples_aug = size(aug_train_d, 1);
    
    n_d = noise_level * std(aug_train_d, 0, 2) .* randn(n_samples_aug, num_days, num_channels);
    sc = (1 - scale_range) + (2 * scale_range) * rand(n_samples_aug, 1, 1);
    aug_train_d = (aug_train_d + n_d) .* sc;

    train_features = extract_features(aug_train_d, phases, pinvA_list);
    train_labels = repmat(train_labels_raw, num_aug, 1);

    n_samples = size(train_features, 1);
    overall_mean = mean(train_features, 1);
    ss_b = zeros(1, size(train_features, 2)); ss_w = zeros(1, size(train_features, 2));
    for k = 1:num_varieties
        c_idx = (train_labels == k);
        c_data = train_features(c_idx, :);
        ss_b = ss_b + size(c_data, 1) * (mean(c_data,1) - overall_mean).^2;
        ss_w = ss_w + sum((c_data - mean(c_data,1)).^2, 1);
    end
    f_scores = (ss_b / (num_varieties-1)) ./ (ss_w / (n_samples - num_varieties) + eps);
    [~, s_idx] = sort(f_scores, 'descend');
    sel_idx = s_idx(1:min(num_feats, length(f_scores)));
    
    train_x = train_features(:, sel_idx); test_x = test_features(:, sel_idx);
    [train_x, mu, sigma] = zscore(train_x);
    test_x = (test_x - mu) ./ (sigma + eps);

    [coeff, score, ~, ~, explained] = pca(train_x, 'Economy', true);
    cve_cum = cumsum(explained);
    n_pca_actual = find(cve_cum >= cve_target, 1);
    if isempty(n_pca_actual), n_pca_actual = length(explained); end
    
    cve_val = cve_cum(n_pca_actual);
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
end