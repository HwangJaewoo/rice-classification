function [error, accuracy, cve_val, n_pca_actual] = logic_framework(opt_vars, data_store, num_varieties, stages23, pinvA23, stages24, pinvA24, num_aug, noise_level, scale_range)

    rng(42, 'twister'); 
    num_feats = opt_vars.num_feats;
    cve_target = opt_vars.cve_target;
    gamma = opt_vars.gamma;
    
    [~, num_days, num_channels] = size(data_store);
    
    test_idx_23 = zeros(num_varieties, 1); test_idx_24 = zeros(num_varieties, 1);
    train_idx_23 = zeros(num_varieties * 3, 1); train_idx_24 = zeros(num_varieties * 3, 1);
    
    for v = 1:num_varieties
        base = (v-1)*8;
        t23 = randi(4); t24 = randi(4);
        test_idx_23(v) = base + t23;
        test_idx_24(v) = base + 4 + t24;
        train_idx_23((v-1)*3+1 : v*3) = base + setdiff(1:4, t23);
        train_idx_24((v-1)*3+1 : v*3) = base + 4 + setdiff(1:4, t24);
    end

    train_labels_raw = repelem((1:num_varieties)', 3); 
    train_labels_aug = repmat(train_labels_raw, num_aug, 1); 

    aug_23 = repmat(data_store(train_idx_23,:,:), num_aug, 1, 1);
    n_d23 = noise_level * std(aug_23, 0, 2) .* randn(size(aug_23,1), num_days, num_channels);
    sc23 = (1 - scale_range) + (2 * scale_range) * rand(size(aug_23,1), 1, 1);
    aug_23 = (aug_23 + n_d23) .* sc23;
    f_tr_23 = extract_features(aug_23, stages23, pinvA23);

    aug_24 = repmat(data_store(train_idx_24,:,:), num_aug, 1, 1);
    n_d24 = noise_level * std(aug_24, 0, 2) .* randn(size(aug_24,1), num_days, num_channels);
    sc24 = (1 - scale_range) + (2 * scale_range) * rand(size(aug_24,1), 1, 1);
    aug_24 = (aug_24 + n_d24) .* sc24;
    f_tr_24 = extract_features(aug_24, stages24, pinvA24);

    [f_tr_23_n, mu23, sig23] = zscore(f_tr_23);
    [f_tr_24_n, mu24, sig24] = zscore(f_tr_24);

    fs23 = calculate_f_scores_local(f_tr_23_n, train_labels_aug, num_varieties);
    fs24 = calculate_f_scores_local(f_tr_24_n, train_labels_aug, num_varieties);
    fs_final = sqrt(fs23 .* fs24);
    [~, s_idx] = sort(fs_final, 'descend');
    sel_idx = s_idx(1:min(num_feats, length(fs_final)));

    train_x = [f_tr_23_n(:, sel_idx); f_tr_24_n(:, sel_idx)];
    train_y = [train_labels_aug; train_labels_aug];

    f_te_23 = extract_features(data_store(test_idx_23,:,:), stages23, pinvA23);
    f_te_24 = extract_features(data_store(test_idx_24,:,:), stages24, pinvA24);
    
    test_x = [(f_te_23 - mu23)./(sig23+eps); (f_te_24 - mu24)./(sig24+eps)];
    test_x = test_x(:, sel_idx);
    test_y = [(1:num_varieties)'; (1:num_varieties)'];

    [coeff, score, ~, ~, explained] = pca(train_x, 'Economy', true);
    n_pca = find(cumsum(explained) >= cve_target, 1);
    if isempty(n_pca), n_pca = size(score, 2); end
    
    cve_val = sum(explained(1:n_pca)); 
    n_pca_actual = n_pca;

    try
        lda = fitcdiscr(score(:, 1:n_pca), train_y, 'DiscrimType', 'linear', 'Gamma', gamma);
        [pred, ~] = predict(lda, test_x * coeff(:, 1:n_pca));
        accuracy = sum(pred == test_y) / length(test_y) * 100;
        error = 100 - accuracy;
    catch
        error = 100; accuracy = 0; 
    end
end

function f = calculate_f_scores_local(X, L, num_v)
    n = size(X, 1); m_all = mean(X, 1);
    sb = zeros(1, size(X, 2)); sw = zeros(1, size(X, 2));
    for k = 1:num_v
        xk = X(L == k, :);
        if size(xk, 1) < 2, continue; end
        sb = sb + size(xk, 1) * (mean(xk, 1) - m_all).^2;
        sw = sw + sum((xk - mean(xk, 1)).^2, 1);
    end
    f = (sb / (num_v - 1)) ./ (sw / (n - num_v) + eps);
end