function F = extract_features(X_cube, phases, pinvA_list)
[N, ~, C] = size(X_cube); F = [];
num_p = length(phases);
p_means = zeros(N, num_p, C);
p_lens = cellfun(@length, phases);
p_ends = cumsum(p_lens);
p_starts = [1, p_ends(1:end-1)+1];

for p = 1:num_p
    idx = p_starts(p):p_ends(p); X_p = X_cube(:, idx, :);
    for i = 1:C
        slice = X_p(:,:,i);
        m = mean(slice, 2); s = std(slice, 0, 2);
        skw = mean((slice - m).^3, 2) ./ (s.^3 + eps);
        coeffs = (pinvA_list{p} * slice')';
        F = [F, m, s, skw, coeffs(:,1), coeffs(:,2), slice(:,end)./(slice(:,1)+eps)];
        p_means(:, p, i) = m;
    end
end

if num_p > 1
    for i = 1:C
        for p = 2:num_p
            F = [F, log1p(p_means(:,p,i) ./ (p_means(:,p-1,i) + eps))];
        end
    end
end
end