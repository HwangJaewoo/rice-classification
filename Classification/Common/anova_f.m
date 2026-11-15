% anova_f.m : One-way ANOVA F score (and p-value) of every column
function [F, p] = anova_f(X, y, V)
n  = size(X, 1);
om = mean(X, 1);
ssb = zeros(1, size(X, 2)); ssw = ssb;
for k = 1:V
    c = X(y == k, :);
    ssb = ssb + size(c, 1) * (mean(c, 1) - om).^2;
    ssw = ssw + sum((c - mean(c, 1)).^2, 1);
end
F = (ssb / (V - 1)) ./ (ssw / (n - V) + eps);
if nargout > 1, p = fcdf(F, V - 1, n - V, 'upper'); end
end
