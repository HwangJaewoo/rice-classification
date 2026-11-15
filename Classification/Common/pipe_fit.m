% pipe_fit.m : ANOVA selection -> standardization -> shrinkage LDA
function M = pipe_fit(Ftr, ytr, opt, V, f_scores)
if nargin < 5 || isempty(f_scores), f_scores = anova_f(Ftr, ytr, V); end
[~, o] = sort(f_scores, 'descend');
sel = o(1:min(opt.num_feats, size(Ftr, 2)));
Z = Ftr(:, sel);
mu = mean(Z, 1);
sg = std(Z, 0, 1);
sg(sg < eps) = 1;
lda = fitcdiscr((Z - mu) ./ sg, ytr, 'DiscrimType', 'linear', 'Gamma', opt.gamma, 'FillCoeffs', 'off');
M = struct('sel', sel, 'mu', mu, 'sg', sg, 'lda', lda);
end
