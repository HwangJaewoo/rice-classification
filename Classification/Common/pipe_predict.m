% pipe_predict.m : Predict labels with a model from pipe_fit
function pred = pipe_predict(M, F, mu, sg)
if nargin < 3, mu = M.mu; sg = M.sg; end
pred = predict(M.lda, (F(:, M.sel) - mu) ./ sg);
end
