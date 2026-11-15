% macro_f1.m : Macro-averaged F1 over V classes
function f = macro_f1(yt, yp, V)
yp = yp(:); yt = yt(:);
yp(~ismember(yp, 1:V)) = 0;
C  = confusionmat(yt, yp, 'Order', 0:V);
tp = diag(C(2:end, 2:end));
pr = tp ./ max(sum(C(:, 2:end), 1)', 1);
rc = tp ./ max(sum(C(2:end, :), 2), 1);
f = mean(2 * pr .* rc ./ max(pr + rc, eps));
end
