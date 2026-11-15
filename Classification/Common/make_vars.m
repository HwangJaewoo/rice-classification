% make_vars.m : Bayesian optimization variables (number of ANOVA features, LDA shrinkage)
function vars = make_vars(NF, cfg)
vars = [optimizableVariable('num_feats', [1, NF], 'Type', 'integer', 'Transform', 'log'), ...
        optimizableVariable('gamma', cfg.gamma_range, 'Type', 'real')];
end
