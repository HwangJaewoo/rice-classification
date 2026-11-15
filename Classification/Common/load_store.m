% load_store.m : Smoothed index time series (N x days x channels) for one year and scenario
function [ds, ch_names] = load_store(cfg, year_str, sc)

%% Setup
base_path = fullfile(cfg.data_root, year_str);
all_days = [sc.stages{:}];
N = cfg.V * cfg.P;
idx = sc.indices;
n_multi = numel(idx);
add_color = sc.has_color;
iw  = find(strcmp(idx, 'width'));
ivh = find(strcmp(idx, 'vertical_h'));
add_shape = ~isempty(iw) && ~isempty(ivh);
extra = {};
if add_color, extra = [extra, {'ExG', 'GLI', 'NGRDI'}]; end
if add_shape, extra = [extra, {'AR'}]; end
cpa = n_multi + numel(extra);
ang_tag = strrep(sc.angles, '_', '');
ang_tag(strcmp(ang_tag, '')) = {'0'};

%% Load
ds = zeros(N, numel(all_days), cpa * numel(sc.angles));
ch_names = cell(1, cpa * numel(sc.angles));
for a = 1:numel(sc.angles)
    off = (a - 1) * cpa;
    raw = zeros(N, numel(all_days), n_multi);
    for b = 1:n_multi
        f = fullfile(base_path, sprintf('%s_%s%s.csv', idx{b}, year_str, sc.angles{a}));
        if ~exist(f, 'file'), error('File not found: %s', f); end
        tmp = readmatrix(f);
        if size(tmp, 1) ~= N, error('Row count %d ~= %d in %s', size(tmp, 1), N, f); end
        if any(isnan(tmp(:, 2 + all_days)), 'all'), error('NaN in %s', f); end
        raw(:, :, b) = movmean(tmp(:, 2 + all_days), 3, 2);
    end
    ds(:, :, off + (1:n_multi)) = raw;
    c = off + n_multi;
    if add_color
        r = raw(:, :, 1); g = raw(:, :, 2); bl = raw(:, :, 3);
        ds(:, :, c + 1) = 2*g - r - bl;
        ds(:, :, c + 2) = (2*g - r - bl) ./ (2*g + r + bl + 1e-9);
        ds(:, :, c + 3) = (g - r) ./ (g + r + 1e-9);
        c = c + 3;
    end
    if add_shape
        ds(:, :, c + 1) = raw(:, :, iw) ./ (raw(:, :, ivh) + 1e-6);
    end
    ch_names(off + (1:cpa)) = strcat([idx, extra], '_', ang_tag{a});
end
end
