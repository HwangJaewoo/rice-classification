% out_dir.m : Results folder (created if missing), e.g. out_dir(cfg, 'single', 'checkpoints')
function d = out_dir(cfg, varargin)
d = fullfile(cfg.results_root, varargin{:});
if ~exist(d, 'dir'), mkdir(d); end
end
