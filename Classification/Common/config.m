% config.m : Settings shared by all scripts
function cfg = config()

%% Paths and sizes
root = fileparts(fileparts(mfilename('fullpath')));
cfg.data_root = fullfile(root, '..', 'Preprocessed_dataset');
cfg.results_root = fullfile(root, 'results');
cfg.V = 97;
cfg.P = 4;

%% Scenario lists
cfg.stage_cases = {{1:25}, {1:25, 26:39}, {1:25, 26:39, 40:61}};
cfg.stage_names = {'Stage 1', 'Stage 1-2', 'Stage 1-3'};
cfg.color_idx = {'r', 'g', 'b', 'h', 's', 'v', 'y', 'cb', 'cr', 'l_lab', 'a_lab', 'b_lab'};
cfg.geo_idx   = {'width', 'diagonal_h', 'area', 'vertical_h', 'Peri', 'FSEP', 'a1', 'b1', 'c1', 'd1', 'F_std'};
cfg.index_names = {'Color', 'Morphology', 'All'};
cfg.angle_cases = {{''}, {'', '_120'}, {'', '_120', '_240'}};
cfg.angle_names = {'0', '0,120', '0,120,240'};
cfg.rep_scenario = 27;

%% Bayesian optimization
cfg.n_bo = 50;
cfg.bo_acq = 'expected-improvement-plus';
cfg.gamma_range = [1e-7, 1];
end
