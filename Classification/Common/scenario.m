% scenario.m : Stages, indices and angles of scenario k (1..27)
function sc = scenario(cfg, k)

%% Index
a = floor((k - 1) / 9) + 1;
s = floor(mod(k - 1, 9) / 3) + 1;
i = mod(k - 1, 3) + 1;

%% Output
sc.k = k;
sc.stages = cfg.stage_cases{s};
sc.angles = cfg.angle_cases{a};
switch i
    case 1, sc.indices = cfg.color_idx;
    case 2, sc.indices = cfg.geo_idx;
    case 3, sc.indices = [cfg.color_idx, cfg.geo_idx];
end
sc.has_color = i ~= 2;
sc.has_geo   = i ~= 1;
sc.label = sprintf('%s | %s | %s', cfg.stage_names{s}, cfg.index_names{i}, cfg.angle_names{a});
sc.stage_name = cfg.stage_names{s};
sc.index_name = cfg.index_names{i};
sc.angle_name = cfg.angle_names{a};
end
