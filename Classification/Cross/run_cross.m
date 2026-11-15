% run_cross.m : All cross-year experiments, in order
clc;

%% Representative scenario, both directions, source- and target-year standardization, 5 seeds
nested_cross('23to24', 1:5, 27, {'source', 'target'});
nested_cross('24to23', 1:5, 27, {'source', 'target'});

%% All 27 scenarios, both directions (scenario 27 reuses the checkpoints above)
batch_cross('23to24', 1:5, 1:27, {'source', 'target'});
batch_cross('24to23', 1:5, 1:27, {'source', 'target'});

%% Min-max scaling before source- and target-year standardization (saved with the suffix of the mode)
nested_cross('23to24', 1:5, 27, {'source_minmax'});
nested_cross('24to23', 1:5, 27, {'source_minmax'});
nested_cross('23to24', 1:5, 27, {'target_minmax'});
nested_cross('24to23', 1:5, 27, {'target_minmax'});
