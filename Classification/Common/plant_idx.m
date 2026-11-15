% plant_idx.m : Row indices and labels of the given plant numbers for all cultivars
function [idx, lab] = plant_idx(V, P, plants)
m   = ((1:V)' - 1) * P + plants(:)';
idx = m(:);
lab = repmat((1:V)', numel(plants), 1);
end
