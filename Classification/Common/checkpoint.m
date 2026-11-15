% checkpoint.m : checkpoint('save', file, R) / R = checkpoint('load', file)
function R = checkpoint(mode, f, R)
switch mode
    case 'save', save(f, '-struct', 'R');
    case 'load', R = load(f);
end
end
