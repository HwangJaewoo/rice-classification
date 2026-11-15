% start_run.m : Clear the command window (top-level calls only) and log output to results/<sub>/logs
function c = start_run(cfg, sub, name)
st = dbstack;
if numel(st) <= 2, clc; end
f = fullfile(out_dir(cfg, sub, 'logs'), sprintf('%s_%s.txt', name, datestr(now, 'yyyymmdd_HHMMSS')));
diary(f);
fprintf('Log: %s\n', f);
c = onCleanup(@() diary('off'));
end
