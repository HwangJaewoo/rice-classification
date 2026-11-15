% progress.m : 20% progress. D = progress('start', total) on client, progress('tick', D, n) anywhere
function D = progress(mode, a, b)
persistent done tot t0 last
D = [];
switch mode
    case 'start'
        done = 0; tot = max(a, 1); t0 = tic; last = 0;
        if license('test', 'Distrib_Computing_Toolbox')
            if isempty(gcp('nocreate')), parpool; end
            D = parallel.pool.DataQueue;
            afterEach(D, @(n) progress('add', n));
        end
    case 'tick'
        if isempty(a), progress('add', b); else, send(a, b); end
    case 'add'
        done = done + a;
        stp = floor(min(100, floor(100 * done / tot)) / 20) * 20;
        if stp <= last, return; end
        last = stp;
        el = toc(t0);
        fprintf('[%3d%%] elapsed %.1f min, ETA %.1f min\n', stp, el / 60, el / done * max(tot - done, 0) / 60);
end
end
