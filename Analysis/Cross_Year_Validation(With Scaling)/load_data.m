function ds = load_data(path_base, year_str, n_days, days, angles, indices, ch_angle, n_multi)
    
    ds = zeros(388, n_days, ch_angle * length(angles));
    
    idx_w = find(strcmp(indices, 'width'));
    idx_vh = find(strcmp(indices, 'vertical_h'));
    
    for a_idx = 1:length(angles)
        ang = angles{a_idx}; 
        ch_offset = (a_idx-1) * ch_angle;
        raw_tmp = zeros(388, n_days, n_multi);
        
        for b_idx = 1:n_multi
            fpath = sprintf('%s%s_%s%s.csv', path_base, indices{b_idx}, year_str, ang);
            if exist(fpath, 'file')
                temp = readmatrix(fpath);
                val = temp(:, 2 + days);
                raw_tmp(:, :, b_idx) = movmean(val, 3, 2); 
            end
        end
        
        ds(:,:,ch_offset + (1:n_multi)) = raw_tmp;
        
        if n_multi >= 3
            r = raw_tmp(:,:,1); g = raw_tmp(:,:,2); b = raw_tmp(:,:,3);
            ds(:,:,ch_offset+n_multi+1) = 2*g - r - b; 
            ds(:,:,ch_offset+n_multi+2) = (2*g - r - b) ./ (2*g + r + b + 1e-9); 
            ds(:,:,ch_offset+n_multi+3) = (g - r) ./ (g + r + 1e-9); 
        end
        
        if ~isempty(idx_w) && ~isempty(idx_vh)
            ds(:,:,ch_offset+n_multi+4) = raw_tmp(:,:,idx_w) ./ (raw_tmp(:,:,idx_vh) + 1e-6); 
        end
    end
end