% feature_map.m : Channel, stage and statistic of every feature column (order of extract_features)
function [col_ch, col_st, col_stat, stat_names] = feature_map(n_stages, C)
stat_names = {'mean', 'std', 'skew', 'quad', 'lin', 'ratio', 'diff'};
NF = n_stages * C * 6 + C * (n_stages - 1);
col_ch = zeros(NF, 1); col_st = col_ch; col_stat = col_ch;
c = 0;
for p = 1:n_stages
    for i = 1:C
        for s = 1:6
            c = c + 1; col_ch(c) = i; col_st(c) = p; col_stat(c) = s;
        end
    end
end
for i = 1:C
    for p = 2:n_stages
        c = c + 1; col_ch(c) = i; col_st(c) = p; col_stat(c) = 7;
    end
end
end
