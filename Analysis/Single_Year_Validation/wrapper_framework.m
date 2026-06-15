function error = wrapper_framework(opt_vars, data_store, num_varieties, phases, pinvA_list, num_aug, noise_level, scale_range)
    persistent best_accuracy;
    
    if isempty(best_accuracy)
        best_accuracy = -1;
    end

    [error, accuracy, ~, ~] = logic_framework(opt_vars, data_store, num_varieties, phases, pinvA_list, num_aug, noise_level, scale_range);
    
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        fprintf('    >>> Accuracy: %.2f%% (Loss: %.2f)\n', accuracy, error);
    end
end