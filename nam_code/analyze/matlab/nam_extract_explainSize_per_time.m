function [explainSize,totalTime,timeout,numProcessedFeatures] = nam_extract_explainSize_per_time(jsonData,timeSteps)

% init empty explanation size
explainSize = [];
timeout = jsonData.timeout;
totalTime = nan;
numProcessedFeatures = [];

% check trivial
if isfield(jsonData,'is_part_explanation') ... 
    && ~all(jsonData.is_part_explanation) ...
    && any(jsonData.is_part_explanation)

    switch jsonData.feat_sort_method
        case 'ours'
            [explainSize, totalTime,numProcessedFeatures] = aux_method_ours(jsonData,timeSteps);
        case {'naive','sensitivity'}
            [explainSize, totalTime,numProcessedFeatures] = aux_method_naive(jsonData,timeSteps);
        otherwise
            throw(error('Unknown feature sorting method: ''%s''',json.feat_sort_method))
    end
    
end

end


% Auxiliary functions -----------------------------------------------------

function [explainSize, totalTime, numProcessedFeatures] = aux_method_ours(jsonData,timeSteps)

    if ~isfield(jsonData,'feat_order')
        explainSize = []; return;
    end

    % extract number of features
    numFeatures = numel(jsonData.feat_order);
    % extract search time
    sortingTime = jsonData.time.sorting_time;
    searchTimePerFeat = jsonData.time.full_feature_times;
    % properly set unset times
    searchTimePerFeat(searchTimePerFeat == 0) = nan;

    % get total time per feature
    totalTimePerFeat = sortingTime + searchTimePerFeat;
    totalTime = max(totalTimePerFeat);
    
    % determine first feature that is not part of the explanation
    % for each time step
    explainSize = numFeatures;
    % deduct freed features not part of explanation
    explainSize = explainSize - max(((totalTimePerFeat <= timeSteps) & ~jsonData.is_part_explanation) .* (numFeatures:-1:1)',[],1);
    % deduct edge case of randomly reached explainSize within 
    % (only need to check border)
    isBorder = max(((totalTimePerFeat <= timeSteps) & [jsonData.is_part_explanation(1:end-1) & ~jsonData.is_part_explanation(2:end);false]) .* (numFeatures:-1:1)',[],1);
    explainSize(isBorder > 0) = nnz(jsonData.is_part_explanation);

    assert(explainSize(end) == nnz(jsonData.is_part_explanation))

    % compute #processed features ---
    % values with nan did not need to be tested after the first split, so
    % choosing that time for those values as well
    totalTimePerFeat(isnan(totalTimePerFeat)) = min(totalTimePerFeat(~isnan(totalTimePerFeat)));
    numProcessedFeatures = sum(totalTimePerFeat <= timeSteps,1);
end

function [explainSize, totalTime, numProcessedFeatures] = aux_method_naive(jsonData,timeSteps)

    % times per feat requires rewriting (struct -> array)
    oldTimePerFeat = jsonData.time.full_feature_times;
    numFeatures = numel(jsonData.is_part_explanation);

    % convert into array
    newTimePerFeat = nan(numFeatures,1);
    for i=1:numFeatures
        label = sprintf('x%i',i-1);
        if isfield(oldTimePerFeat,label)
            newTimePerFeat(i) = oldTimePerFeat.(label);
        end
    end
    totalTime = max(newTimePerFeat);

    % compute explanation size
    explainSize = numFeatures - sum((newTimePerFeat <= timeSteps) & ~jsonData.is_part_explanation);

    % compute #processed features ---
    % here, nan actually means unprocessed.
    numProcessedFeatures = sum(newTimePerFeat <= timeSteps,1);

    assert(all(jsonData.is_part_explanation == isnan(newTimePerFeat)))
end