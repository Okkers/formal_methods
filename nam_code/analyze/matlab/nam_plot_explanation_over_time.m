function [finalExplainSizes,totalTimes,timeouts,idxValid] = nam_plot_explanation_over_time(type, jsonFiles, timeSteps, name, color)

% keep track of explanation sizes
explainSizes = nan(numel(jsonFiles), numel(timeSteps));
numProcessedFeatures = nan(numel(jsonFiles), numel(timeSteps));
totalTimes = nan(1,numel(jsonFiles));
timeouts = nan(1,numel(jsonFiles));

assert(numel(jsonFiles) > 0, 'No json files found.')

% Process each json file
for j = 1:numel(jsonFiles)
    jsonFile = jsonFiles(j);
    % read json file
    jsonData = jsondecode(fileread(fullfile(jsonFile.folder, jsonFile.name)));
    numFeatures = numel(jsonData.is_part_explanation);

    % compute explanation size
    [explainSize,totalTimes(j),timeouts(j),numProcessedFeature] = nam_extract_explainSize_per_time(jsonData, timeSteps);

    if ~isempty(explainSize)
        % save size
        explainSizes(j, :) = explainSize;
        numProcessedFeatures(j,:) = numProcessedFeature;
    end
end

% filter out trivial cases
idxValid = ~isnan(explainSizes(:, 1));
explainSizes = explainSizes(idxValid, :);
numProcessedFeatures = numProcessedFeatures(idxValid, :);

switch type
    case 'explainSize'
        ydata = explainSizes;
    case 'numFeatures'
        ydata = numProcessedFeatures / numFeatures;
    otherwise
        error('Unknown plotting type: %s', type)
end

% compute mean and std
q25 = quantile(ydata,0.25,1);
q50 = quantile(ydata,0.50,1);
q75 = quantile(ydata,0.75,1);

% plot
plot(timeSteps, q50, 'DisplayName', name, 'Color', color)
V = [timeSteps, timeSteps(end:-1:1); 
    min(numFeatures,[q25, q75(end:-1:1)])];
plot(polygon(V), 1:2, 'FaceColor', color, 'EdgeColor', 'none', 'FaceAlpha', 0.1, 'HandleVisibility', 'off')

% prepare return arguments
finalExplainSizes = explainSizes(:,end);
totalTimes = totalTimes(~isnan(totalTimes));
timeouts = timeouts(~isnan(timeouts));

end
