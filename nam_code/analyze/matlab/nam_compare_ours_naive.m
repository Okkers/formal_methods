function nam_compare_ours_naive()

% init
resultdir = './nam_formal_xai/results';
datasets = {'breast_cancer','credit','heloc'}; % ,'credit','heloc'

% plotting type
type = 'explainSize';
% type = 'numFeatures';

% discretization on x-axis
maxTimeStep = 800;
numSteps = maxTimeStep;
timeSteps = linspace(0,maxTimeStep,numSteps);

    
% which?
single = false;
if single
    methods = {'main'}; 
    epsilons = {
        [0.01,0.1,0.2,0.5];
        [0.01,0.1,0.2,0.5];
        [0.01,0.1,0.2,0.5];
    };
else
    epsilons = {0.01,0.5,0.5};
    methods = {'main_fast','main_unsound'}; % ,'local_naive','local_sensitivity','main_gpu_30', 'main' (old)
end

% set up figure
figure;

% prepare CORAtable
config = {
    % column heading
    'Dataset','Epsilon','#','#','Method','Size','Time [s]','T/O [s]'; ...
    % show column?
    true,       false,  true,true, true,    true,   true,      false; ...
    % format
    's','.2f','i','i','s','sum{%6.2f & %6.2f}','sum{%6.2f & %6.2f}','i'; ...
    % column width
    20,0,3,3,20,0,0,0
};
showColumn = [config{2,:}];
table = CORAtable("latex",config(1,showColumn),config(3,showColumn),"ColumnWidths",[config{4,showColumn}]);
table.printHeader();

% iterate through all datasets
for d=1:numel(datasets)
    dataset = datasets{d};
    resultdir_d = sprintf('%s/main-%s-yours-local-naive',resultdir,dataset);
    
    % set up subplot
    subplot(1,numel(datasets),d); hold on;
    legend; title(strrep(dataset,'_','\_'))
    xlabel('Time [s]'); 
    if d == 1
        % show ylabel only for first subplot
        switch type
            case 'explainSize'
                ylabel('Explanation size')
            case 'numFeatures'
                ylabel('#Processed features [%]')
        otherwise
            error('Unknown plotting type: %s', type)
        end
    end
    
    % run experiments
    for e=1:numel(epsilons{d})
        epsilon = epsilons{d}(e);

        idxValid = [];
        
        for i = 1:numel(methods)
            method = methods{i};
            name = strrep(method,'_','\_');
        
            % find json files within folder
            jsonFiles = dir(fullfile(sprintf('%s/final_%s_big_model_exp_logs_%s',resultdir_d,dataset,method), sprintf('*%g.json',epsilon)));
            numJsonFiles = numel(jsonFiles);
            if ~isempty(idxValid)
                jsonFiles = jsonFiles(idxValid);
            end

            % plot
            color = CORAcolor(i);
            [finalExplainSizes,totalTimes,timeouts,idxValid_i] = nam_plot_explanation_over_time(type,jsonFiles, timeSteps, name, color);
        
            if isempty(idxValid)
                % idxValid = idxValid_i;
            end

            % print table
            content = {dataset,epsilon,numel(finalExplainSizes),numJsonFiles,name,finalExplainSizes,totalTimes,mean(timeouts)};
            table.printContentRow(content(showColumn))
        end
    end

    % plot mid row of table
    if d~=numel(datasets)
        table.printMidBoundaryRow();
    end
end

% finish table
table.printFooter();



end