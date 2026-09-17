function nam_analyze_cpu()

% settings
resultdir = './nam_formal_xai/results';

dataset = 'credit';
resultdir_d = sprintf('%s/diff-cpus',resultdir);

% define time span
timeout = 600;
timeSteps = linspace(0,timeout,100);

figure; hold on;
legend; title('Comparison #CPUs')
xlabel('Time [s]'); ylabel('Explanation size')

cpus = [1,2,8,32];
for i = 1:numel(cpus)

    % find json files within folder
    jsonFiles = dir(fullfile(sprintf('%s/%s-diff-cpus/new_method_v2_exp_logs_%s_main_%i_processors',resultdir_d,dataset,dataset,cpus(i)), '*.json'));

    % plot
    name = sprintf('#CPUS=%i',cpus(i));
    color = CORAcolor(i);
    nam_plot_explanation_over_time('explainSize',jsonFiles, timeSteps, name, color);
end



end