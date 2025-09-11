% 加载测试数据和模型
load('X_test_std.mat'); 
load('BNN_model_MaxDrift_orig_6137_random.mat'); 

% 获取X_test_std_*变量名
vars = who('-file', 'X_test_std.mat');
x_vars = vars(contains(vars, 'X_test_std_'));

numSamples = 1000; % 采样数量

Y_test_pre_struct = struct(); % 用于保存所有预测结果

for k = 1:length(x_vars)
    varname = x_vars{k};
    X_test_std = load('X_test_std.mat', varname);
    X_data = X_test_std.(varname);  

    % 取前11行作为模型输入，转换为dlarray格式
    X_input = dlarray(X_data(1:11, :)', 'CB'); 

    % 预测
    predSamples = modelPosteriorSample(bestNet, X_input, samplingNoise, numSamples); 
    preds = squeeze(predSamples); 

    % 预测均值
    Y_pred_log_mean = mean(preds, 1); 
    Y_pred_log_std  = std(preds, 0, 1);

    Y_pred_mean=exp(Y_pred_log_mean);
    Y_pred_84=exp(Y_pred_log_mean+Y_pred_log_std);
    Y_pred_16=exp(Y_pred_log_mean-Y_pred_log_std);

    idx_str = regexp(varname, '\d+', 'match', 'once');
    y_varname = ['Y_test_pre_' idx_str];

Y_pred_combined = [double(Y_pred_mean(:)), double(Y_pred_84(:)),double(Y_pred_16(:))];

Y_test_pre_struct.(['Y_test_pre_' idx_str]) = Y_pred_combined;

end

save('Y_test_pre.mat', '-struct', 'Y_test_pre_struct');

%% 定义预测函数
function predictions = modelPosteriorSample(net, X, samplingNoise, numSamples)
    predictions = zeros(numSamples, 1, size(X, 2));  
    for i = 1:numSamples
        Y = predict(net, X, 'Acceleration', 'none');
        sigmaY = exp(samplingNoise);
        predictions(i,:,:) = Y + sigmaY .* randn(size(Y)); 
    end
end