%% ----------------------------
% BNN 分批预测（优化内存版）
% ----------------------------
%% 1. 加载数据（仅加载一次）
load('BNN_dataset_15829.mat');
load('X_norm_stats.mat');
mu_all = single(mu_all);
sigma_all = single(sigma_all);
[N, D] = size(X_all);

%% 2. 对数变换列（预定义列索引）
log_idx = [... 
    9,10,11,13,15,18,19,20,22,24,27,28,29,31,33,36,37,38,40,42,45,46,47,49,51,54,55,56,58,60,...
    63,64,65,67,69,72,73,74,76,78,81,82,83,85,87,90,91,92,94,96,99,100,101,103,105,108,109,...
    110,112,114,117,118,119,121,123,126,127,128,130,132,135,136,137,139,141,144,145,146,148,...
    150,153,154,155,157,159,162,163,164,166,168,171,172,173,175,177,180,181,182,184,186,189,...
    190,191,193,195,198,199,200,202,204,207,208,209,211,213,216,217,218,220,222,225,226,227,...
    229,231,234,235,236,238,240,243,244,245,247,249,252,253,254,256,258,261,262,263,265,267,...
    270,271,272,274,276,279,280,281,283,285,288,289,290,292,294,297,298,299,301,303,306,307,...
    308,310,312,315,316,317,319,321,324,325,326,328,330,333,334,335,337,339,342,343,344,346,...
    348,351,352,353,355,357,360,361,362,364,366,369,370,371,373,375,378,379,380,382,384,387,...
    388,389,391,393,396,397,398,400,402,405,406,407,409,411,414,415,416,418,420,423,424,425,...
    427,429,432,433,434,436,438,441,442,443,445,447,450,451,452,454,456,459,460,461,463,465,...
    468,469,470,472,474];
%% 3. 预测参数设置
batchSize = 16384;
numSamples = 100;
model_files = {'BNN_model_MaxDrift_orig_6137_random.mat', 'BNN_model_MaxAbsAccel_orig_6137_random.mat', 'BNN_model_ResDrift_orig_6137_random.mat'};
param_names = {'MaxDrift', 'MaxAbsAccel', 'ResDrift'};

Y_pred = zeros(N, 3, 'single');
%% 4. 主预测循环（无文件写入）
for k = 1:3
    fprintf('\n=== 正在预测 %s ===\n', param_names{k});
    tmp = load(model_files{k}, 'bestNet', 'samplingNoise');
    bestNet = tmp.bestNet;
    samplingNoise = single(tmp.samplingNoise);
    
    for b = 1:ceil(N / batchSize)
        idxStart = (b - 1) * batchSize + 1;
        idxEnd = min(b * batchSize, N);
        X_batch = X_all(idxStart:idxEnd, :);

        for i = log_idx
            col = X_batch(:, i);
            X_batch(col ~= 0, i) = log(col(col ~= 0));
        end

        for j = 1:D
            if j == 2 || j == 3 || j == 4  
                continue;
            end
            col = X_batch(:, j);
            if j == 5 || j >= 9
                mask = col ~= 0;
                col(mask) = (col(mask) - mu_all(j)) / sigma_all(j);
            else
                col = (col - mu_all(j)) / sigma_all(j);
            end
            X_batch(:, j) = col;
        end
       
        X_dl = dlarray(X_batch', 'CB');
        predSamples = modelPosteriorSample(bestNet, X_dl, samplingNoise, numSamples); % Numofsample,1,batchsize
        predS = exp(predSamples);
        predSamples = squeeze(predS);
        Y_pred_mean = mean(predSamples, 1);
        Y_pred(idxStart:idxEnd, k)=Y_pred_mean';        
    end
end

%% 5. 转换建筑ID格式
if iscell(BldID_all)
    try
        BldID_all = cell2mat(BldID_all);
    catch
        BldID_all = str2double(BldID_all);
    end
end
BldID_all = single(BldID_all);

%% 6. 构建并保存最终表格
T = table(BldID_all, ...
          double(Y_pred(:,1)), ...
          double(Y_pred(:,2)), ...
          double(Y_pred(:,3)), ...
          'VariableNames', {'ID', 'MaxDrift', 'MaxAbsAccel', 'ResDrift'});
save('BNN_pre_table_orig_15829.mat', 'T');  % 唯一保存操作

%% ========= 辅助函数 =========
function predictions = modelPosteriorSample(net, X, samplingNoise, numSamples)
    predictions = zeros(numSamples, 1, size(X, 2));  
    for i = 1:numSamples
        Y = predict(net, X, Acceleration="none");
        sigmaY = exp(samplingNoise);
        predictions(i,:,:) = Y + sigmaY .* randn(size(Y)); 
    end
end