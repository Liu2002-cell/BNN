% ----------------------------
% Bayesian Neural Network for Structural Response Prediction
% ----------------------------

%% 1. Load and preprocess data
load('X_train_std.mat'); 
load('X_val_std.mat');
load('X_test_std.mat');
load('Y_train_orig.mat');
load('Y_val_orig.mat');
load('Y_test_orig.mat');

X_train = X_train_std;  
X_val = X_val_std;      
X_test = X_test_std;    
Y_train = log(Y_train_orig(:, 2)); 
Y_val = log(Y_val_orig(:, 2));     
Y_test = Y_test_orig(:, 2);        

% 准备数据存储
dsXTrain = arrayDatastore(X_train); 
dsYTrain = arrayDatastore(Y_train);
dsTrain = combine(dsXTrain, dsYTrain);

dsXVal = arrayDatastore(X_val);
dsYVal = arrayDatastore(Y_val);
dsVal = combine(dsXVal, dsYVal);

dsXTest = arrayDatastore(X_test);
dsYTest = arrayDatastore(Y_test);
dsTest = combine(dsXTest, dsYTest);

%% 3. 构建 Bayesian 网络结构（使用 ELBO 框架）
numFeatures = size(X_train, 2);
numResponses = size(Y_train, 2);
numObservations = size(X_train, 1);
inputSize = [numFeatures, 1, 1];
outputSize = numResponses;
sigma1 = 1.5;
sigma2 = 0.1;

layers = [
    featureInputLayer(numFeatures)
%     bayesFullyConnectedLayer(256, Sigma1=sigma1, Sigma2=sigma2)
%     reluLayer
    bayesFullyConnectedLayer(128, Sigma1=sigma1, Sigma2=sigma2)
    reluLayer
    bayesFullyConnectedLayer(64, Sigma1=sigma1, Sigma2=sigma2)
    reluLayer
    bayesFullyConnectedLayer(numResponses, Sigma1=sigma1, Sigma2=sigma2)];
net = dlnetwork(layers);
net = dlupdate(@gpuArray, net);
analyzeNetwork(net)
%学习率与训练设置
samplingNoise = dlarray(1);
samplingNoise = gpuArray(samplingNoise);
doLearnPrior = true;
priorLearnRate = 0.001;
numLearnables = size(net.Learnables, 1);

for i = 1:numLearnables
    layerName = net.Learnables.Layer(i);
    paramName = net.Learnables.Parameter(i);
    if paramName == "Sigma1" || paramName == "Sigma2"
        if doLearnPrior
            net = setLearnRateFactor(net, layerName, paramName, priorLearnRate);
        else
            net = setLearnRateFactor(net, layerName, paramName, 0);
        end
    end
end

%% 4. 准备训练参数
numEpochs = 100000;
miniBatchSize =4096;
numSamplesForavgELBO = 50; 
averageLossComputationFrequency = 50; 
mbq = minibatchqueue(dsTrain, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat=["CB" "CB"]);
trailingAvg = [];
trailingAvgSq = [];
trailingAvgNoise = [];
trailingAvgNoiseSq = [];
numIterationsPerEpoch = ceil(numObservations/miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
monitor = trainingProgressMonitor( ...
    Metrics=["TrainRMSE", "ValRMSE", "AverageELBOLoss", "LogLikelihood", "KL"], ...
    Info="Epoch", ...
    XLabel="Iteration");
bestValRmse = inf;
bestNet = net;
iteration = 0;
epoch = 0;

%% 5. 训练循环
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    miniBatchIdx = 0;
    shuffle(mbq);
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;
        miniBatchIdx = miniBatchIdx + 1;
        [X, T] = next(mbq); 
        X = gpuArray(X);
        T = gpuArray(T);
        %% 向前传播+ELBO损失
        [elboLoss, trainRmse, gradientsNet, gradientsNoise,l,KL] = dlfeval(@modelLoss, ...
            net, X, T, samplingNoise, miniBatchIdx, numIterationsPerEpoch);
        %% 反向传播+使用 Adam 优化器更新网络参数
        [net, trailingAvg, trailingAvgSq] = adamupdate(net, gradientsNet, ...
            trailingAvg, trailingAvgSq, iteration);
        % 更新采样噪声
        [samplingNoise, trailingAvgNoise, trailingAvgNoiseSq] = adamupdate(samplingNoise, ...
            gradientsNoise, trailingAvgNoise, trailingAvgNoiseSq, iteration);
        %% 计算验证集RMSE
        valRmse = calculateValRMSE(net, X_val, Y_val, samplingNoise);
        % 记录训练集和验证集RMSE
        recordMetrics(monitor, iteration, TrainRMSE=double(trainRmse), ValRMSE=double(valRmse), LogLikelihood=double(l), ...
    KL=double(KL))
        % 保存最优网络（基于验证集RMSE）
        if valRmse < bestValRmse
            bestValRmse = valRmse;
            bestNet = net;
        end
        % 记录平均 ELBO 损失
        if mod(iteration, averageLossComputationFrequency) == 0
            avgELBOLoss = averageNegativeELBO(net, X, T, samplingNoise, miniBatchIdx, ...
                numIterationsPerEpoch, numSamplesForavgELBO);
            recordMetrics(monitor, iteration, AverageELBOLoss=double(avgELBOLoss))
        end
        % 更新 epoch 和进度
        updateInfo(monitor, Epoch=string(epoch) + " of " + string(numEpochs));
        monitor.Progress = 100 * (iteration / numIterations);
    end
end

%% 6. 保存模型和训练状态
save('BNN_model_MaxAbsAccel_orig_6137_random.mat', 'bestNet', 'samplingNoise','trailingAvg','trailingAvgSq',...
'trailingAvgNoise','trailingAvgNoiseSq','bestValRmse','numEpochs','miniBatchSize','numSamplesForavgELBO',...
'averageLossComputationFrequency','sigma1','sigma2','priorLearnRate','doLearnPrior','iteration', 'epoch');

%% 7. 使用测试集评估神经网络模型性能（计算RMSE和置信区间覆盖率）
load('BNN_model_MaxAbsAccel_orig_6137_random.mat');
X_test_dl = dlarray(X_test', 'CB');
Y_test_dl = dlarray(Y_test', 'CB');
X_test_dl = gpuArray(X_test_dl);
Y_test_dl = gpuArray(Y_test_dl);
% 采样预测：生成多个后验样本
numSamples = 1000;
predSamples = modelPosteriorSample(bestNet, X_test_dl, samplingNoise, numSamples); 
predS = exp(predSamples);
predSamples = squeeze(predS);
% 预测均值
Y_pred_mean = mean(predSamples, 1);
rmse_test = sqrt(mean((Y_pred_mean - extractdata(Y_test_dl)).^2));
fprintf('测试集 RMSE: %.4f\n', rmse_test);

% 计算置信区间覆盖率
Y_test_true = extractdata(Y_test_dl);
% 定义置信区间概率
intervals = [0.95, 0.70, 0.50];
coverageValues = zeros(1, length(intervals));
for i = 1:length(intervals)
    alpha = (1 - intervals(i)) / 2;
    lower = quantile(predSamples, alpha, 1);
    upper = quantile(predSamples, 1 - alpha, 1);
    coverage = mean((Y_test_true >= lower) & (Y_test_true <= upper));
    coverageValues(i) = coverage;
    fprintf('%.0f%% 置信区间覆盖率: %.2f%%\n', intervals(i)*100, coverage*100);
end

%% 辅助函数
% 模型损失函数
function [elboLoss, meanError, gradientsNet, gradientsNoise,l,KL] = modelLoss(net, X, T, samplingNoise, miniBatchIdx, numBatches)
    [elboLoss, Y,l,KL] = negativeELBO(net, X, T, samplingNoise, miniBatchIdx, numBatches);
    [gradientsNet, gradientsNoise] = dlgradient(elboLoss, net.Learnables, samplingNoise);
    meanError = double(sqrt(mse(Y, T)));
end
% 向前传播 + 证据下限 （ELBO） 损失函数
function [elboLoss, Y,l, KL ] = negativeELBO(net, X, T, samplingNoise, miniBatchIdx, numBatches)
    [Y, state] = forward(net, X, Acceleration="auto");
    beta = KLWeight(miniBatchIdx, numBatches);
    logPosterior = state.Value(state.Parameter == "LogPosterior");
    logPosterior = sum([logPosterior{:}]);
    logPrior = state.Value(state.Parameter == "LogPrior");
    logPrior = sum([logPrior{:}]);
    KL = logPosterior - logPrior;
    l = logLikelihood(Y, T, samplingNoise);
    elboLoss = (-1*l) + KL*beta;
end
% Mini-Batch 预处理功能
function [X, T] = preprocessMiniBatch(dataX, dataY)
    X = cat(1, dataX{:})';
    T = cat(1, dataY{:})';
end
% 模型预测函数
function predictions = modelPosteriorSample(net, X, samplingNoise, numSamples)
    predictions = zeros(numSamples, 1, size(X, 2));  
    for i = 1:numSamples
        Y = predict(net, X, Acceleration="none");
        sigmaY = exp(samplingNoise);
        predictions(i,:,:) = Y + sigmaY .* randn(size(Y)); 
    end
end
% 最大似然估计函数
function l = logLikelihood(Y, T, samplingNoise)
    sigmaY = exp(samplingNoise);
    l = sum(logProbabilityNormal(T, Y, sigmaY), "all");
end
% 小批量和 KL 再称重
function beta = KLWeight(i, m)
    beta = 2^(m - i)/(2^m - 1);
end
% 平均 ELBO 损失
function avgELBO = averageNegativeELBO(net,X,T,samplingNoise,miniBatchIdx,numBatches,numSamples)
    avgELBO = 0;
    for i = 1:numSamples
            [ELBO, ~] = negativeELBO(net, X, T, samplingNoise, miniBatchIdx, numBatches);
        avgELBO = avgELBO + ELBO;
    end
    avgELBO = avgELBO / numSamples;
end
% 计算验证集RMSE
function valRmse = calculateValRMSE(net, X_val, Y_val, samplingNoise)
    X_val_dl = dlarray(X_val', 'CB');
    Y_val_dl = dlarray(Y_val', 'CB');
    X_val_dl = gpuArray(X_val_dl);
    Y_val_dl = gpuArray(Y_val_dl);
    numSamples = 10;
    preds = modelPosteriorSample(net, X_val_dl, samplingNoise, numSamples);
    Y_pred_mean = squeeze(mean(preds, 1));
    Y_pred_mean_dl = dlarray(Y_pred_mean, 'BC');
    valRmse = double(sqrt(mse(Y_pred_mean_dl, Y_val_dl)));
end

