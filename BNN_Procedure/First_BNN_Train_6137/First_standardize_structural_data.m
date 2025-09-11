%% ========== 加载数据并划分训练/验证/测试集（按顺序） ==========
load('X_and_Y_data_Orig_6137.mat');

X_all = datas{:, 2:end-3};
Y_all = datas{:, end-2:end};  

log_cols = [9,10,11,13,15,18,19,20,22,24,27,28,29,31,33,36,37,38,40,42,...
            45,46,47,49,51,54,55,56,58,60,63,64,65,67,69,72,73,74,76,78,...
            81,82,83,85,87,90,91,92,94,96,99,100,101,103,105,108,109,110,...
            112,114,117,118,119,121,123,126,127,128,130,132,135,136,137,139,...
            141,144,145,146,148,150,153,154,155,157,159,162,163,164,166,168,...
            171,172,173,175,177,180,181,182,184,186,189,190,191,193,195,198,...
            199,200,202,204,207,208,209,211,213,216,217,218,220,222,225,226,...
            227,229,231,234,235,236,238,240,243,244,245,247,249,252,253,254,...
            256,258,261,262,263,265,267,270,271,272,274,276,279,280,281,283,...
            285,288,289,290,292,294,297,298,299,301,303,306,307,308,310,312,...
            315,316,317,319,321,324,325,326,328,330,333,334,335,337,339,342,...
            343,344,346,348,351,352,353,355,357,360,361,362,364,366,369,370,...
            371,373,375,378,379,380,382,384,387,388,389,391,393,396,397,398,...
            400,402,405,406,407,409,411,414,415,416,418,420,423,424,425,427,...
            429,432,433,434,436,438,441,442,443,445,447,450,451,452,454,456,...
            459,460,461,463,465,468,469,470,472,474];

for col = log_cols
    non_zero_idx = X_all(:, col) > 0;
    X_all(non_zero_idx, col) = log(X_all(non_zero_idx, col));
end

N = size(X_all, 1);

rng(42);
shuffledIdx = randperm(N);
X_all = X_all(shuffledIdx, :); 
Y_all = Y_all(shuffledIdx, :);

% 划分索引
n_train = floor(0.7 * N);
n_val   = floor(0.15 * N);
n_test  = N - n_train - n_val; 

% 构造逻辑索引向量（基于随机顺序）
trainIdx = false(N, 1);
valIdx   = false(N, 1);
testIdx  = false(N, 1);

trainIdx(1:n_train) = true;
valIdx(n_train+1 : n_train+n_val) = true;
testIdx(n_train+n_val+1 : end) = true;

% 按索引划分数据集
X_train_orig = X_all(trainIdx, :);
X_val_orig   = X_all(valIdx, :);
X_test_orig  = X_all(testIdx, :);

Y_train_orig = Y_all(trainIdx, :);
Y_val_orig   = Y_all(valIdx, :);
Y_test_orig  = Y_all(testIdx, :);

% 获取特征维度
D = size(X_all, 2);

%% -------------------------------
% 1. 标准化整体参数
% 保留未标准化的列
cols_unprocessed = [2, 3, 4];
for c = cols_unprocessed
    X_train_std(:, c) = X_train_orig(:, c);
    X_val_std(:, c) = X_val_orig(:, c);
    X_test_std(:, c) = X_test_orig(:, c);
end

cols_to_norm = [1, 5, 6:8];

for c = cols_to_norm
    x_train = X_train_orig(:, c);
    
    if c == 5
        valid_idx_train = x_train ~= 0; 
        valid_idx_all = X_all(:, 5) ~= 0;
    else
        valid_idx_train = true(sum(trainIdx), 1);
        valid_idx_all = true(N, 1);
    end

    mu = mean(x_train(valid_idx_train));
    sigma = std(x_train(valid_idx_train));
    
    % 应用标准化到 train/val/test
    for subset = ["train", "val", "test"]
        switch subset
            case "train"
                idx = trainIdx;
                valid = valid_idx_train;
            case "val"
                idx = valIdx;
                valid = X_all(valIdx, 5) ~= 0;
            case "test"
                idx = testIdx;
                valid = X_all(testIdx, 5) ~= 0;
        end
        x = X_all(idx, c);
        x_norm = x;
        x_norm(valid) = (x(valid) - mu) / sigma;

        eval(sprintf('X_%s_std(:, c) = x_norm;', subset));
    end
end

%% -------------------------------
% 2. 标准化每层参数，只用训练集统计

col_start = 9;
col_end = D - 1;

mu = zeros(1, D);
sigma = zeros(1, D);

for j = col_start:col_end
    x_col = X_train_orig(:, j);
    non_zero_idx = x_col ~= 0;
    
    mu(j) = mean(x_col(non_zero_idx));
    sigma(j) = std(x_col(non_zero_idx));
    
    % 对train/val/test分别处理
    for subset = ["train", "val", "test"]
        switch subset
            case "train"
                x = X_train_orig(:, j);
            case "val"
                x = X_val_orig(:, j);
            case "test"
                x = X_test_orig(:, j);
        end
        x_std = x;
        nz = x ~= 0;

        if sigma(j) == 0
            x_std(nz) = x(nz); 
        else
            x_std(nz) = (x(nz) - mu(j)) / sigma(j);
        end
        
        eval(sprintf('X_%s_std(:, j) = x_std;', subset));
    end
end

%% -------------------------------
% 3. 标准化 IM（最后一列）
IM_col = D;
mu = mean(X_train_orig(:, IM_col));
sigma = std(X_train_orig(:, IM_col));

X_train_std(:, IM_col) = (X_train_orig(:, IM_col) - mu) / sigma;
X_val_std(:, IM_col)   = (X_val_orig(:, IM_col)   - mu) / sigma;
X_test_std(:, IM_col)  = (X_test_orig(:, IM_col)  - mu) / sigma;

%% -------------------------------
% 6. 保存标准化结果
save('X_all.mat', 'X_all');
save('X_train_orig.mat', 'X_train_orig');
save('X_val_orig.mat', 'X_val_orig');
save('X_test_orig.mat', 'X_test_orig');
save('X_train_std.mat', 'X_train_std');
save('X_val_std.mat', 'X_val_std');
save('X_test_std.mat', 'X_test_std');

save('Y_all.mat', 'Y_all');
save('Y_train_orig.mat', 'Y_train_orig');
save('Y_val_orig.mat', 'Y_val_orig');
save('Y_test_orig.mat', 'Y_test_orig');

%% ========== 保存所有列的均值和标准差 ==========
mu_all = zeros(1, D);
sigma_all = zeros(1, D);

cols_to_norm = [1, 5, 6:8];

for c = cols_to_norm
    x_train = X_train_orig(:, c);
    
    if c == 5
        valid_idx_train = x_train ~= 0;
    else
        valid_idx_train = true(sum(trainIdx), 1); 
    end

    mu_all(c) = mean(x_train(valid_idx_train));
    sigma_all(c) = std(x_train(valid_idx_train));
end

for j = 9:(D-1)
    x_col = X_train_orig(:, j);
    non_zero_idx = x_col ~= 0;
    mu_all(j) = mean(x_col(non_zero_idx));
    sigma_all(j) = std(x_col(non_zero_idx));
end

mu_all(D) = mean(X_train_orig(:, D));
sigma_all(D) = std(X_train_orig(:, D));

save('X_norm_stats.mat', 'mu_all', 'sigma_all');
