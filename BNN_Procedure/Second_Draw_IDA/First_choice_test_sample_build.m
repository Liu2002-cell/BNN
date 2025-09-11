% 加载数据
load('X_and_Y_data_orig_6137.mat', 'datas'); 
load('X_norm_stats.mat', 'mu_all', 'sigma_all');

% 获取测试集数据
nRows = height(datas);
last10pct_rows = round(nRows * 0.85) + 1 : nRows;
data_last15pct = datas(last10pct_rows, :);

% 获取唯一 ID
unique_ids = unique(data_last15pct{:,1});

rng('shuffle'); 
selected_ids = datasample(unique_ids, 10, 'Replace', false);

% 用于存储结果
Build_orig = struct();
X_test_orig = struct();
X_test_std = struct();
Y_test_orig = struct();

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

% 开始处理每个 ID
for i = 1:length(selected_ids)
    id_cell = selected_ids(i);       
    id_str = id_cell{1};          
    
    % 使用字符串比较筛选数据
    id_mask = strcmp(data_last15pct{:,1}, id_str);
    id_data = data_last15pct(id_mask, :);
    id_data = id_data(1:min(110, height(id_data)), :);
    
    % 使用字符串ID命名结构体字段
    build_name = sprintf('Build_orig_%s', id_str);
    Build_orig.(build_name) = id_data;
    
    % 构造X_test_orig
    x_data = id_data{:, 2:end-3};
    X_test_orig.(sprintf('X_test_orig_%s', id_str)) = x_data;
    
    % 构造Y_test_orig
    y_data = id_data{:, end-2:end};
    Y_test_orig.(sprintf('Y_test_orig_%s', id_str)) = y_data;

    % 对数转换 + 标准化（构造 X_test_std）
    x_std = x_data;

    for col = 1:size(x_data,2)
        col_global_idx = col ; 
        apply_log = ismember(col_global_idx, log_cols);
        skip_std = ismember(col, [2,3,4]);

        % 如果是对数列，先取对数（且不对0值取 log）
        if apply_log
            non_zero_idx = x_std(:, col) > 0;
            x_std(non_zero_idx, col) = log(x_std(non_zero_idx, col));
        end

        % 标准化（0值不处理）
        if ~skip_std
            mu = mu_all(col_global_idx);
            sigma = sigma_all(col_global_idx);
            non_zero = x_std(:,col) ~= 0;
            x_std(non_zero, col) = (x_std(non_zero, col) - mu) / sigma;
        end
    end

    X_test_std.(sprintf('X_test_std_%s', id_str)) = x_std;
end

% 保存所有结构体
save('Build_orig.mat', '-struct', 'Build_orig');
save('X_test_orig.mat', '-struct', 'X_test_orig');
save('Y_test_orig.mat', '-struct', 'Y_test_orig');
save('X_test_std.mat', '-struct', 'X_test_std');
