% 加载三个文件中的所有变量  
x_data = load('X_test_orig.mat');
y_orig = load('Y_test_orig.mat');
y_pred = load('Y_test_pre.mat');

% 指定需要提取的列索引
column_index = 1;

% 获取字段名
x_fields = fieldnames(x_data);
num_buildings = length(x_fields);

% 创建图窗口
figure('Name', 'Drift Ratio vs IM 曲线对比', 'NumberTitle', 'off');

% 遍历每个建筑ID的数据
for i = 1:num_buildings
    % 当前建筑变量名
    x_name = x_fields{i};
    
    % 推导出对应的 y_orig 和 y_pred 名称
    id = regexp(x_name, '\d+', 'match'); 
    id_str = id{1};
    y_orig_name = ['Y_test_orig_' id_str];
    y_pred_name = ['Y_test_pre_' id_str];
    
    % 提取数据
    X = x_data.(x_name);     
    Y_true = y_orig.(y_orig_name)(:, column_index);     % 110×1
    Y_pre_full = y_pred.(y_pred_name); % 11×3 （中值，P84，P16）

    % 提取中值、P84、P16
    Y_pre_median = Y_pre_full(:, 1);   % 11×1
    Y_pre_p84 = Y_pre_full(:, 2);      % 11×1
    Y_pre_p16 = Y_pre_full(:, 3);      % 11×1
    
    % 提取最后一列的IM值
    IM = X(:, end);  % 110×1
    
    % 将数据分组，每11个为一组，总共10组
    N = floor(length(IM) / 11);  % 理论为10组
    group_IM = reshape(IM(1:N*11), 11, N);         % 11×N
    group_Ytrue = reshape(Y_true(1:N*11), 11, N);  % 11×N
    
    % 对真实值 group_Ytrue 做对数变换后的统计
    log_Ytrue = log(group_Ytrue);             % 11×N
    mean_logY = mean(log_Ytrue, 2);           % 每一行的均值 → 11×1
    std_logY = std(log_Ytrue, 0, 2);          % 每一行的标准差 → 11×1

    % 反对数转换回线性空间
    Ytrue_median = exp(mean_logY);            % 11×1 中值
    Ytrue_p84 = exp(mean_logY + std_logY);    % 11×1 +1σ
    Ytrue_p16 = exp(mean_logY - std_logY);    % 11×1 -1σ

    % 创建子图
    subplot(2, 5, i);
    hold on;
    
    % 画N根灰色真实曲线
    h1 = plot(group_Ytrue(:, 1), group_IM(:, 1), 'Color', [0.7 0.7 0.7]);  % Individual
    for j = 2:N
        plot(group_Ytrue(:, j), group_IM(:, j), 'Color', [0.7 0.7 0.7]);
    end

    % 画预测曲线（使用虚线）
    h2 = plot(Y_pre_median, IM(1:11), 'k--', 'LineWidth', 2);  % Predict Median
    h3 = plot(Y_pre_p84, IM(1:11), 'g--', 'LineWidth', 1.5);   % Predict 84%
    h4 = plot(Y_pre_p16, IM(1:11), 'b--', 'LineWidth', 1.5);   % Predict 16%

    % 画真实值的统计中值曲线（实线）
    h5 = plot(Ytrue_median, IM(1:11), 'k', 'LineWidth', 2);    % True Median
    h6 = plot(Ytrue_p84, IM(1:11), 'g', 'LineWidth', 1.5);     % True 84%
    h7 = plot(Ytrue_p16, IM(1:11), 'b', 'LineWidth', 1.5);     % True 16%

    % 图形设置
    xlabel('Drift Ratio');
    ylabel('IM');
    title(['Building ID: ' id_str]);
    grid on;
    box on;

    % 添加图例
    legend([h1 h2 h3 h4 h5 h6 h7], ...
        {'Individual', 'Predict Median', 'Predict 84%', 'Predict 16%', ...
         'True Median', 'True 84%', 'True 16%'}, ...
        'FontSize', 7, 'Location', 'southeast');
end

% 优化布局
sgtitle('Drift Ratio vs IM 对比图');
