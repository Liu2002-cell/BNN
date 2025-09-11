%% 得到BNN_dataset.mat:包含X_labels、Y_labels、X_all、Y_all、BldID_all。用于生成之后BNN训练所需的总数据集

% 配置
bld_dir = 'D:\BNN_Bld_strpm';
yld_dir = 'D:\IDA_results';
bld_files = dir(fullfile(bld_dir, 'BNN_Bld_strpm_*.csv'));
n_files = numel(bld_files);

% 初始化 matfile
matObj = matfile('BNN_dataset_6137.mat', 'Writable', true);
cnt = 1;

% 缓存设置
batch_size = 100000;
X_batch = [];
Y_batch = [];
BldID_batch = {};

h = waitbar(0, 'Processing...');
% X 标签初始化
X_labels = {
    'DampingRatio', ...
    'Hyster_ModifiedClough', 'Hyster_Pinching', 'Hyster_KinematicHardening', ...
    'Tao_val', ...
    'StoryHeight', 'T1', 'Cs'};

% 层参数标签
layer_names = {'FloorMass', 'ElasticStiffness', 'DesignShear', 'DesignDisp', ...
               'YieldShear', 'YieldDisp', 'UltimateShear', 'UltimateDisp', 'CompleteDamageDisp'};
for i = 1:52
    for j = 1:numel(layer_names)
        X_labels{end+1} = sprintf('%d_%s', i, layer_names{j});
    end
end

% 添加IM参数标签
X_labels{end+1} = 'IM';

% Y 标签
Y_labels = {'MaxDrift', 'MaxAbsAccel', 'ResDrift'};

% 循环读取数据
for k = 1:n_files
    waitbar(k / n_files, h);

    % 读取结构参数
    fname = bld_files(k).name;
    toks = regexp(fname, 'BNN_Bld_strpm_(\d+)', 'tokens');
    bldID = toks{1}{1};
    row = readtable(fullfile(bld_dir, fname), 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');
    row = row(1, :);

    % 特征提取
    dampingratio = double(row{1, 1});
    hyster_type = string(row{1, 2});
    types = {'Modified-Clough', 'Pinching', 'Kinematic hardening'};
    onehot = double(strcmp(hyster_type, types));
    tao_str = strtrim(string(row{1, 3}));
    tao_val = 0;
    if tao_str ~= "[]"
        val = str2double(tao_str);
        if ~isnan(val)
            tao_val = val;
        end
    end
    vals_4_6 = double(row{1, 4:6});
    feat_overall = [dampingratio onehot tao_val vals_4_6];

    % 层参数
    layer_data = table2array(row(:, 7:end));
    n_layer = size(layer_data, 2) / 9;
    feat_layers = reshape(layer_data, 9, n_layer)';
    feat_layers_flat = reshape(feat_layers', 1, []);

    % 响应数据
    resp_file = fullfile(yld_dir, sprintf('IDA_result_ReprBldID_%s.csv', bldID));
    if ~isfile(resp_file)
        continue
    end

    opts = detectImportOptions(resp_file);
    opts.DataLines = [2 111];  
    resp_tbl = readtable(resp_file, opts);
    IMs = resp_tbl{:, 1};

    % 提取第3列和第4列的最大值，第6列为直接数值
    col3_raw = resp_tbl{:, 3};
    col4_raw = resp_tbl{:, 4};
    col6_raw = resp_tbl{:, 6};

    n_rows = size(resp_tbl, 1);
    Y_vals = zeros(n_rows, 3); 

    % 工具函数：提取字符串中的最大数值
    extract_max = @(val) ...
        max(sscanf(regexprep(strrep(strrep(char(val), '[', ''), ']', ''), '\s+', ' '), '%f'));

    for r = 1:n_rows
        % 第3列
        val3 = col3_raw(r);
        if iscell(val3), val3 = val3{1}; end
        if ischar(val3) || isstring(val3)
            Y_vals(r, 1) = extract_max(val3);
        else
            Y_vals(r, 1) = NaN;
        end

        % 第4列
        val4 = col4_raw(r);
        if iscell(val4), val4 = val4{1}; end
        if ischar(val4) || isstring(val4)
            Y_vals(r, 2) = extract_max(val4);
        else
            Y_vals(r, 2) = NaN;
        end

        % 第6列（直接使用）
        Y_vals(r, 3) = double(col6_raw(r));
    end

    for i = 1:length(IMs)
        x = single([feat_overall feat_layers_flat IMs(i)]);
        y = single(Y_vals(i, :));

        X_batch(end+1, :) = x;
        Y_batch(end+1, :) = y;
        BldID_batch{end+1, 1} = bldID;

        % 写入批次
        if size(X_batch, 1) >= batch_size
            row_end = cnt + size(X_batch, 1) - 1;
            matObj.X_all(cnt:row_end, 1:size(X_batch,2)) = X_batch;
            matObj.Y_all(cnt:row_end, 1:size(Y_batch,2)) = Y_batch;
            matObj.BldID_all(cnt:row_end, 1) = BldID_batch;

            cnt = row_end + 1;
            X_batch = []; Y_batch = [];BldID_batch = {};
        end
    end
end
close(h);

% 写入最后一批
if ~isempty(X_batch)
    row_end = cnt + size(X_batch, 1) - 1;
    matObj.X_all(cnt:row_end, 1:size(X_batch,2)) = X_batch;
    matObj.Y_all(cnt:row_end, 1:size(Y_batch,2)) = Y_batch;
    matObj.BldID_all(cnt:row_end, 1) = BldID_batch; 
    cnt = row_end + 1;
end

matObj.valid_rows = cnt - 1;
matObj.X_labels = X_labels;
matObj.Y_labels = Y_labels;
