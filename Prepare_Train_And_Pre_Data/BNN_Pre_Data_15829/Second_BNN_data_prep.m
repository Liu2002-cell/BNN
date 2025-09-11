%% 得到BNN_dataset.mat。用于之后BNN训练效果验证所需的数据集

% ========= 配置路径 =========
bld_dir = 'D:\BNN_Bld_strpm';
yld_dir = 'D:\RegionalLossSimulations';
bld_files = dir(fullfile(bld_dir, 'BNN_Bld_strpm_*.csv'));
n_files = numel(bld_files);

% ========= 初始化 matfile =========
matObj = matfile('BNN_dataset_15829.mat', 'Writable', true);
cnt = 1;
batch_size = 1000000;
X_batch = cell(batch_size, 1);
Y_batch = cell(batch_size, 1); 
BldID_batch = cell(batch_size, 1);
ptr = 1;

% ========= 定义标签 =========
X_labels = {
    'DampingRatio', ...
    'Hyster_ModifiedClough', 'Hyster_Pinching', 'Hyster_KinematicHardening', ...
    'Tao_val', ...
    'StoryHeight', 'T1', 'Cs'};
layer_names = {'FloorMass', 'ElasticStiffness', 'DesignShear', 'DesignDisp', ...
               'YieldShear', 'YieldDisp', 'UltimateShear', 'UltimateDisp', 'CompleteDamageDisp'};
for i = 1:52
    for j = 1:numel(layer_names)
        X_labels{end+1} = sprintf('%d_%s', i, layer_names{j});
    end
end
X_labels{end+1} = 'IM';

% ========= 提前初始化维度 =========
% 读取第一个建筑用于计算特征维度
row0 = readtable(fullfile(bld_dir, bld_files(1).name), 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');
dampingratio = double(row0{1, 1});
types = {'Modified-Clough', 'Pinching', 'Kinematic hardening'};
hyster_onehot = double(strcmp(string(row0{1, 2}), types));
tao_str = strtrim(string(row0{1, 3}));
tao_val = str2double(tao_str); if isnan(tao_val), tao_val = 0; end
vals_4_6 = double(row0{1, 4:6});
feat_overall = [dampingratio hyster_onehot tao_val vals_4_6];
layer_data = table2array(row0(:, 7:end));
feat_layers_flat = reshape(layer_data, 1, []);
temp_x = single([feat_overall feat_layers_flat 0]); 

feat_dim = numel(temp_x); 
matObj.X_all(1, feat_dim) = single(0); 
matObj.BldID_all(1, 1) = {''};
matObj.Y_all(1, 4) = single(0);

% ========= 主处理循环 =========
h = waitbar(0, 'Processing...');
for k = 1:n_files
    waitbar(k / n_files, h);

    % 获取建筑ID
    fname = bld_files(k).name;
    toks = regexp(fname, 'BNN_Bld_strpm_(\d+)', 'tokens');
    bldID = toks{1}{1};
    bldID_num = single(str2double(bldID));
    % 读取结构数据
    row = readtable(fullfile(bld_dir, fname), 'ReadVariableNames', true, 'VariableNamingRule', 'preserve');
    row = row(1, :);
    % 特征提取
    dampingratio = double(row{1, 1});
    hyster_type = string(row{1, 2});
    hyster_onehot = double(strcmp(hyster_type, types));
    tao_str = strtrim(string(row{1, 3}));
    tao_val = str2double(tao_str); if isnan(tao_val), tao_val = 0; end
    vals_4_6 = double(row{1, 4:6});
    feat_overall = [dampingratio hyster_onehot tao_val vals_4_6];

    layer_data = table2array(row(:, 7:end));
    feat_layers_flat = reshape(layer_data, 1, []);

    % 读取响应数据
    resp_file = fullfile(yld_dir, sprintf('SimEDP_bldID_%s.csv', bldID));
    if ~isfile(resp_file), continue; end

    opts = detectImportOptions(resp_file);
    opts.DataLines = [2 501];
    resp_tbl = readtable(resp_file, opts);
    
    % 修改：提取Y值（建筑ID + 第3、4、5列）
    n_resp = size(resp_tbl, 1);  % 实际响应数量
    Y_vals = single([repmat(bldID_num, n_resp, 1), ... % 第一列：建筑ID
                     resp_tbl{:, 3}, ...               % 第二列：原第三列
                     resp_tbl{:, 4}, ...               % 第三列：原第四列
                     resp_tbl{:, 5}]);  
    IMs = resp_tbl{:, 2};  % 只取IM值

    n_resp = length(IMs);
    x_mat = single([repmat(feat_overall, n_resp, 1), ...
                    repmat(feat_layers_flat, n_resp, 1), ...
                    IMs]);

    % 填入批次缓存
    for j = 1:n_resp
        X_batch{ptr, 1} = x_mat(j, :);
        Y_batch{ptr, 1} = Y_vals(j, :);
        BldID_batch{ptr, 1} = bldID;
        ptr = ptr + 1;

        % 到达批次上限，写入
        if ptr > batch_size
            row_end = cnt + batch_size - 1;
            matObj.X_all(cnt:row_end, :) = cell2mat(X_batch);
            matObj.Y_all(cnt:row_end, :) = cell2mat(Y_batch);
            matObj.BldID_all(cnt:row_end, 1) = BldID_batch;
            cnt = row_end + 1;
            ptr = 1;
        end
    end
end
close(h);

% ========= 写入剩余数据 =========
if ptr > 1
    row_end = cnt + ptr - 2;
    matObj.X_all(cnt:row_end, :) = cell2mat(X_batch(1:ptr-1));
    matObj.Y_all(cnt:row_end, :) = cell2mat(Y_batch(1:ptr-1));
    matObj.BldID_all(cnt:row_end, 1) = BldID_batch(1:ptr-1);
    cnt = row_end + 1;
end

matObj.valid_rows = cnt - 1;
matObj.X_labels = X_labels;