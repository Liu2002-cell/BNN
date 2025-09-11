%% 生成X_and_Y_data_Orig_6137.mat，作为BNN模型 训练+验证+测试 的总数据集

load('BNN_dataset_6137.mat', 'X_all', 'Y_all', 'BldID_all', 'X_labels', 'Y_labels');

% ====== 处理 X_all ======
X_tbl = array2table(X_all, 'VariableNames', X_labels);
X_tbl = addvars(X_tbl, BldID_all, 'Before', 1, 'NewVariableNames', 'ID');

% ====== 处理 Y_all ======
Y_tbl = array2table(Y_all, 'VariableNames', Y_labels);
Y_tbl = addvars(Y_tbl, BldID_all, 'Before', 1, 'NewVariableNames', 'ID');

% ====== 合并 X_tbl 和 Y_tbl 到 datas ======
if height(X_tbl) ~= height(Y_tbl)
    error('X_tbl 和 Y_tbl 的行数不匹配，无法合并。');
end

% 合并表格
datas = [X_tbl, Y_tbl(:,2:end)];
save('X_and_Y_data_Orig_6137.mat', 'datas');
