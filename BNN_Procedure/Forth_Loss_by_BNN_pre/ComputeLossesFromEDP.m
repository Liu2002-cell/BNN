function ComputeLossesFromEDP()

    load('BNN_pre_table_orig_15829.mat', 'T');
    load('ReprBld.mat', 'ReprBld', 'Bld2ReprBld');

    py_path = fullfile(pwd);
    if count(py.sys.path, py_path) == 0
        insert(py.sys.path, int32(0), py_path);
    end
    % 获取所有唯一建筑ID
    unique_bld_ids = unique(T.ID);
    num_buildings = numel(unique_bld_ids);
    % 准备建筑属性矩阵
    bld_props_matrix = zeros(num_buildings, 3); % [BldID, Stories, Area]
    structural_types = cell(num_buildings, 1);
    design_levels = cell(num_buildings, 1);
    occupancy_classes = cell(num_buildings, 1);
    for i = 1:num_buildings
        bld_id = unique_bld_ids(i);
        repr_idx = Bld2ReprBld{Bld2ReprBld{:,'Original Buildings index'} == bld_id, ...
                   'Representative Buildings index'};
        % 提取建筑属性
        bld_props_matrix(i, 1) = bld_id;
        bld_props_matrix(i, 2) = ReprBld{repr_idx+1, 'NumberOfStories'};
        bld_props_matrix(i, 3) = ReprBld{repr_idx+1, 'PlanArea'};
        structural_types{i} = ReprBld{repr_idx+1, 'StructureType'}{1};
        design_levels{i} = ReprBld{repr_idx+1, 'DesignLevel'}{1};
        occupancy_classes{i} = ReprBld{repr_idx+1, 'OccupancyClass'}{1};
    end
    % 准备EDP数据矩阵
    edp_matrix = [T.ID, T.MaxDrift, T.MaxAbsAccel, T.ResDrift];

    clear T ReprBld Bld2ReprBld unique_bld_ids
    
    % 调用Python函数进行批量计算
    losses = py.Tool_LossAssess.compute_losses_from_edp(...
        py.numpy.array(edp_matrix), ...              % EDP数据矩阵
        py.numpy.array(bld_props_matrix), ...        % 建筑属性矩阵
        py.list(structural_types(:)'), ...           % 结构类型列表
        py.list(design_levels(:)'), ...              % 设计水平列表
        py.list(occupancy_classes(:)'));             % 使用类别列表

    clear edp_matrix bld_props_matrix structural_types design_levels occupancy_classes

losses_dict = losses.to_dict(pyargs('orient','list'));
losses_struct = struct(losses_dict);

% 获取所有字段名
fields = fieldnames(losses_struct);
num_rows = length(losses_struct.(fields{1}));
mat_data = table();

for i = 1:numel(fields)
    field = fields{i};
    data_py = losses_struct.(field);
    
        try
            data_np = py.numpy.array(data_py);
            mat_data.(field) = double(data_np(:));
        catch
            mat_data.(field) = reshape(double(py.list(data_py)), [], 1);
        end
end

    AllLossSimResults = groupsummary(mat_data, 'SimID', 'sum');
    
    AllLossSimResults.GroupCount = [];
    
    AllLossSimResults.Properties.VariableNames{1} = 'SimID';
    for i = 2:numel(AllLossSimResults.Properties.VariableNames)
        old_name = AllLossSimResults.Properties.VariableNames{i};
        if startsWith(old_name, 'sum_')
            AllLossSimResults.Properties.VariableNames{i} = extractAfter(old_name, 'sum_');
        end
    end
    
    save('AllLossSimResults_BNN_table.mat', 'AllLossSimResults');
end
