import argparse
import sys
from pathlib import Path
from typing import Tuple
from BldLossAssessment import BldLossAssessment
import pandas as pd
import numpy as np

import MDOF_LU as mlu
import MDOFOpenSees as mops
import BldLossAssessment as bl
import IDA
import Alpha_CNcode as ACN

def compute_losses_from_edp(edp_matrix, bld_props_matrix, structural_types, 
                           design_levels, occupancy_classes):
    """
    批量计算所有建筑的损失
    :param edp_matrix: EDP数据矩阵 [BldID, MaxDrift, MaxAbsAccel, ResDrift] (N_samples × 4)
    :param bld_props_matrix: 建筑属性矩阵 [BldID, NumofStories, FloorArea] (N_buildings × 3)
    :param structural_types: 结构类型列表 (N_buildings)
    :param design_levels: 设计水平列表 (N_buildings)
    :param occupancy_classes: 使用类别列表 (N_buildings)
    :return: 包含所有损失结果的DataFrame
    """
    # 将矩阵转换为DataFrame
    edp_df = pd.DataFrame(edp_matrix, columns=['BldID', 'MaxDrift', 'MaxAbsAccel', 'ResDrift'])
    
    # 创建建筑属性DataFrame
    bld_props_df = pd.DataFrame(bld_props_matrix, columns=['BldID', 'NumofStories', 'FloorArea'])
    bld_props_df['StructuralType'] = structural_types
    bld_props_df['DesignLevel'] = design_levels
    bld_props_df['OccupancyClass'] = occupancy_classes
    
    # 合并数据
    merged_df = pd.merge(edp_df, bld_props_df, on='BldID')
    
    # 分组计算每个建筑的损失
    all_results = []
    
    # 为每种建筑类型创建评估器缓存
    type_cache = {}
    
    # 按建筑ID分组处理
    grouped = merged_df.groupby('BldID')
    total_buildings = len(grouped)
    
    print(f"开始处理 {total_buildings} 个建筑的损失计算...")
    
    for i, (bld_id, group) in enumerate(grouped):
    # 进度显示（保持不变）
        if i % 500 == 0:
            print(f"处理进度: {i}/{total_buildings} 个建筑 ({i/total_buildings*100:.1f}%)")
    
    # 获取建筑属性
        props = group.iloc[0]
    
    # 直接创建新的评估器实例（不再使用缓存）
        loss_assessor = BldLossAssessment(
            int(props['NumofStories']),        # 层数
            float(props['FloorArea']),          # 建筑面积
            props['StructuralType'],            # 结构类型
            props['DesignLevel'],               # 设计等级
            props['OccupancyClass']             # 使用类别
        )
    
    # 提取EDP数据（保持不变）
        max_drifts = group['MaxDrift'].tolist()
        max_accels = (group['MaxAbsAccel'] / 9.8).tolist()
        res_drifts = group['ResDrift'].tolist()
    
    # 计算损失（保持不变）
        loss_assessor.LossAssessment(max_drifts, max_accels, res_drifts)
        
        # 准备结果
        bld_results = pd.DataFrame({
            #'BldID': [bld_id] * len(max_drifts),
            'SimID': range(1, len(max_drifts) + 1),
            #'DS_Struct': loss_assessor.DS_Struct,
            #'DS_NonStruct_DriftSen':loss_assessor.DS_NonStruct_DriftSen,
            #'DS_NonStruct_AccelSen':loss_assessor.DS_NonStruct_AccelSen,
            'RepairCost_Total': loss_assessor.RepairCost_Total,
            'RepairCost_Struct': loss_assessor.RepairCost_Struct,
            'RepairCost_NonStruct_DriftSen': loss_assessor.RepairCost_NonStruct_DriftSen,
            'RepairCost_NonStruct_AccelSen': loss_assessor.RepairCost_NonStruct_AccelSen,
            'RepairTime': loss_assessor.RepairTime,
            'RecoveryTime': loss_assessor.RecoveryTime,
            'FunctionLossTime': loss_assessor.FunctionLossTime
        })
        
        all_results.append(bld_results)
    
    # 合并所有结果
    print("所有建筑处理完成，正在合并结果...")
    result_df = pd.concat(all_results, ignore_index=True)
    
    # 内存优化
    del all_results, merged_df, edp_df, bld_props_df
    print("结果合并完成!")
    
    return result_df

if __name__ == "__main__":
    main(sys.argv[1:])
