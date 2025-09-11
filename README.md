**********************************************************************************
/Prepare_Train_And_Pre_Data/：

    1、/BNN_Train_Data_6137/：
        ------------------------------------------------------
        使用6137个建筑的结构参数文件和IDA结果文件————生成BNN数据集
        ------------------------------------------------------
            First_Reshape_BNN_strpm.m
            Second_BNN_data_prep.m
            Third_Transform_mat.m

    2、/BNN_Pre_Data_15829/：
        -------------------------------------------------------------------
        使用15829个建筑的结构参数文件和IM模拟后损失计算文件————生成BNN预测的案例
        -------------------------------------------------------------------
            First_Reshape_BNN_strpm.m
            Second_BNN_data_prep.m
            
**********************************************************************************        
/BNN_Procedure/：

    1、/First_BNN_Train_And_Test/:
        -----------------------------
        使用6137个建筑的IDA结果进行训练
        -----------------------------
            First_standardize_structural_data.m
            BNN_model_MaxDrift.m
            BNN_model_MaxAbsAccel.m
            BNN_model_RestDrift.m
            
    2、/Second_Draw_IDA/:
        ---------------------------------------------------
        从测试集中随机选出建筑样本进行预测————并绘制IDA对比曲线
        ----------------------------------------------------
             First_choice_test_sample_build.m
             Second_test_sample_pre.m
             Third_Draw_IDA_16and84.m
             
    3、/Third_BNN_Pre_15829/:
        -----------------------------
        使用15829个建筑的模拟结果来预测
        -----------------------------
             BNN_predict_all.m
             
    4、/Forth_Loss_by_BNN_pre/:
        ------------------------------------
        使用BNN预测的15829个建筑的EDP来计算损失
        -------------------------------------
              ComputeLossesFromEDP.m
