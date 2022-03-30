import pandas as pd
import numpy as np


# 对输入的各列feature做正则化或者标准化
##### USER-DEFINED FUNCTIONS
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': #zero mean unit variance
        for j in range(num_Feature):
            if np.nanstd(X[:,j]) != 0:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))/np.nanstd(X[:,j])
            else:
                X[:,j] = (X[:,j] - np.nanmean(X[:, j]))
    elif norm_mode == 'normal': #min-max normalization
        for j in range(num_Feature):
            X[:,j] = (X[:,j] - np.nanmin(X[:,j]))/(np.nanmax(X[:,j]) - np.nanmin(X[:,j]))
    else:
        print("INPUT MODE ERROR!")
    
    return X

# 最后一次测量 to 最大测量年龄（也就是要预测的时间范围都置1）
# [N, num_Event, num_Category] - 最后测量年龄，时间数，最大预测年龄
def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
        mask3 is required to get the contional probability (to calculate the denominator part)
        mask3 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator 样本个数、事件个数、最大预测年龄
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time # 最后一次测量到最大测量年龄（也就是要预测的范围都置1）

    return mask

# 看是否还在出组的标签，即根据PM类别判断？
# time：tte最后一次测量的年龄 
def f_get_fc_mask2(time, label, num_Event, num_Category):
    '''
        mask4 is required to get the log-likelihood loss 
        mask4 size is [N, num_Event, num_Category]
            if not censored : one element = 1 (0 elsewhere)
            if censored     : fill elements with 1 after the censoring time (for all events)
    '''
    mask = np.zeros([np.shape(time)[0], num_Event, num_Category])   # for the first loss function  样本数、事件个数、最大预测年龄
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored                          # 如果没有出组
            mask[i,int(label[i,0]-1),int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i,:,int(time[i,0]+1):] =  1 #fill 1 until from the censoring time (to get 1 - \sum F)
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask5 is required calculate the ranking loss (for pair-wise comparision)
        mask5 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  #lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  #this excludes the last measurement time and includes the event time
    else:                    #single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  #this excludes the last measurement time and includes the event time
    return mask


# 输入的是二维excel表，特征数量
# 目的是把二维表格转成三维矩阵，其中加入了新信息，如时间间隔
##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            - 2: event type 2
            ...
    '''

    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]   #所有样本中的最大测量次数

    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1]) #样本个数*最大测量次数*特征数
    pat_info = np.zeros([len(id_list), 5])  #样本信息？

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]             #number of measurement 测量的次数
        pat_info[i,3] = np.max(tmp['times'])     #last measurement time 最后一次测量时间
        pat_info[i,2] = tmp['label'][0]       #cause PM类别
        pat_info[i,1] = tmp['tte'][0]         #time_to_event 可预测的时间吗，不是。是发作时的年龄，也就是出组的时间
        pat_info[i,0] = tmp['id'][0]      

        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]         # 第i个样本，已有的每次测量，特征数值 （其余空白）
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['times'])  # 第i个样本，已有的每一次测量，距离上一次测量的时间间隔
    
    return pat_info, data


def import_dataset(norm_mode = 'standard'):

    df_                = pd.read_csv('./data/pbc2_cleaned.csv')

    bin_list           = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
    cont_list          = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
    feat_list          = cont_list + bin_list
    df_                = df_[['id', 'tte', 'times', 'label']+feat_list]     # tte: time to event 是做什么的，这么计算出来的？
    df_org_            = df_.copy(deep=True)                                # 为什么要复制一份？

    df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)   

    pat_info, data     = f_construct_dataset(df_, feat_list)    # df_ 导入的数据 feat_list 特征名称列表 返回 pat_info 样本的信息（不含特征） data 三维矩阵
    _, data_org        = f_construct_dataset(df_org_, feat_list)# 复制了一份？

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1    # mask? 置为1
    data_org[np.isnan(data)] = 0    # mask? 置为0
    data[np.isnan(data)]     = 0    # 三维矩阵 空白处也做0

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)  #特征数量：连续+离散+delta测量时间
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    last_meas       = pat_info[:,[3]]  #pat_info[:, 3] contains age at the last measurement # age是最后一次检查的年龄吗
    label           = pat_info[:,[2]]  #two competing risks                                 # label
    time            = pat_info[:,[1]]  #age when event occurred                             # tte - 最后一次测量的年龄

    num_Category    = int(np.max(pat_info[:, 1]) * 1.2) #or specifically define larger than the max tte # num_Category 预测的最大年龄 (建议是乘1.2，但我没有才年龄数据，先预测五年算了吧)
    num_Event       = len(np.unique(label)) - 1 # 事件数，总类别数-1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 #make single risk

    mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)    # 最后一次测量 to 最大测量年龄（也就是要预测的时间范围都置1）
    mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, time, label)
    # DATA            = (data, data_org, time, label)
    MASK            = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi
