# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 21:04:43 2026

@author: FST
"""
import pandas as pd
import numpy as np

def weighted_r2_csiro(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    """
    計算圖片中所定義的全域加權 R2 分數
    y_true, y_pred: 應為 DataFrame 或 ndarray，欄位順序需一致
    """
    # 1. 定義圖片中的權重
    weights_dict = {
        'Dry_Green_g': 0.1,
        'Dry_Dead_g': 0.1,
        'Dry_Clover_g': 0.1,
        'GDM_g': 0.2,
        'Dry_Total_g': 0.5
    }
    
    # 確保輸入是 numpy array 方便計算
    if isinstance(y_true, pd.DataFrame):
        # 依照權重字典的順序提取欄位，確保權重對應正確
        cols = list(weights_dict.keys())
        y_true = y_true[cols].values
        y_pred = y_pred[cols].values
    
    weights = np.array(list(weights_dict.values()))
    
    # 2. 攤平資料 (Flatten)
    # 將所有 (image, target) 對轉為一維長向量
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 3. 建立對應的權重向量 w_j
    # 每個樣本有 5 個 target，所以權重也要重複 N 次
    n_samples = y_true.shape[0]
    w_j = np.tile(weights, n_samples)
    
    # 4. 計算加權平均值 y_bar_w
    y_bar_w = np.sum(w_j * y_true_flat) / np.sum(w_j)
    
    # 5. 計算 Residual Sum of Squares (SSres)
    ss_res = np.sum(w_j * (y_true_flat - y_pred_flat)**2)
    
    # 6. 計算 Total Sum of Squares (SStot)
    ss_tot = np.sum(w_j * (y_true_flat - y_bar_w)**2)
    
    # 7. 計算 R2_w
    r2_w = 1 - (ss_res / ss_tot)
    
    return r2_w