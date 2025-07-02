import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def preprocess_data():
    # 加载数据
    student_info = pd.read_csv('anonymisedData/studentInfo.csv')
    student_assessment = pd.read_csv('anonymisedData/studentAssessment.csv')
    student_vle = pd.read_csv('anonymisedData/studentVle.csv')
    
    # 仅选择课程AAA
    student_info = student_info[student_info['code_module'] == 'AAA']
    student_info = student_info.sample(frac=0.1, random_state=42)
    
    # 合并数据集
    merged_data = pd.merge(student_info, student_assessment, on=['id_student'], how='left')
    merged_data = pd.merge(merged_data, student_vle, on=['code_module', 'code_presentation', 'id_student'], how='left')
    
    # 只使用前4周数据
    merged_data = merged_data[merged_data['date'] <= 28]
    
    # 填充缺失值
    merged_data.fillna(0, inplace=True)
    
    # 特征选择（减少分类变量）
    features = [
        'num_of_prev_attempts',  # 历史尝试次数
        'studied_credits',       # 学习学分
        'score',                 # 评估分数
        'sum_click'              # 点击量
    ]
    X = merged_data[features]
    y = merged_data['final_result'].apply(lambda x: 1 if x == 'Fail' else 0)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("预处理完成。")
    print(f"总样本数：{len(X)}")
    print(f"正样本比例：{y.mean():.2%}")
    return X_train, X_test, y_train, y_test

def dbscan_clustering(X_train, X_test, y_train, y_test):
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dbscan = DBSCAN(
        eps=0.5,          
        min_samples=6,    
        metric='euclidean',
        n_jobs=1          # 禁用并行计算以防内存错误
    )
    
    # 训练模型
    start_time = time.time()
    dbscan.fit(X_train_scaled)
    training_time = time.time() - start_time
    
    # 预测测试集
    y_pred = dbscan.fit_predict(X_test_scaled)
    
    # 简化标签映射：将最大簇设为0，其他为1
    if len(np.unique(y_pred)) > 1:
        main_cluster = np.argmax(np.bincount(y_pred[y_pred != -1] + 1)) - 1
        y_pred_mapped = np.where((y_pred == main_cluster) | (y_pred == -1), 0, 1)
    else:
        y_pred_mapped = np.zeros_like(y_pred)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred_mapped)
    report = classification_report(y_test, y_pred_mapped, digits=4)
    conf_matrix = confusion_matrix(y_test, y_pred_mapped)
    
    print(f"\nDBSCAN聚类结果 (eps=0.7, min_samples=5):")
    print(f"训练时间：{training_time:.2f}秒")
    print(f"噪声点比例：{np.mean(y_pred == -1):.2%}")
    print(f"准确率：{accuracy:.4f}")
    print("\n分类报告：")
    print(report)
    print("\n混淆矩阵：")
    print(conf_matrix)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    dbscan_clustering(X_train, X_test, y_train, y_test)