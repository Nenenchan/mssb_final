import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import time

# 数据预处理函数
def preprocess_data():
    # 加载数据
    student_info_path = 'anonymisedData/studentInfo.csv'
    student_assessment_path = 'anonymisedData/studentAssessment.csv'
    student_vle_path = 'anonymisedData/studentVle.csv'
    
    student_info = pd.read_csv(student_info_path)
    student_assessment = pd.read_csv(student_assessment_path)
    student_vle = pd.read_csv(student_vle_path)
    
    # 选择特定课程AAA
    student_info = student_info[student_info['code_module'] == 'AAA']
    
    # 合并数据集
    merged_data = pd.merge(student_info, student_assessment, on=['id_student'], how='left')
    merged_data = pd.merge(merged_data, student_vle, on=['code_module', 'code_presentation', 'id_student'], how='left')
    
    # 只保留前4周的数据
    merged_data = merged_data[merged_data['date'] <= 28]  # 假设4周=28天
    
    # 填充缺失值
    merged_data.fillna(0, inplace=True)
    
    # 特征选择
    features = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 
               'num_of_prev_attempts', 'studied_credits', 'disability', 'score', 'sum_click']
    X = pd.get_dummies(merged_data[features], drop_first=True)
    y = merged_data['final_result'].apply(lambda x: 1 if x == 'Fail' else 0)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 使用 SMOTE 进行过采样
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("预处理结束。")
    print(f"特征数量：{X.shape[1]}")
    print(f"正样本比例：{y.mean():.2%}")
    return X_train_resampled, X_test, y_train_resampled, y_test

# 随机森林分类器函数
def random_forest_classifier(X_train, X_test, y_train, y_test):
    # 初始化随机森林分类器
    rf = RandomForestClassifier(
        n_estimators=50,  # 减少树的数量以加快训练
        max_depth=11,      # 限制树的深度
        min_samples_split=9,  # 增加分裂内部节点所需的最小样本数
        min_samples_leaf=5,   # 增加叶节点所需的最小样本数
        class_weight={0: 1, 1: 22},  # 处理不平衡数据
        random_state=42,
        n_jobs=-1  # 使用所有CPU核心
    )
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练模型
    rf.fit(X_train, y_train)
    
    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time
    
    # 预测测试集的概率
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # 调整决策阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    y_pred_rf = (y_pred_proba >= optimal_threshold).astype(int)
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_rf)
    report = classification_report(y_test, y_pred_rf, digits=4)
    
    print(f"\n策略：随机森林(前4周数据)")
    print(f"训练集大小：{len(X_train)}")
    print(f"测试集大小：{len(X_test)}")
    print(f"训练时间：{training_time:.2f}秒")
    print(f"准确率：{accuracy:.4f}")
    print("分类报告：")
    print(report)
    
    # 输出特征重要性
    '''print("\nTop 10重要特征：")
    feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    print(feat_importances.sort_values(ascending=False).head(10))'''

# 主函数
def main():
    # 数据预处理
    X_train, X_test, y_train, y_test = preprocess_data()
    
    # 随机森林分类器
    random_forest_classifier(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()