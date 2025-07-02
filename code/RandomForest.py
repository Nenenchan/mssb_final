# train_random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from data_processing import load_and_extract_features

# 加载特征
X, y = load_and_extract_features()

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 训练模型
rf_clf = RandomForestClassifier(
    n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'
)
rf_clf.fit(X_train, y_train)

# 预测与阈值调整
y_prob = rf_clf.predict_proba(X_test)[:, 1]
threshold = 0.65
y_pred = (y_prob >= threshold).astype(int)

# 评估
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 特征重要性
importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
print("\nFeature Importances:\n", importances.sort_values(ascending=False))
