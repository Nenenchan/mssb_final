# train_xgboost.py

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from data_processing import load_and_extract_features

# -----------------------
# 加载数据
# -----------------------
X, y = load_and_extract_features()

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------
# 模型训练
# -----------------------
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # 应对类别不平衡
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train, y_train)

# -----------------------
# 预测与评估
# -----------------------
y_prob = xgb_clf.predict_proba(X_test)[:, 1]
threshold = 0.65
y_pred = (y_prob >= threshold).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------
# 特征重要性
# -----------------------
importances = pd.Series(xgb_clf.feature_importances_, index=X.columns)
print("\nFeature Importances:\n", importances.sort_values(ascending=False))
