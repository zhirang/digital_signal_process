import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. 加载提取好的特征表格
train_df = pd.read_csv("features_extract/features_train.csv")
val_df = pd.read_csv("features_extract/features_val.csv")
test_df = pd.read_csv("features_extract/features_test.csv")

# 2. 分离特征 (X) 和 标签 (y)
X_train = train_df.drop(columns=['filename', 'label'])
y_train = train_df['label']

X_val = val_df.drop(columns=['filename', 'label'])
y_val = val_df['label']

X_test = test_df.drop(columns=['filename', 'label'])
y_test = test_df['label']

# 3. 数据标准化 (🔥极其重要🔥)
# 因为频谱质心的数值可能是几千，而过零率的数值是 0.0几。
# 如果不标准化，数值大的特征会“压死”数值小的特征。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)    # 注意：验证集和测试集只能 transform
X_test_scaled = scaler.transform(X_test)

# 4. 初始化并训练传统分类器 (随机森林)
print("正在使用 CPU 训练随机森林分类器...")
# n_estimators=500 表示种 500 棵决策树
clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# 5.保存模型
print("\n💾 正在保存模型与标准化器...")
# 将标准化器和模型保存到当前目录下
joblib.dump(scaler, 'save_models/rf_scaler.pkl')
joblib.dump(clf, 'save_models/rf_model.pkl')
print("✅ 保存成功！文件已生成: rf_scaler.pkl, rf_model.pkl\n")

# 6. 在验证集和测试集上进行评估
y_val_pred = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val, y_val_pred)
print(f"🌟 验证集 (Val) 准确率: {val_acc * 100:.2f}%")

y_test_pred = clf.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"🏆 测试集 (Test) 准确率: {test_acc * 100:.2f}%\n")

# 打印详细的分类报告 (看看哪些流派容易混淆)
print("测试集详细分类报告:")
print(classification_report(y_test, y_test_pred))