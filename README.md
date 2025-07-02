# 模式识别期末大作业

项目结构如下

```
code/
├── data_processing.py # 数据预处理模块：加载CSV、清洗数据、特征构造等
├── RandomForest.py # 使用随机森林模型进行训练、预测和评估
├── XGBoost.py # 使用XGBoost模型进行训练、预测和评估

oulad/  # 数据集

report/
├── figures/ # 存放报告中使用的图像
├── main.tex # 主报告 LaTeX 文件
├── reference.bib # BibTeX 格式的参考文献库
├── main.pdf # 编译后的报告 PDF 成品
├── ... # 其他自动生成文件，无需在意

```

1. 请将自己的代码文件整合到code中，如果可以尽量使用 `data_processing.py` 提供的预处理数据。

2. 在 `main.tex` 中填充自己部分的实验报告内容，格式可以参考已经完成的部分

