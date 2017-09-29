# virtual\_stock_forecasting
## **虚拟股票预测**

比赛网址：[https://challenger.ai/competition/trendsense](https://challenger.ai/competition/trendsense)

主要任务为通过挖掘虚拟股票大规模历史数据的内在规律，实现对虚拟股票未来趋势的预测。

## 解决方案:

1. 利用clementine对数据集进行分析，观察feature，label，gruop，weight，era的分布.
2. 对训练数据，按照era列进行分组，然后按照train：val＝7:3的比例，设置随机seed，将训练数据划分为训练、验证两份数据.
3. 主要利用sklearn的RandomForestClassifier等树模型进行训练预测，测试时添加投票表决.
4. 公榜成绩最高0.669（当前公榜最高0.63，私榜最高0.69）.


## 数据说明:

**训练数据**，是一个以逗号分隔的文本文件(csv)，格式示例：

|id|feature_0|...|feature_n|weight|label|group|era|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0|0.254232|...|0.473321|9.0|1.0|1.0|1.0|
|1|0.763212|...|0.309311|3.0|0.0|7.0|1.0|

其中id列为数据唯一标识编码，feature列为原始数据经过变换之后得到的特征，weight列为样本重要性，label列为待预测二分类标签，group列为样本所属分组编号，era列为时间区间编号(取值1-20为时间顺序)。

**测试数据**，是一个以逗号分隔的文本文件(csv)，格式示例：

|id|feature_0|...|feature_n|group|
|:---:|:---:|:---:|:---:|:---:|
|600001|0.427248|...|0.754322|3.0|
|600002|0.253232|...|0.543121|5.0|

