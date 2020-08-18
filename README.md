# 在中文自然语言场景，到底需不需要分词？

实验在中文自然语言应用场景下是否有分词的必要性，以及分词后效果到底好多少

## 语料

电商场景下的正负面评论二分类语料

## 环境

tensorflow 2.0+  

python3+

## 说明

以LSTM网络做1个epoch的训练，分别使用one-hot直接丢入与分词后丢入的方式进行效果鉴别，而序列长度分别为200与100。

## 结果

训练一个epoch后，分词后丢入的准确率为0.842，直接one-hot丢入则为0.835。

![image](https://github.com/sun830910/Should_We_Split_Word/blob/master/img/result.jpeg)

## 后续

后续仍可尝试使用one-hot带入embedding以及分词后带入embedding后的效果，但随数据状况差异，分词后的效果明显约优于one-hot的方式一个点左右。