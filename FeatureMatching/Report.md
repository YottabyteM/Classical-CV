# Feature Matching

注：因为课程不要求写报告了，就没有详细的介绍了

简单版本的是SIFT/ORB检测特征点和描述符，然后直接KNN比较距离，只要前两个点的比值小于阈值就舍弃，没啥难度。

提高版本前面的步骤也没啥区别，主要是得到的match对后进一步筛选，主要使用的是[GMS方法](https://link.springer.com/article/10.1007/s11263-019-01280-3)，这个可以Work的原因可能在于扩大了感受野，能够提取到一点语义信息(我猜的，不保真)，这个方法虽然步骤简单，但是效果真的巨好(当然也有特征点的功劳，下一次实验应该要写这个)。

![](https://github.com/YottabyteM/Classical-CV/blob/main/FeatureMatching/img/result.png)
