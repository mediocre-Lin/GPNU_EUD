本文根据西瓜书以及查阅资料所写。
# 机器学习简介

<div align='right'>@马占奎</div>
简单的说，就是使用各种数学知识，将人的一些行为用数学解释出来，再通过计算机去复现。

其实机器学习就是一类算法的总称，这些算法企图从大量历史数据中挖掘出其中隐含的规律，并用于预测或者分类，更具体的说，机器学习可以看作是寻找一个函数，输入是样本数据，输出是期望的结果。

机器学习的目标是使学到的函数很好地适用于“新样本”，而不仅仅是在训练样本上表现很好。学到的函数适用于新样本的能力，称为泛化（Generalization）能力。

机器学习所研究的内容，是关于在计算机上从数据中产生“模型”(model)的算法，即学习算法(learning algorithm)。有了学习算法，我们把经验数据提供给它，它就能基于这些数据产生模型；在面对新情况时，模型就能够通过经验提供相应的判断。

## 发展历程
机器学习是人工智能(artificial intelligence)研究发展到一定阶段的必然产物。
人工智能从二十世纪五十年代初至今已经历过三个时期，分别如下。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916140403108.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzMzMjQyMg==,size_16,color_FFFFFF,t_70#pic_center)


## 基本术语
1. 数据集(data set)、样本(data)——特征向量(feature vector)、特征(feature)、样本空间(sample space)、学习器(learner)
	> 样本又可以称为特征向量，特征的取值称为特征值。一般又将模型称为学习器。

2. 训练集(training data)、测试集(testing data)、预测(predict)、标记(label):
	> 训练过程中使用的数据称为”训练数据”，其中每个样本称为训练样本，训练样本组成的集合称为“训练集”。通过训练样本的标记信息，得到关于预测的模型。
	
3. 分类(classification)、回归(regression)、正类(positive class)、反类(negative class)：
	>若预测的是离散值，如“好”“坏”，此类学习称为分类。若预测的是连续值，例如西瓜的成熟度为0.95、0.37等，此类学习任务称为回归。
	
4. 聚类(clustering)、簇(cluster)：
	>将训练集中的样本分成若干组，每组称为一个簇。
	
5. 监督学习(supervised learning)、无监督学习(unsupervised learning)
	> 分类和回归是前者的代表，聚类是后者的代表
	
6. 泛化(generalization)能力
	> 泛化能力指学得模型适用于新样本的能力。

## 模型评估与选择
### 经验误差与过拟合
通常我们把分类错误的样本数占样本总数的比例称为错误率(error rate)，即如果在m个样本中有a个样本分类错误，则**错误率**$E=\frac{a}{m}$，相应的，$1-\frac{a}{m}$称为**精度**(accuracy)。

更一般的，我们把学习器的实际预测输出与样本的真实输出之间的差异称为“**误差**”(error)，学习器在训练集上的误差称为“**训练误差**”(training error)，在新样本上的误差称为“**泛化误差**”(generalization error)。
	
我们肯定都想要得到泛化误差小的学习器，然而我们并不能提前得知新样本是个什么样，所以我们能做的只是努力使经验误差（训练误差）最小化。
	
我们实际希望的是在新样本上能表现的很好的学习器。为了达到这个目的，应该从训练样本中，尽可能的学出适用于所有潜在样本的“普遍规律”。
	
然而，当学习器把样本学的太好的时候，可能会将一些特殊的或潜在的性质当成所有样本都可能具有的一般性质，这样就会导致学习器的泛化能力下降，这种现象称为过拟合。与之相对的是欠拟合，即没有学到足够的一般性质。
	
有多种因素可能导致过拟合和欠拟合，欠拟合比较好处理，例如在决策树中增加扩展分支，在神经网络中增加训练次数等。而过拟合很麻烦，过拟合是机器学习面临的关键障碍，各类学习算法都有专门针对过拟合的措施，但是必须意识到，过拟合是无法避免的，我们只能尽可能的“缓解”，或者说减小其风险。




### 评估方法

通常，我们通过实验测试来对学习器的泛化误差进行评估而做出选择。为此，需要使用一个“**测试集**”(testing set)来测试学习器对新样本的判别能力，然后测试集的测试误差作为泛化误差的近似。

通常我们假设测试样本也是从样本真实分布中独立同分布采样而得，但需注意的是，测试集应该尽可能的和训练集互斥。

1. **留出法**（hold out）
	直接将数据集D划分成两个互斥的集合。其中一个集合作为训练集S，另一个作为测试集T。在S上训练出模型之后，用T来评估其测试误差，作为对泛化误差的估计
2. **交叉验证法**（cross validation）
	现将数据集D划分为k个大小相似的互斥子集，即$D = D_1\cup{D_2}\cup{D_3}\cup{D_4}\cup{D_5}\cup\cdots\cup{D_k},{D_i}\cap{D_j}=\varnothing(i\neq{q})$
	每个子集都尽可能的保持数据分布的一致性，即从D中分层采样得到。然后，每次用k-1个子集的并集做训练集，余下的那个子集作为测试集；这样就可以获得k组训练集和测试集，从而可进行k次训练和测试，最终返回这k个测试结果的均值。
	显然，交叉验证法评估结果的稳定性和保真性在很大程度上取决于k的取值，为了强调这一点，通常把交叉验证法称为“k折交叉验证”（k-fold cross validation）。k最常用的值为10，此时称为10折交叉验证。

3. **自助法**（bootstrapping）
	自助法以自助采样法（bootstrap sampling）为基础。给定包含m个样本的数据集D，我们对它进行采样产生数据集**D'** ：每次随机从D中挑选一个样本，将其拷贝放入**D'**，然后再将该样本放回初始数据集中，使得该样本下次还有可能被采到。这个重复执行m次后，我们就得到了包含m个样本的数据集**D'**，这就是自助采样的结果。
	
	自助法在数据集较小、难以有效划分训练集和测试集时很有用；此外，自助法能从初始数据集中产生多个不同的训练集，这对集成学习等方法有很大的好处。
	
	然而，自助法产生的数据集改变了初始数据集的分布，这会引进估计偏差，因此，在数据量足够时，留出法和交叉验证法更常用一些。

### 调参与最终模型
机器学习常涉及两类参数：
一类是算法的参数，也称“**超参数**”，数目通常在10以内；
另一类是**模型的参数**，数目可能很多。

两者的调参方式相似，均是产生多个模型之后基于某种评估方法来进行选择。不同之处，在于前者通常是由人工设定多个参数候选值后产生模型，后者则是通过学习来产生多个候选模型。

### 性能度量
对学习器的泛化性能进行评估，不仅需要有效可行的实验估计方法，还需要有衡量模型泛化能力的评价标准，这就是性能度量（performance measure）。

回归任务最常用的性能度量是“**均方误差**”（mean square error）（MSE）
$E(f;D)=\frac{1}{m}\sum\limits_{i=1}^m(f(x_i)-y_i)^2$

主要介绍分类任务中常用的性能度量
1. 错误率与精度
	错误率是指分类错误的样本数占样本总数的比例，精度则是分类正确的样本数占样本总数的比例。

2. 查准率、查全率与F1
	查准率也可叫准确率，查全率也可叫召回率。![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916162545283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzMzMjQyMg==,size_16,color_FFFFFF,t_70#pic_center)

	查准率(precision)$$P=\frac{TP}{TP+FP}$$
	查全率(recall)$$R=\frac{TP}{TP+FN}$$
	以查全率为横轴、查准率为纵轴作图，就得到了查准率-查全率曲线，简称‘**P-R曲线**’。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916220650121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzMzMjQyMg==,size_16,color_FFFFFF,t_70#pic_center)
如果一个学习器的P-R曲线被另一个学习器的P-R曲线完全包住，则可断言后者的性能优于前者。

	例如上面的A和B优于学习器C，但是A和B的性能无法直接判断，但我们往往仍希望把学习器A和学习器B进行一个比较，我们可以根据曲线下方的面积大小来进行比较，较常用的是平衡点。**平衡点（BEP）是查准率=查全率时的取值，如果这个值较大，则说明学习器的性能较好**。

	但是BEP还是过于简化了，所以我们更常用的是F1度量。
	
	F1度量$$F1=\frac{2\times{P}\times{R}}{P+R}=\frac{2\times{TP}}{样例总数+TP-TN}$$
	
	同样，F1值越大，我们可以认为该学习器的性能较好。
### ROC曲线和AUC
很多学习器是为测试样本产生一个实值或概率预测，然后将这个预测值与一个**分类阈值**(threshold)比较，若大于阈值则为正类，若小于阈值则为负类。例如，神经网络在一般情形下是对每个测试样本预测出一个[0.0,1.0]之间的实值，设置阈值为0.5，将预测值与阈值进行比较，大于阈值则判为正例，小于阈值则判为反例。这个实值或概率预测结果的好坏，直接决定了学习器的泛化能力。

 **ROC全称是“受试者工作特征”(Receiver OperatingCharacteristic)曲线。**
 
 我们根据学习器的预测结果，把**阈值**从0变到最大，即刚开始是把每个样本作为正例进行预测，随着阈值的增大，学习器预测正样例数越来越少，直到最后没有一个样本是正样例。

在这一过程中，每次计算出两个重要量的值，分别以它们为横、纵坐标作图，就得到了“ROC曲线”。

ROC曲线的纵轴是“真正例率”(True Positive Rate, 简称TPR)，横轴是“假正例率”(False Positive Rate,简称FPR)。
$$TPR=\frac{TP}{TP+FN}$$
$$FPR=\frac{FP}{TN+FP}$$
我们可以发现：TPR=Recall。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916215751175.png?size_16,color_FFFFFF,t_70#pic_center)


现实任务中通常是利用有限个测试样例来绘制ROC图，此时仅能获得有限个(真正例率，假正例率)坐标对，无法产生图中的光滑ROC曲线，只能绘制出下图所示的近似ROC曲线。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200916224222934.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzMzMjQyMg==,size_16,color_FFFFFF,t_70#pic_center)
对于学习器的比较时，与P-R图类似，若一个学习器的ROC曲线被另一个学习器的曲线完全“包住”，则可断言后者的性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性地断言两者孰优孰劣。
此时如果要进行比较，较为合理的判据是比较ROC曲线下的面积，即AUC(Area Under ROC Curve)。

