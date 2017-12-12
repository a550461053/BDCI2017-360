# 2017BDCI-360搜索：AlphaGo之后“人机大战”Round 2 ——机器写作与人类写作的巅峰对决

## 赛题描述
- 如果说AlphaGo和人类棋手的对决拉响了“人机大战”的序曲，在人类更为通识的写作领域，即将上演更为精彩的机器写作和人类写作的对决。人类拥有数万年的书写历史，人类写作蕴藏无穷的信息、情感和思想。但随着深度学习、自然语言处理等人工智能技术发展，机器写作在语言组织、语法和逻辑处理方面几乎可以接近人类水平。360搜索智能写作助手也在此背景下应运而生。       
- 本次CCF大数据和人工智能大赛上，360搜索智能写作助手（机器写作）和人类写作将狭路相逢，如何辨别出一篇文章是通过庞大数据算法训练出来的机器写作的，还是浸染漫长书写历史的人类创作的？我们拭目以待！
- 本次赛题任务：挑战者能够设计出优良的算法模型从海量的文章中区分出文章是机器写作还是人类写作。

## 数据集
1. 训练集：
	- id, title, content, label
	- label为'POSITIVE'/'NEGATIVE'
2. 测试集：
	- id, title, content
3. 说明：
	- 初赛：由于第一次数据集较为简单，很快就有达到0.98的，所以官方又发布了一次数据，第二次new数据是在第一次old数据的基础上进行了叠加，而且分布不同。
	- 复赛：只用了semi的数据，没有用初赛数据（如果用了，可提高得分点）

## 比赛结果
1. 初赛19，score=0.8123
2. 复赛21，score=0.66657

## 思路
0. 360搜索题解
    - 360搜索核心：
        - 系统：大规模分布式系统，支撑大规模的数据处理容量和在线查询负载
        - 数据：数据处理和挖掘能力
        - 算法：搜索相关性排序，查询分析，分类，等等
    - 赛题目的：
        - 主要是为了区分网上的文本
        - 模型的鲁棒性：能否能抗压
        - 速度：
        - 性能：能否多用CPU，少用昂贵的GPU

1. 初步筛选
	- 使用网上公开的大规模语料训练词向量word2vec-api
	- 使用CNN进行文本分类
	- 步骤：
		+ 分词
			* jieba分词
			* 只保留：'汉字 英文,.，。'
			* 此题最好不去：分词之后再去停用词 和 空格，但保留原始符号分割效果
		+ word2vec
			* 其实最好效果，应该是把train和test都作为词库输入给word2vec
			* 将分词的结果直接输入给word2vec，并保存产生的model
		+ padding
			* 取每句话词向量长度为500，也就是分词后取前500个词
			* 不足500的词向量值填补0

2. 官方提示：
	- 重视数据预处理：
	    - 对于常见停用词，噪声词的去噪，低频词，高频词过滤
	- 重视数据的特征工程：
	    - 常见的如n-gram，tf，idf，词性标注，句法依存，
	    - 文档中主题分散度：各种主题的糅杂在一起的是机器
	    - 语义相关性：各种主题语义的词柔和在一起是机器
	    - 文本内实体关联度等。
	- 重视对模型的选择：
	    - 考虑使用深度模型，CNN、RNN、LSTM等
	    - 强烈建议使用这个，不然使用传统的svm或者xgboost等特征工程的方法，很浪费调参、等时间。

3. 其他人提示：
    - 用训练集的正样本，LDA分主题，再随机拼贴插入同主题其他文章的句子，机器效果更好
    - 只需要train的正样本做word2vec即可，加入train的负样本反而效果会不好
    - nlp的gan效果没那么好：
        - gan
        - seq2seq：直接生成部分
    - 初赛0.97：
 	+ cnn ＋ BN（batchnormaliztion） ＋ dropout ＋ 预训练词向量
 	+ gensim 的 word2vec，把训练集和测试集的预料都用上
 	+ titanX下：网络参数不用太多，也不要很多层。先调到能过拟合，再慢慢加正则。一个小时就差不多了
    - 启发：
    	+ 通过拼接句子构造负样本，然后识别区分人机的nlp任务
    - 结巴：
    	+ 结巴分完词后解决100w，选词频前20w，其余的直接扔掉，不然按unknown处理会加大内存消耗
    	+ 我觉得这样不合理，最起码人眼看的话，如果去掉unknown反而更加混淆了。。
    - 正则：
    	+ 直接利用正则表达式找负样本的特征
    	+ 比如ab+这个表达式描述了特征是一个a和任意个b的特征字串
    	+ 这个方法就是纯规则，还是有用的
    
    - 为了找到最优的过滤器(Filter)大小，可以使用线性搜索的方法。通常过滤器的大小范围在1-10之间，当然对于长句，使用更大的过滤器也是有必要的

    - 模型：
    	- Feature Map的数量在100-600之间；
    	- 可以尽量多尝试激活函数，实验发现ReLU和tanh两种激活函数表现较佳
    	- 使用简单的1-max pooling就已经足够了，可以没必要设置太复杂的pooling方式
    	- 当发现增加Feature Map的数量使得模型的性能下降时，可以考虑增大正则的力度，如调高dropout的概率
    	- 为了检验模型的性能水平，多次反复的交叉验证是必要的，这可以确保模型的高性能并不是偶然——此题没时间用交叉。

3. 优化：
    - 使用已有的文本自行生成词向量
    - 给jieba词典添加一些常用的词典
    - 加入文本特征的规则阈值ifelse
    - 加入svm、xgboost等混合模型
    - 具体特征：
        + 题目：
        + 内容：
            - 标点问题
            - 语义连贯性
            - 不要去停用词，这样就破坏特征了，有的机器写的还会有语法错误
            - 长度：机器人一不小心就会写很长。。。：字数和句子数
            - 关于长短文本的处理：
            	+ 直接补齐最长的不可能，内存不允许，也容易过拟合
            	+ 将长句子切割为多个子句单元，最后输出的时候整合，或者在网络中间整合为一个输出。
            - 特殊长度：
            	+ 如果文本太长，检查是否包含大量字母，包含，则为Negative
        + 生成来源：
            - 很多句子应该是从其他文章中放进去的，应该不是从词单位来生成文章。
            - 所以直接针对某个局部的句子是看不出来的，那么就可以通过jieba分词，选取词频前多少的词进行词向量的建立
4. myidea:
	- 机器和人最大的区别是：
		+ 主要分析初赛的两次新旧数据：
		+ 1.old：多主题混合，一般是2个主题；
			* 重复句子
			* 含有特殊句子：用微信扫描二维码分享至好友和朋友圈
				- 但是经过分析，重复三次以上的是NEG
				- 重复两次的都是在content的开头出现，POS和NEG都差不多，所以需要把此类re.sub()重新考虑后面的内容来判断分类。 
		+ 2.new：句子换乱，某些句子的语法或者句意表示不清

	- 解决：
		+ 1.多主题检测
		+ 2.句意合理性检测

	- 和队友之间的区别：
		+ 词向量维度200
		+ 词向量训练mincount=3
		+ 大小写 统一为 小写
		+ TextCNN的input为[,,,1]最后的1
		+ 重复的句子，重复出现过的句子为neg

	- trick:
            	+ 使用pretrain的词向量进行后面的模型训练，提高效果。
		+ 预处理要做好：先去掉每段话的开头重复句子：“用微信扫描二维码分享至好友和朋友圈”
			* 相当于是对文本的某些特定部分进行了提前标识，标注了更多信息
		+ 每段话中含有重复的句子，一定为neg
		+ 每段话中含有句子特别长，大于50，则几乎为neg
			* 按，、。；：!分句子；
		+ feature_map的数量要比word vector size值大一些
		+ “推荐阅读”
		+ “、：” 得保留
		+ “1.1”，“...”处理"." ，直接去掉.即可
		+ "1-2"，处理"-"
	- 最终过程：
		+ 先用cnn跑public_eval_seg.tsv，得到所有数据的预测结果
		+ 然后利用0.92的规则跑public_eval_seg.tsv，得到所有数据的预测结果
		+ 然后取前150000-77为规则的结果，后面剩下的为cnn的结果 - 一共40w
			        # evaluation_public.tsv 400000 = 
			        	evaluation_public_old.tsv 150000，还存在77个不在现在数据集上的废弃数据
			        	evaluation_public_change.tsv 250077, 
			        # train.tsv             500000 = pos:180263, neg:319737
			        # train_old.tsv         200000 = pos:120163, neg:79837
			        # train_change          300000 = pos:60100,  neg:239900
		+ 训练速度：
			* 20w train，wordlen 400
				- 每个epoch为1561个steps
				- 开始：11:50 - 20个step就达到0.6loss，0.6acc
				- 中间：17.18 - 3000step的时候0.66loss，acc0.61
				- 结束：24.53 - 7131step
				- 相当于12个小时跑完4个半epoch
			* 计划为：3个epoch，wordlen 降为300, 20w train，大概10h跑完，lr=0.001每隔1个epoch的steps就*0.1，最终0.00001，看是否能降为0.8的acc
			* 新计划为：
				- win跑前20w train，测试是否收敛到0.8以上，3个epoch；好测试eval结果
				- linux跑50w train，测试是否能够学习到两种不同分布，而不是2-8分的样本集导致的初始输出就是0.8acc，所以需要进一步查看，epoch5

## 模型分析：
- 传统编码器-解码器结构的lstm、rnn模型存在一个问题：不论输入长短都将其编码成一个固定长度的向量表示，这使得模型对于长输入序列的学习效果很差，也就是解码效果很差；
- attention机制克服了该问题，原理是：
	* 在模型输出时会选择性地专注考虑输入中的对应相关的信息，使用attention机制的方法被广泛应用在各种序列预测任务上；
- fasttext使用的是线性模型，把句子中的所有词向量进行平均，然后直接接softmax层；
	* 没有考虑语序
	* 但是加了一些n-gram的trick来增加局部的词序信息
- textCNN：
	* CNN的核心在于捕捉局部相关性，利用到nlp中，也就是利用CNN来提取句子中类似n-gram的信息
	* 需要做non-static训练词向量，也就是用pretrain好的word2vec初始化，进一步进行fine-turing，效果更好
	* cnn局部视野，无法建模更长的序列信息
	* filter_size的超参调节也很麻烦
- textrnn：
	+ cnn在文本上本质只是一种特征表达，nlp还是常用rnn
	+ rnn可以更好的表达上下文信息
	+ Bi-directional RNN实际上使用的是双向LSTM，本质上可以看作是捕获变长且双向的n-gram信息
	+ RNN可以处理变长序列：
		* 最初实验：将序列中的元素逐个输入网络，并预测下一时刻的输入，比如输入一个句子到网络，在输入第n个字符时预测第n+1个字符，这种方式虽然能够处理变长的输入序列，但得到的输出序列和输入序列一样长，限制仍然存在
		* 改进实验：在输出时增加一个空白的输出候选，然后在每次输出时取每一种可能输出结果的概率，得到一张路径网络，用beam search方法组装真正的输出，这样最后得到的非空白输出序列的长度就变成可变的了。
		* 句子的每个单词都被加上了时间步数，也就是时间步长就等于最大序列长度
- cnn+rnn：
	+ 先利用cnn将句子各个局部进行计算，得到n个向量（也就是n个filters）
	+ 把各个向量同一时间的元素取出来组成新向量，每个向量内部各个元素都对应同一时间点的单词，而向量之间对应句子中单词的顺序关系
	+ 最后把新组建的向量当做输入向量序列，放入LSTM中进行训练。
- 最本质问题：
	+ 两个分布的数据由于数据量的限制，导致2-8分现象，需要进一步解决该数据不均匀问题：
	+ 参见：http://www.cnblogs.com/zhaokui/p/5101301.html

## 模型初级调参经验
1. 使用relu+bn是万金油
2. 尽量做shuffle和aug
3. 学习率：
	- 学习率降低的一瞬间，网络会有一个巨大的性能提升
	- fine-tuning需要根据模型的性能设置合适的学习率，比如一个已经训练好的模型，如果上来给1e-3的学习率，那之前就白训练了，也就是说网络性能越好，学习率要越好
4. batchsize通常影响没那么大，塞满卡就行
	- batchsize的增加可以调高训练速度，但是很有可能使收敛结果变差
	- 一般初始设置大，因为不会对结果有太大影响，而太小反而会使得结果很差
5. adam收敛速度快，结果却不如其他优化算法
6. dropout放置位置以及大小
7. 词向量embedding：128、256
8. 正负样本比例：
	- 默认数据的比例不平衡时，很容易影响训练结果
	- 对数目少的样本进行过采样，比如复制，提高其比例
9. 每个mini-batch中：
	- 正负样本的比例，尽量让一个batch内，各类别的比例平衡，图像上尤为重要
10. 自动调参：
	- grid search：网格搜索，就是遍历，太浪费时间
	- random search：更有效，先用grid得到候选参数，然后从中随机选择进行训练
	- bayesian optimization：贝叶斯优化，考虑到不同参数对应的实验结果值
		+ 有现成的库，还有的支持并行调参

## 模型 高级调参经验：
1. 卷积核的分解：从5*5分解为两个3*3，再到后来的3*3分解为1*3和3*1，。。。网络的计算量越来越小，层数越来越多，性能越来越好
2. 不同尺寸的feature maps的concat，只用一层的feature map可能不如concat好
3. 正则化范数：
	- w = argmin(Loss) + ||w||0 + ||w||1 + ||w||2
	- l2暂时不需要，因为没有过拟合
4. 试试迁移学习，把训练集连同测试集的预测结果一起进行训练：（即可能包含有误的数据）但有提升
5. 别人：
	- 随机打乱训练数据
	- 增加隐层，和验证集
	- 正则化
	- 对原数据进行PCA预处理
	- 调节训练参数（迭代次数，batch大小等）
6. lstm：
	- 层数使用2-4层
	- 单向lstm和双向lstm有时候区别不大
	- 使用w2v表现会有提升
	- 使用l2正则后，会有提升，l2=0.001
	- batch_size从64改为16后，acc提升了10%
8. rnn：
	- 论文：先用ada系列跑，最后快收敛的时候，换sgd继续训练，会有所提升
	- 除了gate之类的地方,需要把输出限制成0-1之外,尽量不要用sigmoid,可以用tanh或者relu之类的激活函数.
	- LSTM 的forget gate的bias,用1.0或者更大的值做初始化,可以取得更好的结果,来自这篇论文:http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf, 我这里实验设成1.0,可以提高收敛速度.实际使用中,不同的任务,可能需要尝试不同的值.
	- ！！！使用orthogonal进行初始化
		+ 正交初始化（有待测试）

9. Ensemble: 论文刷结果的终极核武器,深度学习中一般有以下几种方式
	- 同样的参数，不同的初始化方式
	- 不同的参数，通过cross-validation 选取最好的几组
	- 同样的参数，模型训练的不同阶段
	- 不同的模型，进行线性融合，例如RNN和传统模型

## 模型 改进方案：
1. 词向量训练：
	- 输入数据有两个channel,其中一个是静态的embedding,另一个是动态的embedding.静态embedding是事先训练好的,在此次训练中不再变化;动态embedding会在训练过程中也进行参数求解.
	- 仅使用静态embedding的话可能因为embedding训练数据集与实际数据集有偏差而导致不准确;仅用动态数据集的话其初始化会对训练有影响.
	- 本项目：考虑训练数据足够充足，采用静态embedding的方法，使用gensim提前训练好词向量
2. 数据不均衡问题：参见Learning from Imbalanced Data
	- 大数据+均衡
	- 大数据+不均衡
		+ 经验表明：训练数据中每个类别有5000个以上样本，数据量是足够的，正负样本差一个数量级以内是可以接受的，不太需要考虑数据不均衡问题。
		+ 上采样：
			* 把数据小的类别复制多份
		+ 下采样：
			* 把数据大的类别剔除一些
	- 小数据+均衡
	- 小数据+不均衡
	- 综上：我们的数据还算均衡。。同一量级内
3. NLP中DNN模型选型问题：
	- 分层架构：
		* CNN善于抽取位置不变特征
		* 应用：常用于分类，如情感分类，因为情感通常由一些关键词决定
			- 但是，RNN在文本级别的情感分类表现很好
	- 连续架构：
		* RNN善于按序列对单元建模
		* 应用：常用于顺序建模，语言建模任务，了解上下文的基础上建模
			- 但是，封闭的CNN在语言建模任务上比LSTM更优
		* 加入闸门机制：
			+ LSTM
			+ GRU
	- 总结：
		+ cnn和rnn在文本分类问题上提供补充信息，至于选择哪种架构，取决于对整个序列语义理解的重要程度。
		+ rnn在大范围任务中都较为稳健，除了：需要识别关键词的任务。
		+ 相比而言，学习率的变化相对平稳，而隐藏层尺寸hidden size和批尺寸batch size会引起很大的波动。
	- 本题方案：
		+ old data主要是主题融合，主题错误，用cnn更好
		+ new data主要是语法错误，用rnn更好
		+ 如何融合两者。
4. semi-final：
	- 复赛初步机器分工：
		+ 笔记本验证之前的规则是否适应新数据；
		+ 台式机验证fasttext效果如何
		+ 服务器验证textcnn效果如何
	- 词向量训练：
		+ 生成词库90w和300w差距不明显，使用fasttext进行测试，300w的acc=0.623，,9w的acc=0.61
	- 分工：
		+ 服务器验证前10w数据的textcnn性能，len取前2000
			* 测试前1w数据，5个epoch，lr=0.0001
				- 不算标点：1.0 loss，acc=0.6
				- 算标点的性能：1.5loss, acc=0.5,也有一次0.83
		+ 台式机验证前10w数据的rnn性能，
	- pretrain word2vec model：
		+ 使用pretrain词向量的方式，加载gensim预训练好的词向量，
			* 1. w = tf.constant(embedding, name='W')
			* 2. w = tf.Variable()
		+ 使用tf.train.Saver()加载tensorflow本身embedding层训练好的词向量
	- 模型阈值修改：
		+ nn模型默认都是0.5作为分界线阈值，也就是大于0.5为pos，可以调整这个阈值；
	- 大规模网络 结构优化：
		+ 正常时候同时想要valid model，如果把valid和train放在同一张图里进行，通常会导致训练的中断和训练时间延长
		+ 分离train和valid：让valid单在单独的进程里面
			* tf.saver()保存模型，然后在另一个进程读取最新的模型进行valid
			* GPU进行training，CPU进行valid，CPU进行predict
		+ 分布式：
			* server进行调度
		+ 使用pretrained model：
			* 例如：基于inception模型实现，先将inception模型的输出层去掉，然后将模型的输出作为一个RNN模型的输入来构造计算图，在训练的时候会读取inception，并进行二次训练。

## 经验总结：
抽空汇总一下实际项目需要用的：
1. python性能优化
	* a = b = [] 不能这样写，ab指向同一对象
	* c1=c2=0 可以这样写，常数会生成对象
2. tensorflow使用：
	- 待完善
3. tensorflow常见错误总结：
	- setting an array element with a sequence：因为这儿需要的是 array，你用的是 list，或者需要的是 list，你用的 array，从这方面入手进行改错
	- 优化器 optimizer，GradientDescentOptimizer 不报错，RMSPropOptimizer，AdamOptimizer 会报错：
		+ 因为 AdamOptimizer， RMSPropOptimizer 他们在内部会生成新的变量，所以 tf.initialize_all_variables() 应该在 optimizer 定义的后面再运行，不能在前面运行。
4. nn如何进行融合：