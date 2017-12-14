
import tensorflow as tf
import numpy as np


class RNN_Model(object):



    def __init__(self,config,is_training=True, using_api=True, usemydata=False, pretrain_word2vec=True):
        if not usemydata:
            self.keep_prob = config.keep_prob
            self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)

            num_step = config.num_step
            self.input_data = tf.placeholder(tf.int32, [None, num_step])
            self.target = tf.placeholder(tf.int64, [None])
            self.mask_x = tf.placeholder(tf.float32, [num_step, None])

            class_num = config.class_num
            hidden_neural_size = config.hidden_neural_size
            vocabulary_size = config.vocabulary_size
            embed_dim = config.embed_dim
            hidden_layer_num = config.hidden_layer_num
            self.new_batch_size = tf.placeholder(tf.int32, shape=[], name="new_batch_size")
            self._batch_size_update = tf.assign(self.batch_size, self.new_batch_size)

            # build LSTM network
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
                if self.keep_prob < 1:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(
                        lstm_cell, output_keep_prob=self.keep_prob
                    )
                return lstm_cell

            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell()] * hidden_layer_num, state_is_tuple=True)

            self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # embedding layer
            with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
                # embedding = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embed_dim]),
                #                         trainable=False, name='embedding')
                # embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, embed_dim])
                # embedding_init = embedding.assign(embedding_placeholder)

                embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)

            if self.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            out_put = []
            state = self._initial_state
            with tf.variable_scope("LSTM_layer"):
                for time_step in range(num_step):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    out_put.append(cell_output)

            out_put = out_put * self.mask_x[:, :, None]

            with tf.name_scope("mean_pooling_layer"):

                out_put = tf.reduce_sum(out_put, 0) / (tf.reduce_sum(self.mask_x, 0)[:, None])

            with tf.name_scope("Softmax_layer_and_output"):
                softmax_w = tf.get_variable("softmax_w", [hidden_neural_size, class_num], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [class_num], dtype=tf.float32)
                self.logits = tf.matmul(out_put, softmax_w) + softmax_b

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=self.target)
                self.cost = tf.reduce_mean(self.loss)

            with tf.name_scope("accuracy"):
                self.prediction = tf.argmax(self.logits, 1)
                correct_prediction = tf.equal(self.prediction, self.target)
                self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

            # add summary
            loss_summary = tf.summary.scalar("loss", self.cost)
            # add summary
            accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

            if not is_training:
                return

            self.globle_step = tf.Variable(0, name="globle_step", trainable=False)
            self.lr = tf.Variable(0.0, trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                              config.max_grad_norm)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries)

            self.summary = tf.summary.merge([loss_summary, accuracy_summary, self.grad_summaries_merged])

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer.apply_gradients(zip(grads, tvars))
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self.lr, self.new_lr)

        if usemydata:
            self.keep_prob=config.keep_prob
            self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)

            class_num=config.class_num
            hidden_neural_size=config.hidden_neural_size
            vocabulary_size=config.vocabulary_size
            embed_dim=config.embed_dim
            hidden_layer_num=config.hidden_layer_num

            num_step = config.num_step
            if not usemydata:
                self.input_data=tf.placeholder(tf.int32,[None,num_step])
                self.target = tf.placeholder(tf.int64,[None])
                self.mask_x = tf.placeholder(tf.float32,[num_step,None])
            else:
                if pretrain_word2vec:
                    self.input_data = tf.placeholder(tf.int64, [None, num_step])  # 单词的ids， 得是int64
                else:
                    self.input_data = tf.placeholder(tf.float32, [None, num_step, embed_dim]) # num_step也就是句子长度，本应该也是变化的，但无法实现，就取最大句子长度
                self.seq_length = tf.placeholder(tf.int64, [None]) # 每个batch的句子长度
                self.target = tf.placeholder(tf.int64, [None]) # None
                # self.mask_x = tf.placeholder(tf.float32,[embed_dim, num_step, None])


            self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
            self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)



            #build LSTM network
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0,
                                                         state_is_tuple=True, # 以元祖形式state=(c,h)
                                                         reuse=tf.get_variable_scope().reuse)
                if self.keep_prob<1: # 有1-keep_prob的概率丢弃，防止过拟合
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(
                        lstm_cell, output_keep_prob=self.keep_prob
                    )
                return lstm_cell

            # 定义多层LSTM网络 :MultiRNNCell([cell()]*num_layers)改为MultiRNNCell([cell for _ in range(nums)])
            cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(hidden_layer_num)], state_is_tuple=True)
            # cell = lstm_cell()
            self._initial_state = cell.zero_state(self.batch_size,dtype=tf.float32)

            # 初始化词向量表示，后续训练过程会更新，embedding layer
            # 输入：batch_size * max_sentence_length
            # 词库：vocabulary_size * embed_dim
            # 输出：batch_size * max_sentence_length * embed_dim
            # 例如：
            #   input = '我 喜欢 宝马'
            #   假设词典的索引为：'我':0, '你':1, '喜欢':2, '讨厌':3, '奔驰':4, '宝马':5
            #       '我':   1 0 0 0 0 0
            #       '你':   0 1 0 0 0 0
            #       '喜欢': 0 0 1 0 0 0
            #       '讨厌': 0 0 0 1 0 0
            #       '奔驰': 0 0 0 0 1 0
            #       '宝马': 0 0 0 0 0 1
            #   会做一次每个词索引编码输入进来input = [0, 2, 5]
            #   embedding_lookup 的 输出为：inputs = [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,0,1]]
            if not usemydata:
                with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
                    embedding = tf.get_variable("embedding", [vocabulary_size, embed_dim], dtype=tf.float32)
                    inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            else:
                inputs = self.input_data

            if pretrain_word2vec:
                W = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, embed_dim]),
                                trainable=False, name="W")
                self.embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, embed_dim])
                self.embedding_init = W.assign(self.embedding_placeholder)

                with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
                    embedding_placeholder = tf.get_variable("embedding_placeholder", [vocabulary_size, embed_dim], dtype=tf.float32)
                    inputs = tf.nn.embedding_lookup(embedding_placeholder, self.input_data)


            # print('\t', inputs)
            if self.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            print('this is running model.')

            if not using_api:
                # 定义输出层
                out_put = []
                state = self._initial_state
                with tf.variable_scope("LSTM_layer"):
                    for time_step in range(num_step):
                        if time_step > 0: tf.get_variable_scope().reuse_variables() # 由于同一个RNN作用域内使用for，需要多次调用LSTM循环，所以需要保证网络节点参数的重复使用
                        (cell_output,state)=cell(inputs[:,time_step,:], state) # 对LSTM网络的1次循环调用
                        out_put.append(cell_output)

                # out_put = out_put * self.mask_x[:, :, :, None]
                # with tf.name_scope("mean_pooling_layer"):
                #
                #     out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:, :, None])
                # out_put = tf.reshape(out_put, [-1, num_step, embed_dim]) # 应该是每一个step，也就是每个词，有一个输出


                # 这种写的是错误的，因为outputs本身得到的是所有的结果，而我们只需要最后的结果，如果是边长的话，只需要取得最后的state.h作为final-output即可
                # outputs = tf.reshape(tf.concat(out_put, 1), [-1, embed_dim, hidden_neural_size]) # 1表示在第二维上相连
                # with tf.name_scope("mean_pooling_layer"):
                #     outputs = tf.reduce_sum(outputs, 1) # 在embed_dim一列合并, keep_dims=False降维
                if not usemydata:
                    out_put = out_put * self.mask_x[:, :, None]

                    with tf.name_scope("mean_pooling_layer"):
                        outputs = tf.reduce_sum(out_put, 0) / (tf.reduce_sum(self.mask_x, 0)[:, None])
                else:
                    outputs = out_put[-1] # 不需要所有的中间step，只需要最终的final-step即可

                # 图片为[28, 28]像素，文字为：[length, word2vec]，所以两者是对应的
                # inputs = [n_step, n_input] = [28, 28] = [length, word2vec]
                # input: [batch_size, inputs]
                # hidden: hidden_size
                # output: [batch_size, hidden_size]
                # weight and biases: w = [hidden_size, classes], b = [classes]
                    # softmax: weight*output + b
                # predict: [batch_size, classes]
            else:
                # using api
                # outputs: 保存了每次step隐层的输出:[batch_size, max_step, output_size=hidden_size]
                # final_states: 保存了最后一次状态的输出（c, h）:[batch_size, state_size=hidden_size]
                # 例如：
                #   传入sequence_length：max=20，len=10，那么后10的padding就不计算了，outputs直接置0
                #   而final_states则保持第10step时刻的状态，所以此时应该用final_states更方便，不然用outputs接得用mask！

                # seq_length = [len(i) for i in inputs] # 'Tensor' object is not iterable.
                out_put, final_states = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=self.input_data,
                    sequence_length=self.seq_length, # 每个batch中每个sentence的长度
                    initial_state=self._initial_state,
                    dtype=tf.float32,
                    time_major=False
                )
                outputs = final_states[hidden_layer_num-1].h # rnn的输出就是state中的h # output[-1]
                # outputs = final_states.h
            # 定义softmax层 - [batch_size, classes]
            with tf.name_scope("Softmax_layer_and_output"):
                softmax_w = tf.get_variable("softmax_w", [hidden_neural_size, class_num], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [class_num], dtype=tf.float32)
                self.logits = tf.matmul(outputs, softmax_w)+softmax_b

            # 定义损失函数 - logits=[batch_size, classes] vs target=[batch_size, classes]
            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits+1e-10,
                    labels=self.target)
                self.cost = tf.reduce_mean(self.loss)

            with tf.name_scope("accuracy"):
                self.prediction = tf.argmax(self.logits,1)  # 取得预测的label
                correct_prediction = tf.equal(self.prediction, self.target)
                self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))  # 计数
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")  # 准确率

            #add summary
            loss_summary = tf.summary.scalar("loss", self.cost)
            #add summary
            accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

            if not is_training:
                return



            # 定义优化模块，根据损失函数，计算梯度，使用梯度下降法更新网络
            self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
            self.lr = tf.Variable(0.0,trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm) # 梯度裁剪，防止梯度爆炸

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries)

            self.summary =tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])

            optimizer = tf.train.GradientDescentOptimizer(self.lr) # 梯度下降法
            optimizer.apply_gradients(zip(grads, tvars))
            self.train_op=optimizer.apply_gradients(zip(grads, tvars))

            self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
            self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update, feed_dict={self.new_lr:lr_value})
    def assign_new_batch_size(self,session, batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})

    def assign_pretrain_embedding(self, session, embedding_value):
        session.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding_value})

