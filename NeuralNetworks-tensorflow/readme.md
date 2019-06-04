# 深度学习-神经网络-tensorflow实践
# 目录架构
1. CNN目录-(CNN模型)<br>
    * alex_net_model目录-(AlexNet CNN模型)<br>
        * alex_net_cnn.py-(构建alex_net网络，分类imageNet网络上的图片以及cifar10图片)
        * cifar10_input_for_alex_net.py-(将cifar10_input改造成可训练alex_net网络的格式)
        * constants.py-(常量)
    * cifar10_model目录-(Cifar10 CNN模型)<br>
        * cifar10.py-(构建分类cifar10图片的cnn网络)
        * cifar10_cnn.py-(构建分类cifar10图片的cnn网络)
    * res_net_model目录-(resnet 模型)<br>
        * resnet.py-(构建resnet网络模型，分类imageNet网络上的图片以及cifar10图片)
        * cifar10_input_for_resnet.py-(将cifar10_input改造成可训练resnet网络的格式)
2. datasets_learn目录-(tf.data用法学习)<br>
    * practice.py-(实践测试tf.data的用法，包括迭代器使用、解析tfRecords数据、解析csv数据、解析image数据等)
3. estimator_learn目录-(Estimator模块学习)
    * census_data_download.py-(下载census_data)
    * DNN_classifier.py-(学习使用tensorflow的feature_columns和DNN_classifier
    estimator的用法)
    * DNN_linear_combined_classifier.py-(使用tf.estimator.DNNLinearCombinedClassifier
    分类census数据)
4. image_net_origin_files目录<br>
    * 从ImageNet下载下来的几个分类图片的urls文件，每个分类大概有1000个图片
    * 下载器会把原始图片保存在该目录
5. MNIST_data目录-(手写识别训练和测试数据)<br>
6. model_saver目录-(模型训练结果保存)<br>
    * 保存各个模型的model saver数据文件
7. plots目录-(训练结果的plots图片)<br>
    * Bi_RNN_train_mnist_50_epochs.png-(mnist训练双向RNN 50个epochs)
    * CIFAR10-CNN.png-(cifar10数据训练cnn网络)
    * cifar10_train_alex_net_top_1_100_epochs.png-
    (cifar10数据训练alex_net网络100个epochs)
    * cifar10_train_alex_net_top_1_150_epochs.png-
    (cifar10数据训练alex_net网络150个epochs)
    * cifar10_train_alex_net_top_1_300_epochs.png-
    (cifar10数据训练alex_net网络300个epochs)
    * cifar10_train_pre-trained_resnet101_top_1_50_epochs.png-
    (cifar10数据训练已经预训练过的resnet101模型50个epochs)
    * cifar10_train_resnet101_top_1_150_epochs.png-
    (cifar10数据训练resnet101模型150个epochs)
    * image_net_train_alex_net_top_1.png-(image_net数据训练alex_net网络)
    * MLP.png-(mnist训练一个多层全连接网络模型)
    * reinforcement_learning_value_network_for_grid_world.png-
    (使用grid_world小游戏训练Deep Q-Learning network)
    * softmax_regression.png-(mnist训练softmax回归分类模型)
    * words2vec_by_skip_gram.png-(skip-gram算法训练出的words向量分布图)
8. reinforcement_learning目录-深度强化学习<br>
    * grid_world.py-(grid_world游戏模型)
    * policy_network_for_cart_pole.py-(构建强化学习策略网络模型来玩cart_pole游戏)
    * value_network_for_grid_world.py-(构建强化学习估值网络Deep Q-learning
    network来玩grid_world游戏)
9. RNN目录-(RNN模型)<br>
    * Bi_RNN_model目录-(双向RNN模型)<br>
        * bi_rnn_train_mnist.py-(构建双向RNN网络，并使用mnist数据来测试)
    * RNN_ptb_model目录-(使用LSTM训练的Ptb 模型)<br>
        * ptb_language_model_by_lstm.py-(构建LSTM网络训练ptb模型)
        * reader.py-(words读取者)
        * ptb_model_tensor_board.png-(tensor board画出ptb model graph图)
    * RNN_Word2Vec目录-(使用skip-gram算法实现word2vec)<br>
        * word2vec_by_skip_gram.py-(使用skip-gram算法实现word2vec模型)
10. tools目录-(工具包)<br>
    * bz2_decompress.py-(bz2文件自动检索解压器)
    * image_net_downloader.py-(image_net图片下载器，支持多线程)
    * img_net_input.py-(将图片转成一批数组，并进行增强处理，最后发现跑不动，只记录一下思考过程)
    * image_net_tf_records_reader.py-(从tfRecords文件中读取image_net图片数据，并进行增强处理,
    alex_net模型读取image_net数据时使用的该reader)
    * image_net_tf_records_writer.py-(将image_net图片转成tensor并保存到tfRecords中)
    * words_downloader_and_reader.py-(word2vec使用到的words downloader and reader)
11. words_data目录-(words_vec模型使用的训练语言数据)<br>
12. MLP.py-(简单多层全连接神经网络分类器)<br>
13. auto_encoder.py-（自动编码器）<br>
14. softmax_regression.py-(根据softmax 回归的简单分类器)<br>
