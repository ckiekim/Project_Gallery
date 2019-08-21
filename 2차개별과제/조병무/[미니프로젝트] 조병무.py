import tensorflow as tf
import numpy as np

# CIFAR-100 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar100 import load_data


# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# CNN 모델을 정의합니다.
def build_CNN_classifier(x):
    # 입력 이미지
    x_image = x

    # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping)합니다.
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

    # 네번째 convolutional layer
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

    # 다섯번째 convolutional layer
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

    # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
    # 이를 384개의 특징들로 맵핑(maping)합니다.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits


# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = build_CNN_classifier(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    # 모든 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())

    # 10000 Step만큼 최적화를 수행합니다.
    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

            print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})

    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(10, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)

반복(Epoch): 0, 트레이닝 데이터 정확도: 0.000000, 손실 함수(loss): 15.557163
반복(Epoch): 100, 트레이닝 데이터 정확도: 0.015625, 손실 함수(loss): 2973929308160.000000
반복(Epoch): 200, 트레이닝 데이터 정확도: 0.007812, 손실 함수(loss): 53837415055360.000000
반복(Epoch): 300, 트레이닝 데이터 정확도: 0.007812, 손실 함수(loss): 669044235567104.000000
반복(Epoch): 400, 트레이닝 데이터 정확도: 0.015625, 손실 함수(loss): 1262445507117056.000000
반복(Epoch): 500, 트레이닝 데이터 정확도: 0.015625, 손실 함수(loss): 7375455279644672.000000
반복(Epoch): 600, 트레이닝 데이터 정확도: 0.007812, 손실 함수(loss): 25598005084160000.000000
반복(Epoch): 700, 트레이닝 데이터 정확도: 0.031250, 손실 함수(loss): 36256945681727488.000000
반복(Epoch): 800, 트레이닝 데이터 정확도: 0.015625, 손실 함수(loss): 115250808823480320.000000
반복(Epoch): 900, 트레이닝 데이터 정확도: 0.000000, 손실 함수(loss): 471021985246216192.000000
반복(Epoch): 1000, 트레이닝 데이터 정확도: 0.859375, 손실 함수(loss): nan
반복(Epoch): 1100, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 1200, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 1300, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 1400, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 1500, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
반복(Epoch): 1600, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 1700, 트레이닝 데이터 정확도: 0.867188, 손실 함수(loss): nan
반복(Epoch): 1800, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 1900, 트레이닝 데이터 정확도: 0.867188, 손실 함수(loss): nan
반복(Epoch): 2000, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 2100, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 2200, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 2300, 트레이닝 데이터 정확도: 0.867188, 손실 함수(loss): nan
반복(Epoch): 2400, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 2500, 트레이닝 데이터 정확도: 0.937500, 손실 함수(loss): nan
반복(Epoch): 2600, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 2700, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 2800, 트레이닝 데이터 정확도: 0.882812, 손실 함수(loss): nan
반복(Epoch): 2900, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 3000, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
반복(Epoch): 3100, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 3200, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 3300, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 3400, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 3500, 트레이닝 데이터 정확도: 0.859375, 손실 함수(loss): nan
반복(Epoch): 3600, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 3700, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 3800, 트레이닝 데이터 정확도: 0.937500, 손실 함수(loss): nan
반복(Epoch): 3900, 트레이닝 데이터 정확도: 0.976562, 손실 함수(loss): nan
반복(Epoch): 4000, 트레이닝 데이터 정확도: 0.882812, 손실 함수(loss): nan
반복(Epoch): 4100, 트레이닝 데이터 정확도: 0.960938, 손실 함수(loss): nan
반복(Epoch): 4200, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 4300, 트레이닝 데이터 정확도: 0.875000, 손실 함수(loss): nan
반복(Epoch): 4400, 트레이닝 데이터 정확도: 0.960938, 손실 함수(loss): nan
반복(Epoch): 4500, 트레이닝 데이터 정확도: 0.937500, 손실 함수(loss): nan
반복(Epoch): 4600, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 4700, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 4800, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 4900, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 5000, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 5100, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 5200, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
반복(Epoch): 5300, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 5400, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 5500, 트레이닝 데이터 정확도: 0.867188, 손실 함수(loss): nan
반복(Epoch): 5600, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 5700, 트레이닝 데이터 정확도: 0.875000, 손실 함수(loss): nan
반복(Epoch): 5800, 트레이닝 데이터 정확도: 0.875000, 손실 함수(loss): nan
반복(Epoch): 5900, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 6000, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 6100, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 6200, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 6300, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 6400, 트레이닝 데이터 정확도: 0.953125, 손실 함수(loss): nan
반복(Epoch): 6500, 트레이닝 데이터 정확도: 0.976562, 손실 함수(loss): nan
반복(Epoch): 6600, 트레이닝 데이터 정확도: 0.953125, 손실 함수(loss): nan
반복(Epoch): 6700, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 6800, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 6900, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 7000, 트레이닝 데이터 정확도: 0.875000, 손실 함수(loss): nan
반복(Epoch): 7100, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 7200, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 7300, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 7400, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 7500, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 7600, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
반복(Epoch): 7700, 트레이닝 데이터 정확도: 0.882812, 손실 함수(loss): nan
반복(Epoch): 7800, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 7900, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 8000, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 8100, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 8200, 트레이닝 데이터 정확도: 0.867188, 손실 함수(loss): nan
반복(Epoch): 8300, 트레이닝 데이터 정확도: 0.851562, 손실 함수(loss): nan
반복(Epoch): 8400, 트레이닝 데이터 정확도: 0.835938, 손실 함수(loss): nan
반복(Epoch): 8500, 트레이닝 데이터 정확도: 0.937500, 손실 함수(loss): nan
반복(Epoch): 8600, 트레이닝 데이터 정확도: 0.953125, 손실 함수(loss): nan
반복(Epoch): 8700, 트레이닝 데이터 정확도: 0.914062, 손실 함수(loss): nan
반복(Epoch): 8800, 트레이닝 데이터 정확도: 0.882812, 손실 함수(loss): nan
반복(Epoch): 8900, 트레이닝 데이터 정확도: 0.875000, 손실 함수(loss): nan
반복(Epoch): 9000, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
반복(Epoch): 9100, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 9200, 트레이닝 데이터 정확도: 0.898438, 손실 함수(loss): nan
반복(Epoch): 9300, 트레이닝 데이터 정확도: 0.929688, 손실 함수(loss): nan
반복(Epoch): 9400, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 9500, 트레이닝 데이터 정확도: 0.937500, 손실 함수(loss): nan
반복(Epoch): 9600, 트레이닝 데이터 정확도: 0.921875, 손실 함수(loss): nan
반복(Epoch): 9700, 트레이닝 데이터 정확도: 0.906250, 손실 함수(loss): nan
반복(Epoch): 9800, 트레이닝 데이터 정확도: 0.890625, 손실 함수(loss): nan
반복(Epoch): 9900, 트레이닝 데이터 정확도: 0.945312, 손실 함수(loss): nan
테스트 데이터 정확도: 0.920000


import tensorflow as tf
import numpy as np
# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar10 import load_data
# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
# CNN 모델을 정의합니다.
def build_CNN_classifier(x):
    # 입력 이미지
    x_image = x
    # 첫번째 convolutional layer - 하나의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping)합니다.
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    # 첫번째 Pooling layer
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    # 두번째 pooling layer.
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 세번째 convolutional layer
    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)
    # 네번째 convolutional layer
    W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)
    # 다섯번째 convolutional layer
    W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
    b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)
    # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
    # 이를 384개의 특징들로 맵핑(maping)합니다.
    W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
    h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
    W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_pred = tf.nn.softmax(logits)
    return y_pred, logits
# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)
# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = build_CNN_classifier(x)
# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
    # 모든 변수들을 초기화한다.
    sess.run(tf.global_variables_initializer())
    # 10000 Step만큼 최적화를 수행합니다.
    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())
        # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
        # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
    # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(10, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("테스트 데이터 정확도: %f" % test_accuracy)

반복(Epoch): 0, 트레이닝 데이터 정확도: 0.109375, 손실 함수(loss): 90.088570
반복(Epoch): 100, 트레이닝 데이터 정확도: 0.195312, 손실 함수(loss): 2.235763
반복(Epoch): 200, 트레이닝 데이터 정확도: 0.289062, 손실 함수(loss): 1.890145
반복(Epoch): 300, 트레이닝 데이터 정확도: 0.406250, 손실 함수(loss): 1.576636
반복(Epoch): 400, 트레이닝 데이터 정확도: 0.421875, 손실 함수(loss): 1.556597
반복(Epoch): 500, 트레이닝 데이터 정확도: 0.539062, 손실 함수(loss): 1.336723
반복(Epoch): 600, 트레이닝 데이터 정확도: 0.507812, 손실 함수(loss): 1.344029
반복(Epoch): 700, 트레이닝 데이터 정확도: 0.500000, 손실 함수(loss): 1.511029
반복(Epoch): 800, 트레이닝 데이터 정확도: 0.625000, 손실 함수(loss): 1.292868
반복(Epoch): 900, 트레이닝 데이터 정확도: 0.492188, 손실 함수(loss): 1.298407
반복(Epoch): 1000, 트레이닝 데이터 정확도: 0.585938, 손실 함수(loss): 1.186696
반복(Epoch): 1100, 트레이닝 데이터 정확도: 0.492188, 손실 함수(loss): 1.331959
반복(Epoch): 1200, 트레이닝 데이터 정확도: 0.601562, 손실 함수(loss): 1.147280
반복(Epoch): 1300, 트레이닝 데이터 정확도: 0.656250, 손실 함수(loss): 0.998883
반복(Epoch): 1400, 트레이닝 데이터 정확도: 0.687500, 손실 함수(loss): 0.839435
반복(Epoch): 1500, 트레이닝 데이터 정확도: 0.531250, 손실 함수(loss): 1.211388
반복(Epoch): 1600, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 1.004663
반복(Epoch): 1700, 트레이닝 데이터 정확도: 0.679688, 손실 함수(loss): 0.885975
반복(Epoch): 1800, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 0.964106
반복(Epoch): 1900, 트레이닝 데이터 정확도: 0.656250, 손실 함수(loss): 0.974774
반복(Epoch): 2000, 트레이닝 데이터 정확도: 0.578125, 손실 함수(loss): 1.074664
반복(Epoch): 2100, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 1.005914
반복(Epoch): 2200, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 1.040246
반복(Epoch): 2300, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 0.844962
반복(Epoch): 2400, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.875315
반복(Epoch): 2500, 트레이닝 데이터 정확도: 0.718750, 손실 함수(loss): 0.854545
반복(Epoch): 2600, 트레이닝 데이터 정확도: 0.640625, 손실 함수(loss): 1.012000
반복(Epoch): 2700, 트레이닝 데이터 정확도: 0.703125, 손실 함수(loss): 0.875686
반복(Epoch): 2800, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 0.901704
반복(Epoch): 2900, 트레이닝 데이터 정확도: 0.632812, 손실 함수(loss): 0.980566
반복(Epoch): 3000, 트레이닝 데이터 정확도: 0.726562, 손실 함수(loss): 0.813753
반복(Epoch): 3100, 트레이닝 데이터 정확도: 0.742188, 손실 함수(loss): 0.812394
반복(Epoch): 3200, 트레이닝 데이터 정확도: 0.828125, 손실 함수(loss): 0.658245
반복(Epoch): 3300, 트레이닝 데이터 정확도: 0.789062, 손실 함수(loss): 0.634361
반복(Epoch): 3400, 트레이닝 데이터 정확도: 0.593750, 손실 함수(loss): 1.318053
반복(Epoch): 3500, 트레이닝 데이터 정확도: 0.656250, 손실 함수(loss): 0.885246
반복(Epoch): 3600, 트레이닝 데이터 정확도: 0.609375, 손실 함수(loss): 1.196858
반복(Epoch): 3700, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.810267
반복(Epoch): 3800, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 0.985981
반복(Epoch): 3900, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 0.946472
반복(Epoch): 4000, 트레이닝 데이터 정확도: 0.687500, 손실 함수(loss): 1.045523
반복(Epoch): 4100, 트레이닝 데이터 정확도: 0.617188, 손실 함수(loss): 1.072807
반복(Epoch): 4200, 트레이닝 데이터 정확도: 0.726562, 손실 함수(loss): 0.846877
반복(Epoch): 4300, 트레이닝 데이터 정확도: 0.750000, 손실 함수(loss): 0.726776
반복(Epoch): 4400, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.866100
반복(Epoch): 4500, 트레이닝 데이터 정확도: 0.757812, 손실 함수(loss): 0.743096
반복(Epoch): 4600, 트레이닝 데이터 정확도: 0.648438, 손실 함수(loss): 0.961608
반복(Epoch): 4700, 트레이닝 데이터 정확도: 0.609375, 손실 함수(loss): 1.116221
반복(Epoch): 4800, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 1.031736
반복(Epoch): 4900, 트레이닝 데이터 정확도: 0.609375, 손실 함수(loss): 1.127813
반복(Epoch): 5000, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.876368
반복(Epoch): 5100, 트레이닝 데이터 정확도: 0.757812, 손실 함수(loss): 0.781402
반복(Epoch): 5200, 트레이닝 데이터 정확도: 0.687500, 손실 함수(loss): 0.927764
반복(Epoch): 5300, 트레이닝 데이터 정확도: 0.648438, 손실 함수(loss): 0.985826
반복(Epoch): 5400, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.894600
반복(Epoch): 5500, 트레이닝 데이터 정확도: 0.671875, 손실 함수(loss): 0.999186
반복(Epoch): 5600, 트레이닝 데이터 정확도: 0.648438, 손실 함수(loss): 0.974703
반복(Epoch): 5700, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.859643
반복(Epoch): 5800, 트레이닝 데이터 정확도: 0.789062, 손실 함수(loss): 0.677530
반복(Epoch): 5900, 트레이닝 데이터 정확도: 0.679688, 손실 함수(loss): 0.958168
반복(Epoch): 6000, 트레이닝 데이터 정확도: 0.742188, 손실 함수(loss): 0.819204
반복(Epoch): 6100, 트레이닝 데이터 정확도: 0.750000, 손실 함수(loss): 0.777314
반복(Epoch): 6200, 트레이닝 데이터 정확도: 0.757812, 손실 함수(loss): 0.697239
반복(Epoch): 6300, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 1.009428
반복(Epoch): 6400, 트레이닝 데이터 정확도: 0.796875, 손실 함수(loss): 0.647557
반복(Epoch): 6500, 트레이닝 데이터 정확도: 0.703125, 손실 함수(loss): 0.850309
반복(Epoch): 6600, 트레이닝 데이터 정확도: 0.679688, 손실 함수(loss): 1.086260
반복(Epoch): 6700, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.996235
반복(Epoch): 6800, 트레이닝 데이터 정확도: 0.742188, 손실 함수(loss): 0.888403
반복(Epoch): 6900, 트레이닝 데이터 정확도: 0.632812, 손실 함수(loss): 0.888495
반복(Epoch): 7000, 트레이닝 데이터 정확도: 0.734375, 손실 함수(loss): 0.818786
반복(Epoch): 7100, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.839529
반복(Epoch): 7200, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.882286
반복(Epoch): 7300, 트레이닝 데이터 정확도: 0.679688, 손실 함수(loss): 0.854007
반복(Epoch): 7400, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.935512
반복(Epoch): 7500, 트레이닝 데이터 정확도: 0.718750, 손실 함수(loss): 0.872969
반복(Epoch): 7600, 트레이닝 데이터 정확도: 0.742188, 손실 함수(loss): 0.770165
반복(Epoch): 7700, 트레이닝 데이터 정확도: 0.773438, 손실 함수(loss): 0.613780
반복(Epoch): 7800, 트레이닝 데이터 정확도: 0.734375, 손실 함수(loss): 0.795672
반복(Epoch): 7900, 트레이닝 데이터 정확도: 0.703125, 손실 함수(loss): 0.739092
반복(Epoch): 8000, 트레이닝 데이터 정확도: 0.773438, 손실 함수(loss): 0.717612
반복(Epoch): 8100, 트레이닝 데이터 정확도: 0.718750, 손실 함수(loss): 0.977266
반복(Epoch): 8200, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.961987
반복(Epoch): 8300, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 1.025553
반복(Epoch): 8400, 트레이닝 데이터 정확도: 0.742188, 손실 함수(loss): 0.778923
반복(Epoch): 8500, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 0.738598
반복(Epoch): 8600, 트레이닝 데이터 정확도: 0.703125, 손실 함수(loss): 0.986140
반복(Epoch): 8700, 트레이닝 데이터 정확도: 0.664062, 손실 함수(loss): 0.946008
반복(Epoch): 8800, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 0.863580
반복(Epoch): 8900, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.804335
반복(Epoch): 9000, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 1.037005
반복(Epoch): 9100, 트레이닝 데이터 정확도: 0.804688, 손실 함수(loss): 0.631460
반복(Epoch): 9200, 트레이닝 데이터 정확도: 0.640625, 손실 함수(loss): 0.920637
반복(Epoch): 9300, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.861506
반복(Epoch): 9400, 트레이닝 데이터 정확도: 0.687500, 손실 함수(loss): 0.917404
반복(Epoch): 9500, 트레이닝 데이터 정확도: 0.710938, 손실 함수(loss): 1.053233
반복(Epoch): 9600, 트레이닝 데이터 정확도: 0.640625, 손실 함수(loss): 0.919674
반복(Epoch): 9700, 트레이닝 데이터 정확도: 0.656250, 손실 함수(loss): 0.885188
반복(Epoch): 9800, 트레이닝 데이터 정확도: 0.695312, 손실 함수(loss): 0.819427
반복(Epoch): 9900, 트레이닝 데이터 정확도: 0.750000, 손실 함수(loss): 0.831248
테스트 데이터 정확도: 0.680000