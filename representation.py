import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class Pyramid(tfkl.Layer):
    def __init__(self):
        super(Pyramid, self).__init__()
        self.net = tfk.Sequential([
            tfkl.TimeDistributed(
                tfkl.Conv2D(32, 2, strides=2, padding='valid', activation=tf.nn.relu)
            ),
            tfkl.TimeDistributed(
                tfkl.Conv2D(64, 2, strides=2, padding='valid', activation=tf.nn.relu)
            ),
            tfkl.TimeDistributed(
                tfkl.Conv2D(128, 2, strides=2, padding='valid', activation=tf.nn.relu)
            ),
            tfkl.TimeDistributed(
                tfkl.Conv2D(256, 8, strides=8, padding='valid', activation=tf.nn.relu)
            )
        ])

    def call(self, inputs):
        x, v = inputs
        v = tf.expand_dims(tf.expand_dims(v, 2), 2)
        v = tf.tile(v, [1, 1, 64, 64, 1])
        r = tf.concat([v, x], -1)

        return self.net(r)


class Tower(tfkl.Layer):
    def __init__(self):
        super(Tower, self).__init__()
        self.conv_1 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 2, strides=2, padding='valid', activation=tf.nn.relu)
        )
        self.block_1 = TimeDistributedResBlock2D((128, 256), 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv_2 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 2, strides=2, padding='valid', activation=tf.nn.relu)
        )
        self.block_2 = TimeDistributedResBlock2D((128, 256+7), 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv_3 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 3, strides=1, padding='same', activation=tf.nn.relu),
        )
        self.conv_4 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 1, strides=1, padding='same', activation=tf.nn.relu)
        )

    def call(self, inputs):
        x, v = inputs
        x = self.conv_1(x)
        x = self.block_1(x)
        x = self.conv_2(x)
        v = tf.expand_dims(tf.expand_dims(v, 2), 2)
        v = tf.tile(v, [1, 1, 16, 16, 1])
        r = tf.concat([x, v], -1)
        r = self.block_2(r)
        r = self.conv_3(r)
        r = self.conv_4(r)

        return r


class Pool(tfkl.Layer):
    def __init__(self):
        super(Pool, self).__init__()
        self.conv_1 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 2, strides=2, padding='valid', activation=tf.nn.relu)
        )
        self.block_1 = TimeDistributedResBlock2D((128, 256), 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv_2 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 2, strides=2, padding='valid', activation=tf.nn.relu)
        )
        self.block_2 = TimeDistributedResBlock2D((128, 256+7), 3, strides=1, padding='same', activation=tf.nn.relu)
        self.conv_3 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 3, strides=1, padding='same', activation=tf.nn.relu),
        )
        self.conv_4 = tfkl.TimeDistributed(
            tfkl.Conv2D(256, 1, strides=1, padding='same', activation=tf.nn.relu)
        )
        self.pool = tfkl.TimeDistributed(tfkl.AveragePooling2D(16))

    def call(self, inputs):
        x, v = inputs
        x = self.conv_1(x)
        x = self.block_1(x)
        x = self.conv_2(x)
        v = tf.expand_dims(tf.expand_dims(v, 2), 2)
        v = tf.tile(v, [1, 1, 16, 16, 1])
        r = tf.concat([x, v], -1)
        r = self.block_2(r)
        r = self.conv_3(r)
        r = self.conv_4(r)
        r = self.pool(r)

        return r
