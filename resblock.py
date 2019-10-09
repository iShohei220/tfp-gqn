import tensorflow as tf

tfkl = tf.keras.layers


class TimeDistributedResBlock2D(tfkl.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TimeDistributedResBlock2D, self).__init__()
        filters1, filters2 = filters
        self.conv2a = tfkl.TimeDistributed(
            tfkl.Conv2D(filters=filters1,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        activation=tf.nn.relu,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)
        )
        self.conv2b = tfkl.TimeDistributed(
            tfkl.Conv2D(filters=filters2,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                        dilation_rate=dilation_rate,
                        activation=None,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)
        )
        self.activation = activation

    def call(self, inputs):
        x = self.conv2a(inputs)
        x = self.conv2b(x)

        return self.activation(inputs + x)
