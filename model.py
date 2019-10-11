import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2DCell

from representation import Pyramid, Tower, Pool

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers


class GQN(tfk.Model):
    def __init__(self, representation='pool', shared_core=False, L=12):
        super(GQN, self).__init__()
        if representation == 'pyramid':
            self.phi = Pyramid()
        elif representation == 'tower':
            self.phi = Tower()
        else:
            self.phi = Pool()
        self.shared_core = shared_core
        self.L = L

        self.downsample_x = tfkl.Conv2D(3, 4, strides=4, padding='valid', use_bias=False)
        self.upsample_v = tfkl.Conv2DTranspose(7, 16, strides=16, padding='valid', use_bias=False)
        self.upsample_r = tfkl.Conv2DTranspose(256, 16, strides=16, padding='valid', use_bias=False)\
            if representation != 'tower' else None
        self.downsample_u = tfkl.Conv2D(128, 4, strides=4, padding='valid', use_bias=False)
        self.upsample_h = tfkl.Conv2DTranspose(128, 4, strides=4, padding='valid', use_bias=False)

        self.eta_pi = tfk.Sequential([
            tfkl.Conv2D(3 + 3, 5, strides=1, padding='same'),
            tfkl.Flatten()
        ])
        self.eta_e = tfk.Sequential([
            tfkl.Conv2D(3 + 3, 5, strides=1, padding='same'),
            tfkl.Flatten()
        ])
        self.eta_g = tfk.Sequential([
            tfkl.Conv2D(3, 1, strides=1, padding='same'),
        ])

        self.inference_core = ConvLSTM2DCell(128, 5, strides=1, padding='same')\
            if shared_core else [ConvLSTM2DCell(128, 5, strides=1, padding='same') for _ in range(L)]
        self.generation_core = ConvLSTM2DCell(128, 5, strides=1, padding='same')\
            if shared_core else [ConvLSTM2DCell(128, 5, strides=1, padding='same') for _ in range(L)]

        self.sigma = 2.0

    def call(self, inputs):
        x, v, v_q, x_q = inputs

        # Scene encoder
        r = self.phi([x, v])
        r = tf.reduce_sum(r, 1)

        x_q_ = self.downsample_x(x_q)
        v_q_ = tf.expand_dims(tf.expand_dims(v_q, 1), 1)
        v_q_ = self.upsample_v(v_q_)
        r_ = self.upsample_r(r) if self.upsample_r is not None else r

        # Initial state
        h_g = c_g = h_e = c_e = tf.tile(tf.reduce_sum(tf.zeros_like(r), axis=(1, 2, 3), keepdims=True), [1, 16, 16, 128])
        u = tf.tile(tf.reduce_sum(tf.zeros_like(r), axis=(1, 2, 3), keepdims=True), [1, 64, 64, 128])

        for l in range(self.L):
            # Prior factor
            pi = tfpl.IndependentNormal([16, 16, 3])(self.eta_pi(h_g))

            # Inference state update
            u_ = self.downsample_u(u)
            _, (h_e, c_e) = self.inference_core(tf.concat([x_q_, v_q_, r_, h_g, u_], -1), [h_e, c_e])\
                if self.shared_core else self.inference_core[l](tf.concat([x_q_, v_q_, r_, h_g, u_], -1), [h_e, c_e])

            # Posterior factor
            q = tfpl.IndependentNormal(
                [16, 16, 3],
                activity_regularizer=tfpl.KLDivergenceRegularizer(pi,
                                                                  use_exact_kl=True,
                                                                  weight=1.0)
            )(self.eta_e(h_e))
            # Posterior sample
            z = q.sample()

            # Generator state update
            _, (h_g, c_g) = self.generation_core(tf.concat([v_q_, r_, z], -1), [h_g, c_g])\
                if self.shared_core else self.generation_core[l](tf.concat([v_q_, r_, z], -1), [h_g, c_g])
            u += self.upsample_h(h_g)

        rv_x_q = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Independent(
                tfd.Normal(loc=t[0], scale=t[1])
            ),
            convert_to_tensor_fn=tfd.Independent.mean)([self.eta_g(u), self.sigma])

        return rv_x_q

    @tf.function
    def generate(self, inputs):
        x, v, v_q = inputs

        # Scene encoder
        r = self.phi([x, v])
        r = tf.reduce_sum(r, 1)

        v_q_ = tf.expand_dims(tf.expand_dims(v_q, 1), 1)
        v_q_ = self.upsample_v(v_q_)
        r_ = self.upsample_r(r) if self.upsample_r is not None else r

        # Initial state
        h_g = c_g = tf.tile(tf.reduce_sum(tf.zeros_like(r), axis=(1, 2, 3), keepdims=True), [1, 16, 16, 128])
        u = tf.tile(tf.reduce_sum(tf.zeros_like(r), axis=(1, 2, 3), keepdims=True), [1, 64, 64, 128])

        for l in range(self.L):
            # Prior factor
            pi = tfpl.IndependentNormal([16, 16, 3])(self.eta_pi(h_g))
            # Prior sample
            z = pi.sample()

            # Generator state update
            _, (h_g, c_g) = self.generation_core(tf.concat([v_q_, r_, z], -1), [h_g, c_g])\
                if self.shared_core else self.generation_core[l](tf.concat([v_q_, r_, z], -1), [h_g, c_g])
            u += self.upsample_h(h_g)

        rv_x_q = tfpl.DistributionLambda(
            make_distribution_fn=lambda t: tfd.Independent(
                tfd.Normal(loc=t[0], scale=t[1])
            ),
            convert_to_tensor_fn=tfd.Independent.mean)([self.eta_g(u), self.sigma])

        return rv_x_q
