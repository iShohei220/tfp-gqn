from datetime import datetime
import sys
sys.path.append('gqn-datasets')

from absl import app
from absl import flags
import tensorflow as tf

from callback import PixelVarianceScheduler, TensorBoardImage
from data_reader import data_reader
from model import GQN

tfk = tf.keras

FLAGS = flags.FLAGS
# Hyper-parameters
flags.DEFINE_float('mu_i', 5e-4, 'Initial learning rate')
flags.DEFINE_float('mu_f', 5e-5, 'Final learning rate')
flags.DEFINE_integer('n_mu', int(1.6e6), 'Number of anealing steps of learning rate')
flags.DEFINE_float('beta_1', 0.9, 'Adam exponential decay rate for the 1st moment estimates')
flags.DEFINE_float('beta_2', 0.999, 'Adam exponential decay rate for the 2nd moment estimates')
flags.DEFINE_float('epsilon', 1e-8, 'Adam regularisation parameter')
flags.DEFINE_float('sigma_i', 2.0, 'Initial pixel standard-deviation')
flags.DEFINE_float('sigma_f', 0.7, 'Final pixel standard-deviation')
flags.DEFINE_integer('n_sigma', int(2e5), 'Number of anealing steps of pixel standard-deviation')
flags.DEFINE_integer('L', 12, 'Number of generative layers')
flags.DEFINE_integer('B', 36, 'Number of scenes over which each weight update is computed')
flags.DEFINE_integer('S_max', int(2e6), 'Maximum number of training steps')
flags.DEFINE_enum('representation', 'pool', ['pyramid', 'tower', 'pool'], 'Representation network architecture')
flags.DEFINE_bool('shared_core', False, 'Whether to share the weights of the cores across generation steps')
# Experimental setups
flags.DEFINE_enum(
        'D', None,
        ['rooms_ring_camera',
         'rooms_free_camera_no_object_rotations',
         'rooms_free_camera_with_object_rotations',
         'jaco',
         'shepard_metzler_5_parts',
         'shepard_metzler_7_parts',
         'mazes'],
        'Dataset')
flags.DEFINE_string('root_path', None, "Path to dataset's root folder")
flags.DEFINE_string('log_dir', '/tf/logs/' + datetime.now().strftime('%Y%m%d-%H%M%S'), 'Log directory')
flags.DEFINE_string('validation_freq', 100,
                    'How many training epochs to run '
                    'before a new validation run is performed')
flags.DEFINE_string('validation_context_size', None,
                    'Context size in test time')
flags.DEFINE_string('save_freq', 50000,
                    'The callback saves the model at end of a batch '
                    'at which this many samples have been seen since last saving')
flags.DEFINE_integer('seed', None, 'Seed')

def lr_schedule(epoch):
    return max(FLAGS.mu_f + (FLAGS.mu_i - FLAGS.mu_f) * (1 - epoch / FLAGS.n_mu), FLAGS.mu_f)

if __name__ == '__main__':
    dataset = data_reader(dataset=FLAGS.D,
                          root=FLAGS.root_path,
                          mode='train',
                          batch_size=FLAGS.B,
                          custom_frame_size=64,
                          seed=FLAGS.seed)
    validation_data = data_reader(dataset=FLAGS.D,
                                  root=FLAGS.root_path,
                                  mode='test',
                                  batch_size=FLAGS.B,
                                  custom_frame_size=64,
                                  shuffle_buffer_size=256,
                                  seed=FLAGS.seed)
    
    test_inputs, test_target = next(iter(validation_data))
        
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = GQN(FLAGS.representation, FLAGS.shared_core, FLAGS.L)
        optimizer = tfk.optimizers.Adam(
            learning_rate=mu_i,
            beta_1=FLAGS.beta_1,
            beta_2=FLAGS.beta_2,
            epsilon=FLAGS.epsilon)
        negative_log_likelihood = lambda x_q, rv_x_q: -rv_x_q.log_prob(x_q)
        model.compile(optimizer=optimizer, loss=negative_log_likelihood)
    
    checkpoint_path = FLAGS.log_dir + '/model/cp-{epoch:07d}.ckpt'
    callbacks = [tfk.callbacks.TensorBoard(log_dir=FLAGS.log_dir),
                 tfk.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               save_weights_only=True,
                                               verbose=1, save_freq=FLAGS.save_freq),
                 tfk.callbacks.LearningRateScheduler(lr_schedule),
                 PixelVarianceScheduler(FLAGS.sigma_i, FLAGS.sigma_f, FLAGS.n_sigma),
                 TensorBoardImage(FLAGS.log_dir, test_inputs, test_target, FLAGS.validation_freq)]
    model.fit(dataset,
              verbose=0,
              epochs=FLAGS.S_max,
              callbacks=callbacks,
              validation_data=validation_data,
              steps_per_epoch=1,
              validation_steps=1,
              validation_freq=FLAGS.validation_freq)