import tensorflow as tf


class PixelVarianceScheduler(tf.keras.callbacks.Callback):
    def __init__(self, sigma_i, sigma_f, n_sigma):
        super(PixelVarianceScheduler, self).__init__()
        
        self.sigma_i = sigma_i
        self.sigma_f = sigma_f
        self.n_sigma = n_sigma
        
    def on_epoch_begin(self, epoch, logs=None):
        self.model.sigma = max(self.sigma_f + (self.sigma_i - self.sigma_f) * (1 - epoch / self.n_sigma), self.sigma_f)
        

class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, inputs, target, validation_freq):
        self.inputs = inputs
        self.target = target
        self.validation_freq
        
        self.file_writer_image = tf.summary.create_file_writer(log_dir + '/image')
        with self.file_writer_image.as_default():
            tf.summary.image("Ground Truth", target, 0)
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.validation_freq == 0:
            self._log_image(epoch + 1)
            
    def on_train_begin(self, logs):
        self._log_image(0)
                
    def _log_image(self, epoch):
        context_frames, context_cameras, query_camera, target = self.inputs
        target_gen = self.model.generate((context_frames, context_cameras, query_camera))
        with self.file_writer_image.as_default():
            tf.summary.image("Generation", target_gen, epoch)