"""
This module includes a number of custom callbacks that aren't used or even do not work.
"""


import seaborn as sb
import tensorflow as tf
from matplotlib import pyplot as plt


class GetWeights(tf.keras.callbacks.Callback):  # Not Used
    #  https://stackoverflow.com/questions/70133704/how-to-get-weight-in-each-layer-and-epoch-then-save-in-file#answer-70136833
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        drop_out_index = 2
        for i, layer in enumerate(self.model.layers):
            if layer.weights and drop_out_index != i:
                w = layer.get_weights()[0]
                b = layer.get_weights()[1]
                heat_map = sb.heatmap(w)
                plt.show()
                print('Layer %s has weights of shape %s and biases of shape %s' % (i, np.shape(w), np.shape(b)))
                if epoch == 0:
                    # create array to hold weights and biases
                    self.weight_dict['w_' + str(i + 1)] = w
                    self.weight_dict['b_' + str(i + 1)] = b
                else:
                    # append new weights to previously-created weights array
                    self.weight_dict['w_' + str(i + 1)] = np.dstack(
                        (self.weight_dict['w_' + str(i + 1)], w))
                    # append new weights to previously-created weights array
                    self.weight_dict['b_' + str(i + 1)] = np.dstack(
                        (self.weight_dict['b_' + str(i + 1)], b))


class GradientPlotCallback(tf.keras.callbacks.Callback):  # Not Working
    def __init__(self):
        self.gradients = []

    def on_train_batch_end(self, batch, logs=None):
        with tf.GradientTape() as tape:
            loss_tensor = tf.convert_to_tensor(logs['loss'])
            gradients = tape.gradient(loss_tensor, self.model.trainable_weights)
            self.gradients.append([tf.reduce_mean(g).numpy() for g in gradients])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:  # Plot gradients every 5 epochs
            self.plot_gradients()

    def plot_gradients(self):
        plt.figure(figsize=(10, 5))
        for i, grad in enumerate(self.gradients):
            plt.plot(grad, label=f'Layer {i+1}')
        plt.xlabel('Batch')
        plt.ylabel('Gradient')
        plt.title('Gradient Magnitude per Layer')
        plt.legend()
        plt.show()
