import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.applications import vgg19


class Transfer:

    def __init__(self, content_path, style_path, max_dim):
        content_img = cv2.imread(content_path)
        content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        self.content_img = cv2.resize(content_img, (max_dim, max_dim))

        style_img = cv2.imread(style_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        self.style_img = cv2.resize(style_img, (max_dim, max_dim))

        self.CONTENT_LAYERS = ["block5_conv2"]
        self.OUTPUT_LAYERS = ["block4_conv1", "block4_conv2", "block4_conv3",
                         "block4_conv4"]

    def make_model(self, include_full=False, input_shape=None):
        if include_full:
            base_model = vgg19.VGG19(include_top=True, weights="imagenet")
            return base_model
        elif input_shape is not None:
            base_model = vgg19.VGG19(include_top=False, input_shape=input_shape,
                                     weights="imagenet")
        else:
            base_model = vgg19.VGG19(include_top=False, weights="imagenet")

        base_model.trainable = False
        content_layers = self.CONTENT_LAYERS
        style_layers = self.OUTPUT_LAYERS
        output_layers = [base_model.get_layer(layer).output for layer in
                         (content_layers + style_layers)]
        return tf.keras.models.Model(base_model.input, output_layers)

    def deprocess(self, processed_img):
        VGG_BIASES = vgg19.preprocess_input((np.zeros((3))).astype("float32"))

        unprocessed_img = processed_img - VGG_BIASES
        unprocessed_img = tf.unstack(unprocessed_img, axis=-1)
        unprocessed_img = tf.stack(
            [unprocessed_img[2], unprocessed_img[1], unprocessed_img[0]],
            axis=-1)
        return unprocessed_img

    def get_content_loss(self, new_image_content, base_image_content):
        return np.mean(np.square(new_image_content - base_image_content))

    def get_gram_matrix(self, output):
        first_style_layer = output
        A = tf.reshape(first_style_layer, (-1, first_style_layer.shape[-1]))
        n = A.shape[0]
        gram_matrix = tf.matmul(A, A, transpose_a=True)
        n = gram_matrix.shape[0]
        return gram_matrix / tf.cast(n, "float32"), n

    def get_style_loss(self,new_image_style, base_style):
        new_style_gram, gram_num_height = self.get_gram_matrix(new_image_style)
        base_style_gram, gram_num_height2 = self.get_gram_matrix(base_style)
        assert gram_num_height == gram_num_height2
        gram_num_features = new_style_gram.shape[0]
        loss = tf.reduce_sum(tf.square(base_style_gram - new_style_gram) / (
                    4 * (gram_num_height ** 2) * (gram_num_features ** 2)))
        return loss

    def get_total_loss(self, new_image_output, base_content_image_output,
                       base_style_image_output, alpha=.999):
        new_image_styles = new_image_output[len(self.CONTENT_LAYERS):]
        base_image_styles = base_style_image_output[len(self.CONTENT_LAYERS):]
        style_loss = 0
        N = len(new_image_styles)
        for i in range(N):
            style_loss += self.get_style_loss(new_image_styles[i],
                                         base_image_styles[i])

        new_image_contents = new_image_output[:len(self.CONTENT_LAYERS)]
        base_image_contents = base_content_image_output[:len(self.CONTENT_LAYERS)]
        content_loss = 0
        N = len(new_image_contents)
        for i in range(N):
            content_loss += self.get_content_loss(new_image_contents[i],
                                             base_image_contents[i]) / N

        return (1 - alpha) * style_loss + alpha * content_loss

    def get_content_img(self):
        return self.content_img

    def get_style_img(self):
        return self.style_img


