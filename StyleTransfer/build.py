from StyleTransfer.transfer_style import Transfer as painter
import numpy as np
from tensorflow.keras.applications import vgg19
import tensorflow as tf
from IPython.display import display,clear_output
import matplotlib.pyplot as plt


class ArtBuilder:

    def __init__(self, content_path, style_path, max_dim):
        self.painter = painter(content_path, style_path, max_dim)

    def build(self, iterations=150):
        plt.subplot(1, 2, 1)
        self.imshow(self.painter.content_img)

        plt.subplot(1, 2, 2)
        self.imshow(self.painter.style_img)
        plt.show()

        base_model = self.painter.make_model()
        style_img = self.painter.get_style_img()
        content_img = self.painter.get_content_img()


        processed_content = vgg19.preprocess_input(np.expand_dims(content_img, axis=0))
        processed_style = vgg19.preprocess_input(np.expand_dims(style_img, axis=0))


        base_style_outputs = base_model(processed_style)
        base_content_output = base_model(processed_content)

        processed_content_var = tf.Variable(
            processed_content + tf.random.normal(
                processed_content.shape))

        optimizer = tf.optimizers.Adam(5, beta_1=.99, epsilon=1e-3)

        images = []
        losses = []
        i = 0
        best_loss = 200000
        VGG_BIASES = vgg19.preprocess_input((np.zeros((3))).astype("float32"))

        min_vals = VGG_BIASES
        max_vals = 255 + VGG_BIASES
        for i in range(iterations):
            print(i)
            with tf.GradientTape() as tape:
                tape.watch(processed_content_var)
                content_var_outputs = base_model(processed_content_var)
                loss = self.painter.get_total_loss(content_var_outputs, base_content_output,
                                      base_style_outputs, alpha=.97)
                grad = tape.gradient(loss, processed_content_var)
                losses.append(loss)
                optimizer.apply_gradients(zip([grad], [processed_content_var]))
                clipped = tf.clip_by_value(processed_content_var, min_vals,
                                           max_vals)
                processed_content_var.assign(clipped)
                if i % 5 == 0:
                    images.append(self.painter.deprocess(processed_content_var))
                if loss < best_loss:
                    best_image = processed_content_var
                    best_loss = loss
                display(loss)
                clear_output(wait=True)

        deprocessed_best_image = self.painter.deprocess(best_image)
        plt.figure(figsize=(5, 5))
        plt.imshow(deprocessed_best_image[0] / 255)
        plt.show()

    def imshow(self, img):
        plt.imshow(img)



