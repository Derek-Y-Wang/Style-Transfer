import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.applications import vgg19

# Style image location
wave_loc = "./imgs/starry_night_full.jpg"
wave_img = cv2.imread(wave_loc)
wave_img = cv2.cvtColor(wave_img, cv2.COLOR_BGR2RGB)
wave_img = cv2.resize(wave_img, (512, 512))

# plt.imshow(wave_img)
# plt.show()


base_model = vgg19.VGG19(include_top=False, weights="imagenet")
base_model.summary()

golden_gate = cv2.imread("./imgs/golden_gate.jpg", cv2.COLOR_BGR2RGB)
golden_gate = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
golden_gate = cv2.resize(golden_gate, (512, 512))
# plt.imshow(golden_gate)
# plt.show()

style_img = wave_img
content_img = golden_gate

processed_wave = vgg19.preprocess_input(np.expand_dims(style_img, axis=0))

processed_gate = vgg19.preprocess_input(np.expand_dims(content_img, axis=0))
# plt.imshow(processed_wave[0])
# plt.show()

# We need to unprocess the image
VGG_BIASES = vgg19.preprocess_input((np.zeros((3))).astype('float32'))
# print(VGG_BIASES)


def deprocess(processed_img):
    deprocessed_img = processed_img - VGG_BIASES
    deprocessed_img = tf.unstack(deprocessed_img, axis=-1)
    deprocessed_img = tf.stack(
        [deprocessed_img[2], deprocessed_img[1], deprocessed_img[0]], axis=-1)
    return deprocessed_img


# plt.figure(figsize=(15, 15))
# plt.imshow(deprocess(processed_gate)[0].numpy().astype("int32"))
# plt.show()

CONTENT_LAYERS = ["block5_conv2"]
STYLE_LAYERS = ["block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4"]


def make_model():
    base_model = vgg19.VGG19(include_top=False, weights="imagenet")
    base_model.translate = False
    content_layers = CONTENT_LAYERS
    style_layers = STYLE_LAYERS
    output_layers = [base_model.get_layer(layer).output for layer in (content_layers+style_layers)]
    return tf.keras.models.Model(base_model.input, output_layers)


base_model = make_model()
golden_gate_output = base_model(processed_gate)
wave_output = base_model(processed_wave)

golden_gate_content = golden_gate_output[0]


def get_content_loss(new_image_content, base_image_content):
    # print(len(new_image_content))
    return np.mean(np.square(new_image_content-base_image_content))


def get_gram_matrix(output):
    first_style_layer = output
    style_layer = tf.reshape(first_style_layer,(-1, first_style_layer.shape[-1]))
    gram_matrix = tf.matmul(style_layer, style_layer, transpose_a=True)
    n = style_layer.shape[0]
    return gram_matrix/tf.cast(n, "float32"), n


# gram_matrix, N = get_gram_matrix(wave_output[2].numpy())
# plt.figure(figsize=(10, 10))
# plt.imshow(gram_matrix.numpy())
# plt.show()

def get_style_loss(new_image_style, base_style):
    new_style_gram, gram_num_height = get_gram_matrix(new_image_style)
    base_style_gram, gram_num_height2 = get_gram_matrix(base_style)
    assert gram_num_height == gram_num_height2
    gram_num_features = gram_num_height
    loss = tf.reduce_sum(tf.square(base_style_gram-new_style_gram)/(4*(gram_num_height**2)*(gram_num_features**2)))
    return loss


def get_total_loss(new_image_output, base_content_image_output,
                   base_style_image_output, alpha=.999):
    new_image_styles = new_image_output[len(CONTENT_LAYERS):]
    base_image_styles = base_style_image_output[len(CONTENT_LAYERS):]
    style_loss = 0
    N = len(new_image_styles)
    for i in range(N):
        style_loss += get_style_loss(new_image_styles[i], base_image_styles[i])

    new_image_contents = new_image_output[:len(CONTENT_LAYERS)]
    base_image_contents = base_content_image_output[:len(CONTENT_LAYERS)]
    content_loss = 0
    N = len(new_image_contents)
    for i in range(N):
        content_loss += get_content_loss(new_image_contents[i],
                                         base_image_contents[i]) / N

    return (1 - alpha) * style_loss + alpha * content_loss


print(get_total_loss(wave_output, golden_gate_output, wave_output))



base_style_outputs = base_model(processed_wave)
base_content_output = base_model(processed_gate)
# print(tf.random.normal(processed_gate.shape))
processed_content_var = tf.Variable(processed_gate+tf.random.normal(processed_gate.shape)) #*tf.random.normal(processed_gate.shape)

optimizer = tf.optimizers.Adam(5,beta_1=.99,epsilon=1e-3)

from IPython.display import display, clear_output
images = []
losses = []

i=0
best_loss = 200000
min_vals = VGG_BIASES
max_vals = 255 + VGG_BIASES

for i in range(100):
    with tf.GradientTape() as tape:
        tape.watch(processed_content_var)
        content_var_outputs = base_model(processed_content_var)
        loss = get_total_loss(content_var_outputs, base_content_output,
                              base_style_outputs, alpha=.97)
        grad = tape.gradient(loss, processed_content_var)
        losses.append(loss)
        optimizer.apply_gradients(zip([grad], [processed_content_var]))
        clipped = tf.clip_by_value(processed_content_var, min_vals, max_vals)
        processed_content_var.assign(clipped)
        if i % 5 == 0:
            images.append(deprocess(processed_content_var))
        if loss < best_loss:
            best_image = processed_content_var
            best_loss = loss
        display(loss)
        clear_output(wait=True)

deprocessed_best_image = deprocess(best_image)
plt.figure(figsize=(10,10))
plt.imshow(deprocessed_best_image[0]/255)
plt.show()



