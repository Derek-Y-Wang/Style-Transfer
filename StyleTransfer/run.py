import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.applications import vgg19
from StyleTransfer.transfer_style import Transfer as painter
from StyleTransfer.build import ArtBuilder


if __name__ == "__main__":
    art = ArtBuilder("./imgs/e30.jpg", "./imgs/starry_night_full.jpg", 256)
    art.build()
