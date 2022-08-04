import numpy as np
from keras.utils.all_utils import Sequence
import tensorflow as tf
from skimage.io import imread


class Test_img_data_generator(Sequence):
    """
    Take 15 positive self selected examples and try counterfactual generation
    """

    def __init__(self, batch_size, input_size=(512, 512), transform=None):
        self.input_size = input_size
        self.t = transform
        # root = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/"
        root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid/XR_WRIST/'
        """
                                root + "patient11186/study3_positive/image3.png",
                                root + "patient11188/study1_positive/image1.png",
                                root + "patient11188/study1_positive/image2.png",
                                root + "patient11188/study1_positive/image3.png",
                                root + "patient11188/study1_positive/image4.png",
                                root + "patient11190/study1_positive/image1.png",
                                root + "patient11190/study1_positive/image2.png",
                                root + "patient11186/study2_positive/image2.png",
                                root + "patient11192/study1_positive/image1.png",
                                root + "patient11192/study1_positive/image2.png",
                                root + "patient11192/study1_positive/image3.png"
        """
        self.pos_image_paths = [root + "patient11186/study2_positive/image1.png",
                                root + "patient11186/study2_positive/image3.png",
                                root + "patient11186/study3_positive/image1.png",
                                root + "patient11186/study3_positive/image2.png",
                                root + "patient11205/study1_positive/image1.png",
                                root + "patient11205/study1_positive/image2.png",
                                root + "patient11205/study1_positive/image3.png",
                                root + "patient11267/study1_positive/image1.png"]
        self.batch_size = len(self.pos_image_paths)

    def __len__(self):
        return (np.ceil(len(self.pos_image_paths) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        # TODO: ONLY FOR BATCH SIZE OF 1
        batch_pos = self.pos_image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        y = np.array([0, 1])
        batches = [batch_pos]
        pos = []
        ys = []
        for i, batch in enumerate(batches):
            for file in batch:
                img = imread(file)
                if len(img.shape) < 3:
                    img = tf.expand_dims(img, axis=-1)
                if img.shape[-1] != 3:
                    img = tf.image.grayscale_to_rgb(img)
                img = tf.image.resize_with_pad(img, self.input_size[0], self.input_size[1])
                pos.append(img)
                ys.append(y)
        pos = tf.stack(pos)
        ys = np.reshape(ys, (-1, 2))
        return pos, ys
