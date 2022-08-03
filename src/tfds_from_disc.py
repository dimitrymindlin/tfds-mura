from typing import List

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tf2lib as tl
import tensorflow_addons as tfa


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1,
                 labels=None, special_normalisation=None):
    if training:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            img = tfa.image.equalize(img)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize_with_pad(img, load_size, load_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            if not special_normalisation:
                img = img / 255.0 * 2 - 1
            else:
                img = special_normalisation(img)
            if label is not None:
                return img, label
            return img
    else:
        @tf.function
        def _map_fn(img, label=None):  # preprocessing
            img = tf.cast(img, tf.float32)
            # img = tfa.image.equalize(img)
            img = tf.image.resize_with_pad(img, crop_size, crop_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if not special_normalisation:
                img = img / 255.0 * 2 - 1
            else:
                img = special_normalisation(img)
            if label is not None:
                return img, label
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat,
                                       labels=labels)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                             shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size
    return A_B_dataset, len_dataset


def make_concat_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True,
                        repeat=False, special_normalisation=None):
    dataset_length = len(A_img_paths) + len(B_img_paths)
    class_labels = [(1, 0) for _ in range(len(A_img_paths))]
    class_labels.extend([(0, 1) for _ in range(len(B_img_paths))])
    A_img_paths.extend(B_img_paths)  # becoming all_image_paths
    all_image_paths = A_img_paths
    return make_dataset(all_image_paths, batch_size, load_size, crop_size, training, drop_remainder=True,
                        shuffle=shuffle, repeat=repeat, labels=class_labels,
                        special_normalisation=special_normalisation), dataset_length


def get_mura_data_paths(body_parts: List[str], tfds_path: str, test_size=0.2):
    """
    body_parts: List of body parts to work with. Check MURA documentation for available body_parts.
    tfds_path: Path to tensorflow datasets directory.
    """

    # To get the filenames for a task
    def to_categorical(x, y):
        y = [0 if x == 'negative' else 1 for x in y]
        y = tf.keras.utils.to_categorical(y)
        x, y = shuffle(x, y)
        return x, y

    def filenames(parts, train=True):
        data_root = tfds_path + '/downloads/cjinny_mura-v11/'
        if train:
            csv_path = data_root + "/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = data_root + "/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            if parts == 'all':
                imgs = [data_root + str(x, encoding='utf-8').strip() for x in d]
            else:
                imgs = [data_root + str(x, encoding='utf-8').strip().replace("MURA-v1.1", "MURA-v1.1_transformed") for x
                        in d
                        if
                        str(x, encoding='utf-8').strip().split('/')[2] in body_parts]

        # imgs= [x.replace("/", "\\") for x in imgs]
        labels = [x.split('_')[-1].split('/')[0] for x in imgs]
        return imgs, labels

    train_x, train_y = filenames(parts=body_parts)  # train data
    test_x, test_y = filenames(parts=body_parts, train=False)  # test data
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y,
                                                          test_size=test_size)  # split train and valid data

    train_x, train_y = to_categorical(train_x, train_y)
    valid_x, valid_y = to_categorical(valid_x, valid_y)
    test_x, test_y = to_categorical(test_x, test_y)

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_split_dataset_paths(body_parts: List[str], tfds_path: str):
    # A = 0 = negative, B = 1 = positive
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_mura_data_paths(body_parts, tfds_path)
    A_img_paths = [filename for filename in train_x if "negative" in filename]
    B_img_paths = [filename for filename in train_x if "positive" in filename]
    A_img_paths_test = [filename for filename in test_x if "negative" in filename]
    B_img_paths_test = [filename for filename in test_x if "positive" in filename]
    return A_img_paths, B_img_paths, A_img_paths_test, B_img_paths_test
