"""mura dataset."""

import tensorflow_datasets as tfds
import os

_DESCRIPTION = """
"A dataset of musculoskeletal radiographs consisting of 14,863 studies from 12,173 patients, with
a total of 40,561 multi-view radiographic images. Each belongs to one of seven standard upper
extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist.
Table 1 summarizes the distribution of normal and abnormal studies.
Each study was manually labeled as normal (negative, 0) or abnormal (positive, 1) by board-certified radiologists from the
Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment
between 2001 and 2012. The labeling was performed during interpretation on DICOM
images presented on at least 3 megapixel PACS medical grade display with max luminance 400
cd/m2 and min luminance 1 cd/m2 with pixel size of 0.2 and native resolution of 1500 x 2000
pixels. The clinical images vary in resolution and in aspect ratios. We split the dataset into training
(11,184 patients, 13,457 studies, 36,808 images), validation (783 patients, 1,199 studies, 3,197 images), 
and test (206 patients, 207 studies, 556 images) sets. There is no overlap in patients between
any of the sets. (See Mura Homepage)"
"""

_CITATION = """
@article{rajpurkar2017mura,
  title={Mura: Large dataset for abnormality detection in musculoskeletal radiographs},
  author={Rajpurkar, Pranav and Irvin, Jeremy and Bagul, Aarti and Ding, Daisy and Duan, Tony and Mehta, Hershel and Yang, Brandon and Zhu, Kaylie and Laird, Dillon and Ball, Robyn L and others},
  journal={arXiv preprint arXiv:1712.06957},
  year={2017}
}
"""

_Classes = [
    "negative",
    "positive"]

_DIR = os.path.dirname(__file__)


class Mura(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for mura dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'name': tfds.features.Text(),  # patient id
                'image': tfds.features.Image(),
                'image_num': tfds.features.Text(),  # image number of a patient
                'label': tfds.features.ClassLabel(names=['normal', 'abnormal']),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://stanfordmlgroup.github.io/competitions/mura/',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_kaggle_data(competition_or_dataset='cjinny/mura-v11')

        return {
            'train': self._generate_examples(os.path.join(path, 'MURA-v1.1/train')),
            'test': self._generate_examples(os.path.join(path, 'MURA-v1.1/valid'))
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # body_parts = ["XR_WRIST"]
        body_parts = ["XR_HAND", "XR_FINGER", "XR_FOREARM", "XR_SHOULDER"]

        # Read the input data out of the source files
        root = "/".join(path.split("/")[:-2])  # ../.. to get to root dataloader folder
        if "train" in path:
            csv_path = root + "/MURA-v1.1/train_image_paths.csv"
        else:
            csv_path = root + "/MURA-v1.1/valid_image_paths.csv"

        with open(csv_path, 'rb') as F:
            d = F.readlines()
            for row in d:
                img_path = str(row, encoding='utf-8').strip()
                if img_path.split('/')[2] in body_parts:
                    # And yield (key, feature_dict)
                    patient_id = img_path.split("/")[-3].replace("patient", "")
                    yield img_path, {
                        'name': patient_id,  # patient id ## ex. 0692
                        'image': root + "/" + img_path,
                        'image_num': img_path.split("/")[-1].split(".")[0].replace("image", ""),
                        # image count for patient
                        'label': img_path.split('_')[-1].split('/')[0],
                    }
