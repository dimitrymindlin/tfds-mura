"""cxr14 dataset."""

import tensorflow_datasets as tfds
import csv
import os

_DESCRIPTION = """
"ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,805 unique patients with 
the text-mined fourteen disease image labels (where each image can have multi-labels), mined 
from the associated radiological reports using natural language processing. Fourteen common 
thoracic pathologies include Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, 
Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_thickening, Cardiomegaly, Nodule, Mass and 
Hernia, which is an extension of the 8 common disease patterns listed in our CVPR2017 paper. 
Note that original radiology reports (associated with these chest x-ray studies) are not 
meant to be publicly shared for many reasons. The text-mined disease labels are expected to 
have accuracy >90%."
"""

_CITATION = """
@article{DBLP:journals/corr/WangPLLBS17,
  author    = {Xiaosong Wang and
               Yifan Peng and
               Le Lu and
               Zhiyong Lu and
               Mohammadhadi Bagheri and
               Ronald M. Summers},
  title     = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on
               Weakly-Supervised Classification and Localization of Common Thorax
               Diseases},
  journal   = {CoRR},
  volume    = {abs/1705.02315},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.02315},
  eprinttype = {arXiv},
  eprint    = {1705.02315},
  timestamp = {Thu, 03 Oct 2019 13:13:22 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/WangPLLBS17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_Classes = [
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Pleural_Thickening",
  "Hernia"]

_DIR = os.path.dirname(__file__)

class CXR14(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cxr14 dataset."""

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
            'name': tfds.features.Text(), #patient id
            'image': tfds.features.Image(),
            'label': tfds.features.Sequence(
              tfds.features.ClassLabel(names=['0','1']),
              length=len(_Classes)),
        }),
        supervised_keys=('image', 'label'),
        homepage='https://nihcc.app.box.com/v/ChestXray-NIHCC',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    data_part_links = [
      'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
      'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
      'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
      'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
      'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
      'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
      'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
      'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
      'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
      'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
      'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
      'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz']

    image_paths = dl_manager.download_and_extract(data_part_links) #TODO how to merge dataset parts

    csv_train_path = os.path.join(_DIR, 'default_split/train.csv')
    csv_test_path = os.path.join(_DIR, 'default_split/test.csv')

    return {
        'train': self._generate_examples(image_paths, csv_train_path),
        'test': self._generate_examples(image_paths, csv_test_path),
    }

  def _generate_examples(self, image_paths, csv_path):
    """Yields examples."""
    assert len(image_paths) > 0, 'could not download and extract all dataset paths'

    with open(csv_path, newline='') as csv_file:
      cur_part_dir = image_paths[0] #TODO how to merge dataset parts
      csv_reader = csv.reader(csv_file, delimiter=',')
      next(csv_reader)
      for row in csv_reader:
        image_path = cur_part_dir / 'images' / str(row[0])

        #check in which dataset part the image exists
        image_exists = True
        if not image_path.exists(): #TODO possible performance leak
          image_exists = False
          for new_path in image_paths:
            image_path = new_path / 'images' / str(row[0])
            #print('check path' + str(image_path))
            if image_path.exists():
              cur_part_dir = new_path
              image_exists = True
              break

        #skip missing images
        if image_exists:
          image_id = int(((image_path.stem).replace('_',''))) #00000901_005.png to 901005
          yield image_id, {
              'name': row[1],
              'image': image_path,
              'label': row[3:],
            }
