from pathlib import Path
import numpy as np
import cv2

from dataset.shapenetsem import shapenetsem
from dataset.shapenetsem_index import load_normalizations


class shapenetsem_normalized(shapenetsem):
    dataset_name = 'Shapenet Sem (Normalized)'

    def __init__(self, data_path, train_txt_path=None, test_txt_path=None,
                 metadata_path=None, is_train=True, crop_size=(448, 576),
                 scale_size=None):
        super().__init__(data_path, train_txt_path, test_txt_path,
                         is_train, crop_size, scale_size)
        self.normalizations = load_normalizations(metadata_path)

    def __getitem__(self, idx):
        rgb_path, depth_path = self._get_record(idx)
        object_id = Path(depth_path).parent.name

        image = cv2.imread(str(self.data_path / rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = self._read_depth(self.data_path / depth_path)
        normalization = self.normalizations[object_id]

        if self.scale_size:
            image = cv2.resize(image, self.scale_size)
            depth = cv2.resize(depth, self.scale_size, interpolation=self.depth_interpolation)

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        return {
            'image': image,
            'depth': depth,
            'depth_normalized': depth / normalization,
            'normalization': np.float32(normalization),
            'filename': object_id + '_' + Path(rgb_path).name,
        }
