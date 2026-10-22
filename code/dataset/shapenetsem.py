from pathlib import Path
import numpy as np
from collections import defaultdict
import cv2

from dataset.base_dataset import BaseDataset
from dataset.shapenetsem_index import read_records
from utils.exr import read_exr_red


class shapenetsem(BaseDataset):
    dataset_name = 'Shapenet Sem'
    depth_interpolation = cv2.INTER_LINEAR

    def __init__(self, data_path, train_txt_path=None, test_txt_path=None,
                 is_train=True, crop_size=(448, 576), scale_size=None):
        super().__init__(crop_size)
        self.is_train = is_train
        self.scale_size = scale_size
        self.data_path = Path(data_path)
        self.records = read_records(train_txt_path if is_train else test_txt_path)

        if self.is_train:
            grouped = defaultdict(list)
            for record in self.records:
                grouped[Path(record[1]).parent.name].append(record)
            self.objects = list(grouped.values())

        phase = 'train' if is_train else 'test'
        print(f'Dataset: {self.dataset_name}')
        print(f'# of {phase} images: {len(self.records)}')

    def __len__(self):
        if self.is_train:
            return len(self.objects) * 3
        return len(self.records)

    def _get_record(self, idx):
        if self.is_train:
            views = self.objects[idx % len(self.objects)]
            return views[np.random.randint(len(views))]
        return self.records[idx]

    def _read_depth(self, path):
        return read_exr_red(path)

    def __getitem__(self, idx):
        rgb_path, depth_path = self._get_record(idx)
        object_id = Path(depth_path).parent.name

        image = cv2.imread(str(self.data_path / rgb_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = self._read_depth(self.data_path / depth_path)

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
            'filename': object_id + '_' + Path(rgb_path).name,
        }
