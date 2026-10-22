import cv2

from dataset.shapenetsem_normalized import shapenetsem_normalized
from utils.legacy_depth_correction import correct_legacy_depth


class shapenetsem_normalized_corrected(shapenetsem_normalized):
    dataset_name = 'Shapenet Sem (Normalized Corrected)'
    depth_interpolation = cv2.INTER_NEAREST

    def _read_depth(self, path):
        return correct_legacy_depth(super()._read_depth(path))