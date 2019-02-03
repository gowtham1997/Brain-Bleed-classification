from radio.batchflow import action
import argparse
from radio import CTImagesBatch
from radio.batchflow import FilesIndex, Dataset
from plot_utils import show_slices
import os
# import numpy as np


class CTImagesCustomBatch(CTImagesBatch):
    """ Ct-scans batch class with your own action """

    @action
    # action-decorator allows you to chain your method with other actions in
    # pipelines
    def center(self):
        """ Center values of pixels in each scan from batch """
        for ix in self.indices:

            images_ix = getattr(self[ix], 'images')
            # remove HU below -10 (fats and other useless things)
            # images_ix[images_ix <= -10] = 0

        return self  # action must always return a batch-object


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display CT volumes in a \
        directory using matplotlib Event handler API')
    parser.add_argument('--dir', type=str,
                        help='directory where CT files are located')

    args = parser.parse_args()
    _dir = args.dir
    ext_prefix = 'no'

    if not os.path.isdir(_dir):
        # if directory does not exist print this message
        print("{} directory does not exist. Please recheck the path".format(
            _dir))
    else:
        contains_ext = False
        if any(File.endswith(".dcm") for File in os.listdir(".")):
            contains_ext = True

    if contains_ext:
        ext_prefix = ''
    print('Opening {} folder with {} dcm extensions'.format(_dir, ext_prefix))
    # set up the index
    dicom_ix = FilesIndex(path=_dir, dirs=True, no_ext=not contains_ext)

    dicom_dataset = Dataset(index=dicom_ix, batch_class=CTImagesCustomBatch)

    pipeline = (
        dicom_dataset.p
        .load(fmt='dicom')
        .center()
        .unify_spacing(spacing=(1.0, 1.0, 1.0), shape=(128, 299, 299))
    )
    batch = (dicom_dataset >> pipeline).next_batch(batch_size=1)

    show_slices(batch, scan_indices=0, grid=False)
