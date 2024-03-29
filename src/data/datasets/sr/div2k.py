import os

from .image import ImageDataset


class DIV2K(ImageDataset):
    """
    Root directory should look like
        root/
            DIV2K_train_HR/
                0000.png
                0001.png
                ...
            DIV2K_train_LR_bicubic/
                x2/
                    0000x2.png
                    0001x2.png
                    ...
                x3/
                x4/
            DIV2K_valid_HR/
                0801.png
                0802.png
                ...
            DIV2K_valid_LR_bicubic/
                x2/
                x3/
                x4/
    """

    def __init__(
        self,
        root,
        scale,
        transform=None,
        is_binary=True,
        train=True,
    ):
        split = "train" if train else "valid"
        deg = "bicubic"
        hr_data_dir = os.path.join(root, f"DIV2K_{split}_HR")
        lr_data_dir = os.path.join(root, f"DIV2K_{split}_LR_{deg}", f"X{scale}")
        data_dirs = {"img_hr": hr_data_dir, "img_lr": lr_data_dir}
        super().__init__(data_dirs, transform, is_binary, train)

    def __len__(self):
        return super().__len__()
