"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from .imagelistratio import ImageListRatio
from ._util import download as download_data, check_exits


class Office31ratio(ImageListRatio):
    """Office31 Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/d9bca681c71249f19da2/?dl=1"),
        ("amazon", "amazon.tgz", "https://cloud.tsinghua.edu.cn/f/edc8d1bba1c740dc821c/?dl=1"),
        ("dslr", "dslr.tgz", "https://cloud.tsinghua.edu.cn/f/ca6df562b7e64850ad7f/?dl=1"),
        ("webcam", "webcam.tgz", "https://cloud.tsinghua.edu.cn/f/82b24ed2e08f4a3c8888/?dl=1"),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt",
        "A2D": "image_list/AtoD.txt",
        "A2W": "image_list/AtoW.txt",
        "D2A": "image_list/DtoA.txt",
        "D2W": "image_list/DtoW.txt",
        "W2A": "image_list/WtoA.txt",
        "W2D": "image_list/WtoD.txt",
        "A2W_t": "image_list/AtoW_t.txt",
        "W2D_t": "image_list/WtoD_t.txt",
        "D2W_t": "image_list/DtoW_t.txt",
        "D2A_t": "image_list/DtoA_t.txt",
        "W2A_t": "image_list/WtoA_t.txt",
        "A2D_t": "image_list/AtoD_t.txt",
    }
    CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp',
               'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray',
               'mobile_phone', 'monitor', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector',
               'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

    def __init__(self, root: str, task: str, baseDatasetLength, ratio , download: Optional[bool] = True,
                                              **kwargs):
        assert task in self.image_list
        dataset_length = ratio*baseDatasetLength

        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(Office31ratio, self).__init__(root, Office31ratio.CLASSES, data_list_file=data_list_file, dataset_length=dataset_length, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())