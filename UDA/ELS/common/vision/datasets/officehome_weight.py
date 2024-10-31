import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class OfficeHome_weight(ImageList):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
        "super_two_stage_CltoPr1":"image_list/super_two_stage_CltoPr1.txt",
        "nosuper_two_stage_CltoPr1": "image_list/nosuper_two_stage_CltoPr1.txt",
        "super_all_CltoPr1": "image_list/super_all_CltoPr1.txt",
        "super_all_noise_CltoPr1": "image_list/super_all_noise_CltoPr1.txt",
        "super_two_stage_ArtoCl1": "image_list/super_two_stage_ArtoCl1.txt",
        "super_all_noise_ArtoPr1": "image_list/super_all_noise_ArtoPr1.txt",
        "super_all_noise_ArtoCl_rank32": "image_list/super_all_noise_ArtoCl_rank32.txt",
        "super_all_noise_ArtoCl_rank64": "image_list/super_all_noise_ArtoCl_rank64.txt",
        "super_all_noise_CltoAr_rank32": "image_list/super_all_noise_CltoAr_rank32.txt",
        "super_all_noise_ArtoRw_rank32": "image_list/super_all_noise_ArtoRw_rank32.txt",
        "super_all_noise_CltoRw_rank32": "image_list/super_all_noise_CltoRw_rank32.txt",
        "super_all_noise_PrtoAr_rank32": "image_list/super_all_noise_PrtoAr_rank32.txt",
        "super_all_noise_PrtoCl_rank32": "image_list/super_all_noise_PrtoCl_rank32.txt",
        "super_all_noise_ArtoCl_rank64_5000": "image_list/super_all_noise_ArtoCl_rank64_5000.txt",
        "super_all_noise_raw": "image_list/super_all_noise_raw.txt",
        "super_all_noise_PrtoRw_rank32": "image_list/super_all_noise_PrtoRw_rank32.txt",
        "super_all_noise_RwtoAr_rank32": "image_list/super_all_noise_RwtoAr_rank32.txt",
        "super_all_noise_RwtoCl_rank32": "image_list/super_all_noise_RwtoCl_rank32.txt",
        "super_all_noise_RwtoPr_rank32": "image_list/super_all_noise_RwtoPr_rank32.txt",
        "super_all_noise_ArtoCl_flow": "image_list/super_all_noise_ArtoCl_flow.txt",
        "super_all_noise_ArtoCl_flow_random": "image_list/super_all_noise_ArtoCl_flow_random.txt",
        "CltoPr1": "image_list/CltoPr1.txt",
        "super_all_noise_ArtoPr_flow_random": "image_list/super_all_noise_ArtoPr_flow_random.txt",
        "super_all_noise_ArtoRw_flow_random": "image_list/super_all_noise_ArtoRw_flow_random.txt",
        "super_all_noise_CltoAr_flow_random": "image_list/super_all_noise_CltoAr_flow_random.txt",
        "super_all_noise_CltoPr_flow_random": "image_list/super_all_noise_CltoPr_flow_random.txt",
        "super_all_noise_PrtoAr_flow_random": "image_list/super_all_noise_PrtoAr_flow_random.txt",
        "super_all_noise_PrtoCl_flow_random": "image_list/super_all_noise_PrtoCl_flow_random.txt",
        "super_all_noise_CltoPr_flow_select": "image_list/super_all_noise_CltoPr_flow_select.txt",
        "super_all_noise_raw_SD21": "image_list/super_all_noise_raw_SD21.txt",
        "super_all_noise_RwtoAr_200": "image_list/super_all_noise_RwtoAr_200.txt",
        "super_all_noise_ArtoPr_rank64": "image_list/super_all_noise_ArtoPr_rank64.txt",
        "PrtoCl_oneperclass_2500_rank8": "image_list/PrtoCl_oneperclass_2500_rank8.txt",
        "PrtoCl_labels_rank32": "image_list/PrtoCl_labels_rank32.txt",
        "PrtoCl_oneperclass_labels_rank8": "image_list/PrtoCl_oneperclass_labels_rank8.txt",
        "PrtoCl_oneperclass_labels_flow_rank8": "image_list/PrtoCl_oneperclass_labels_flow_rank8.txt",
        "PrtoCl_labels_flow_new": "image_list/PrtoCl_labels_flow_new.txt",
        "labels_PrtoCl2": "image_list/labels_PrtoCl2.txt",
        "labels_PrtoCl3": "image_list/labels_PrtoCl3.txt",
        "super_Cl": "image_list/super_Cl.txt",
        "super_Pr": "image_list/super_Pr.txt",
        "super_pseudo_PrtoCl09": "image_list/super_pseudo_PrtoCl09.txt",
        "super_pseudo_PrtoCl095": "image_list/super_pseudo_PrtoCl095.txt",
        "PrtoCl0dot3": "image_list/PrtoCl0.3.txt",
        "PrtoCl0dot7": "image_list/PrtoCl0.7.txt",
        "VAE_PrtoCl1": "image_list/VAE_PrtoCl1.txt",
        "diffusers_PrtoCl_all": "image_list/diffusers_PrtoCl_all.txt",
        "diffusers_PrtoCl_10": "image_list/diffusers_PrtoCl_10.txt",
        "super_all_noise_PrtoCl_rank32_100": "image_list/super_all_noise_PrtoCl_rank32_100.txt",
        "super_all_noise_PrtoCl_rank32_200": "image_list/super_all_noise_PrtoCl_rank32_200.txt",
        "two_stage_PrtoCl_rank32": "image_list/two_stage_PrtoCl_rank32.txt",
        "two_stage_PrtoCl_rank32_0dot5": "image_list/two_stage_PrtoCl_rank32_0dot5.txt",
        "two_stage_PrtoCl_rank32_flow": "image_list/two_stage_PrtoCl_rank32_flow.txt",
        "A_C": "ArtoCl_256_20k.txt",
        "A_P": "ArtoPr_256_20k.txt",
        "A_R": "ArtoRw_256_20k.txt",
        "C_A": "CltoAr_256_20k.txt",
        "C_R": "CltoRw_256_20k.txt",
        "C_P": "CltoPr_256_20k.txt",
        "P_A": "PrtoAr_256_20k.txt",
        "P_C": "PrtoCl_256_20k.txt",
        "P_R": "PrtoRw_256_20k.txt",
        "R_A": "RwtoAr_256_20k.txt",
        "R_C": "RwtoCl_256_20k.txt",
        "R_P": "RwtoPr_256_20k.txt",
        "A_C_p": "Ar2Cl_p.txt",
        "A_P_p": "Ar2Pr_p.txt",
        "A_R_p": "Ar2Rw_p.txt",
        "C_A_p": "Cl2Ar_p.txt",
        "C_P_p": "Cl2Pr_p.txt",
        "C_R_p": "Cl2Rw_p.txt",
        "P_A_p": "Pr2Ar_p.txt",
        "P_C_p": "Pr2Cl_p.txt",
        "P_R_p": "Pl2Rw_p.txt",
        "R_A_p": "Rw2Ar_p.txt",
        "R_C_p": "Rw2Cl_p.txt",
        "R_P_p": "Rw2Pr_p.txt",
        "A_C_50": "ArtoCl_400_50.txt",
        "A_P_50": "ArtoPr_400_50.txt",
        "A_R_50": "ArtoRw_400_50.txt",
        "C_A_50": "CltoAr_400_50.txt",
        "C_R_50": "CltoRw_400_50.txt",
        "C_P_50": "CltoPr_400_50.txt",
        "P_A_50": "PrtoAr_400_50.txt",
        "P_C_50": "PrtoCl_400_50.txt",
        "P_R_50": "PrtoRw_400_50.txt",
        "R_A_50": "RwtoAr_400_50.txt",
        "R_C_50": "RwtoCl_400_50.txt",
        "R_P_50": "RwtoPr_400_50.txt",
        "A_C_100": "ArtoCl_400_100.txt",
        "A_P_100": "ArtoPr_400_100.txt",
        "A_R_100": "ArtoRw_400_100.txt",
        "C_A_100": "CltoAr_400_100.txt",
        "C_R_100": "CltoRw_400_100.txt",
        "C_P_100": "CltoPr_400_100.txt",
        "P_A_100": "PrtoAr_400_100.txt",
        "P_C_100": "PrtoCl_400_100.txt",
        "P_R_100": "PrtoRw_400_100.txt",
        "R_A_100": "RwtoAr_400_100.txt",
        "R_C_100": "RwtoCl_400_100.txt",
        "R_P_100": "RwtoPr_400_100.txt",
        "A_C_300": "ArtoCl_400_300.txt",
        "A_P_300": "ArtoPr_400_300.txt",
        "A_R_300": "ArtoRw_400_300.txt",
        "C_A_300": "CltoAr_400_300.txt",
        "C_R_300": "CltoRw_400_300.txt",
        "C_P_300": "CltoPr_400_300.txt",
        "P_A_300": "PrtoAr_400_300.txt",
        "P_C_300": "PrtoCl_400_300.txt",
        "P_R_300": "PrtoRw_400_300.txt",
        "R_A_300": "RwtoAr_400_300.txt",
        "R_C_300": "RwtoCl_400_300.txt",
        "R_P_300": "RwtoPr_400_300.txt",
        "A_C_400": "ArtoCl_400.txt",
        "A_P_400": "ArtoPr_400.txt",
        "A_R_400": "ArtoRw_400.txt",
        "C_A_400": "CltoAr_400.txt",
        "C_R_400": "CltoRw_400.txt",
        "C_P_400": "CltoPr_400.txt",
        "P_A_400": "PrtoAr_400.txt",
        "P_C_400": "PrtoCl_400.txt",
        "P_R_400": "PrtoRw_400.txt",
        "R_A_400": "RwtoAr_400.txt",
        "R_C_400": "RwtoCl_400.txt",
        "R_P_400": "RwtoPr_400.txt",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    # CLASSES = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
    #            'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
    #            'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
    #            'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse',
    #            'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin',
    #            'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda',
    #            'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

    def __init__(self, root: str, task: str, download: Optional[bool] = False, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        # if download:
        #     list(map(lambda args: download_data(root, *args), self.download_list))
        # else:
        #     list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(OfficeHome_weight, self).__init__(root, OfficeHome_weight.CLASSES, data_list_file=data_list_file, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # 从文件名中提取权重信息
        weight = self.extract_weight_from_filename(path)

        # 返回图像、标签和权重
        return sample, target, weight
    
    def extract_weight_from_filename(self, filename):
        # 根据新的文件名格式提取权重
        basename = os.path.basename(filename)
        weight = 1.0  # 默认权重
        if 'time_t_' in basename:
            try:
                start = basename.index('time_t_') + 7  # 跳过 "time_t_"
                end = basename.index('_', start)  # 找到下一个下划线，即权重值的结束
                weight = float(basename[start:end])
            except ValueError:
                pass  # 如果转换失败，保持默认权重
            except IndexError:
                # 如果找不到下一个下划线，尝试直接到 ".png" 之前
                try:
                    end = basename.index('.png', start)
                    weight = float(basename[start:end])
                except ValueError:
                    pass
        return weight

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
