import json
import os
import os
import copy
import torch
import pickle
import numpy as np
import json
import random
import cv2
import sys
import tqdm
from end2end_system.strcut_recover.libs.utils.vocab import TypeVocab, RelationVocab
from torchvision.transforms import functional as F

from PIL import Image
from torch.utils.data import Dataset


def get_file_names_without_extension(directory):
    """
    获取当前目录下所有文件名（不包含后缀）
    @param directory:
    @return:
    """
    file_names = []
    # 获取目录下的所有文件
    files = os.listdir(directory)
    # 遍历文件列表
    for file in files:
        # 构建文件的完整路径
        file_path = os.path.join(directory, file)
        # 检查路径是否为文件而非文件夹
        if os.path.isfile(file_path):
            # 获取文件名（不包含后缀）
            file_name_without_extension, _ = os.path.splitext(file)
            file_names.append(file_name_without_extension)
    return file_names


def get_all_files_in_directory(directory):
    """
    获取目录下所有文件名
    @return:
    """
    # 获取当前目录下的所有文件和文件夹
    files_and_folders = os.listdir(directory)
    # 过滤出当前目录下的所有文件
    files = [file for file in files_and_folders if os.path.isfile(os.path.join(directory, file))]
    return files


def list_subdirectories(folder_path):
    """
    统计指定文件夹下所有子目录的名称，并返回一个包含子目录名称的列表。
    参数：
    - folder_path: 字符串，表示文件夹路径。
    返回：
    - subdirectories_list: 包含子目录名称的列表。
    """
    subdirectories_list = []
    try:
        # 获取文件夹下所有文件和子文件夹
        files_and_folders = os.listdir(folder_path)
        # 迭代处理每个文件或文件夹
        for item in files_and_folders:
            # 拼接文件或文件夹的完整路径
            item_path = os.path.join(folder_path, item)
            # 如果是子目录，则将目录名称添加到列表
            if os.path.isdir(item_path):
                subdirectories_list.append(item)
    except Exception as e:
        print(f"Error: {e}")
    return subdirectories_list


class CustomDataset(Dataset):
    def __init__(self, root_dir, split='train', ly_vocab=TypeVocab(), re_vocab=RelationVocab()):
        self.root_dir = root_dir
        self.split = split  # 'train' or 'test' 默认train
        self.json_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(root_dir, 'images')
        self.info = []
        self.init()  # 先初始化

    def __len__(self):
        return len(self.info)

    def init(self):
        """
        初始化
        @return:
        [
        {
                'lines': lines,
                'page_num': len(lines),
                'imgs_path': imgs_path
            }
        ]
        """
        print('init')
        json_files = get_file_names_without_extension(self.json_dir)
        for json_file in json_files:
            json_file_path = os.path.join(self.json_dir, json_file + '.json')
            image_directory = os.path.join(self.images_dir, json_file)
            with open(json_file_path, 'r') as jf:
                jd = json.load(jf)
            images_name = get_all_files_in_directory(image_directory)
            # print(images_name)
            lines = list()
            pages = set()
            for cl in jd:
                pages.add(cl['page'])
            for page_id in sorted(list(pages)):
                # 把每一页的data用[]封装，最终形成[[page0.data],[page1.data],...]格式
                lines.append([x for x in jd if x['page'] == page_id])
            imgs_path = []
            for item in images_name:
                img_path = os.path.join(image_directory, item)
                imgs_path.append(img_path)
            # print(imgs_path)
            temp_data = {
                'lines': lines,
                'page_num': len(lines),
                'imgs_path': imgs_path,
                'pdf_path': json_file,
                'targets': target
            }
            self.info.append(temp_data)

    def __getitem__(self, idx):
        data = self.info[idx]
        img_lst = []
        for img_path in data['imgs_path']:
            img = cv2.imread(img_path)
            img_lst.append(img)
        data['imgs'] = img_lst
        encoder_input, texts, bboxes = self.cal_items(data)
        if texts == []:
            print('texts is [] when idx =', idx, data['pdf_path'])
            return self[random.randint(0, len(self) - 1)]
        return dict(
            idx=idx,
            bboxes=bboxes,
            transcripts=texts,
            encoder_input=encoder_input,
            lines=data['lines'],
            pdf_path=data['pdf_path']
        )

    def cal_items(self, data):
        texts, bboxes = [], []

        for page_id, lines_pg in enumerate(data['lines']):
            texts_pg = []
            bboxes_pg = []
            for line_idx, line in enumerate(lines_pg):
                bboxes_pg.append(line['box'])
                texts_pg.append(line['text'])
            texts.append(texts_pg)
            bboxes.append(bboxes_pg)

        imgs = data['imgs']

        encoder_input = list()
        for image in imgs:
            image = F.to_tensor(image)
            image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False)
            encoder_input.append(image)

        return encoder_input, texts, bboxes


def train_collate_func(batch_data):
    batch_size = len(batch_data)
    input_channels = batch_data[0]['encoder_input'][0].shape[0]

    max_H = max([max([page.shape[1] for page in data['encoder_input']]) for data in batch_data])
    max_W = max([max([page.shape[2] for page in data['encoder_input']]) for data in batch_data])
    max_page = max([len(data['encoder_input']) for data in batch_data])

    batch_encoder_input = []
    batch_encoder_input_mask = []
    batch_image_size = []

    batch_transcripts = []
    batch_bboxes = []
    batch_lines = []
    pdf_paths = []

    for batch_idx, data in enumerate(batch_data):
        pdf_paths.append(data['pdf_path'])
        encoder_input = torch.zeros(len(data['encoder_input']), input_channels, max_H, max_W).to(torch.float32)
        encoder_input_mask = torch.zeros(len(data['encoder_input']), 1, max_H, max_W).to(torch.float32)
        image_size = []

        for page_id, encoder_input_page in enumerate(data['encoder_input']):
            encoder_input[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = encoder_input_page
            encoder_input_mask[page_id, :, :encoder_input_page.shape[1], :encoder_input_page.shape[2]] = 1.
            image_H = encoder_input_page.shape[1]
            image_W = encoder_input_page.shape[2]
            image_size.append((image_H, image_W))

        batch_encoder_input.append(encoder_input)
        batch_encoder_input_mask.append(encoder_input_mask)
        batch_image_size.append(image_size)

        batch_transcripts.append(data['transcripts'])
        batch_bboxes.append(data['bboxes'])
        batch_lines.append(data['lines'])

    return dict(
        encoder_input=batch_encoder_input,
        encoder_input_mask=batch_encoder_input_mask,
        bboxes=batch_bboxes,
        transcripts=batch_transcripts,
        image_size=batch_image_size,
        lines=batch_lines,
        pdfs=pdf_paths
    )


if __name__ == "__main__":
    # 示例用法
    # folder_path = "../../HRDS/images"
    # subdirectories = list_subdirectories(folder_path)
    # print("Subdirectories in the directory:")
    # print(subdirectories)
    # print(len(subdirectories))
    #
    # folder_path = "../../HRDS/test"
    # subdirectories = list_files_in_directory(folder_path)
    # print("Subdirectories in the directory:")
    # print(subdirectories)
    # print(len(subdirectories))
    #
    # folder_path = "../../HRDS/train"
    # subdirectories = list_files_in_directory(folder_path)
    # print("Subdirectories in the directory:")
    # print(subdirectories)
    # print(len(subdirectories))
    root_dir = '../../../HRDS'
    dataset = CustomDataset(root_dir, 'train')
