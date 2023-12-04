# -*- coding:utf-8 -*-

import sys
import os
import json
import tqdm
from strcut_recover.libs.configs import cfg, setup_config
from strcut_recover.libs.model import build_model
from strcut_recover.libs.utils import logger
from strcut_recover.libs.data import create_train_dataloader
from datasets import Dataset
import torch
from transformers import TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, targets, predictions):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')

        # 计算权重项
        weights = (1 - F.softmax(predictions, dim=1).gather(1, targets.view(-1, 1))) ** self.gamma

        # 计算最终损失
        focal_loss = torch.mean(weights * ce_loss)
        return focal_loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, targets, predictions):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(predictions, targets)
        return ce_loss


def setup_seed(seed):
    """
    设置随机种子
    @param seed:
    @return:
    """
    import torch, numpy as np, random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init():
    """
    初始化配置参数 end2end_system\strcut_recover\libs\configs\infer.py
    @return:
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='infer')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    setup_config(args.cfg)

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    logger.setup_logger('Document Decoder Model', cfg.work_dir, 'train.log')
    logger.info('Use config: %s' % args.cfg)


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


if __name__ == '__main__':
    setup_seed(2021)
    init()
    # train_dataloader = create_train_dataloader(cfg.ly_vocab, cfg.re_vocab, cfg.train_data_path, cfg.train_batch_size,
    #                                            cfg.train_num_workers)
    train_dataloader = create_train_dataloader(cfg.ly_vocab, cfg.re_vocab, 'E:\Static\HRDoc\end2end_system\HRDS',
                                               cfg.train_batch_size,
                                               cfg.train_num_workers)

    logger.info(
        'Train dataset have %d samples, %d batchs with batch_size=%d' % \
        (
            len(train_dataloader.dataset),
            len(train_dataloader.batch_sampler),
            train_dataloader.batch_size
        )
    )
    # 模型参数保存输出地址
    training_args = TrainingArguments(output_dir="./strcut_recover/output")

    model = build_model(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model.train()
    tokenizer = cfg.tokenizer
    extractor = cfg.extractor.to(device)
    bert = cfg.bert.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = None
    num_epochs = 2

    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):
        loss_hist.reset()

        for it, data_batch in enumerate(tqdm.tqdm(train_dataloader)):
            encoder_input = [data.to(cfg.device) for data in data_batch['encoder_input']]
            encoder_input_mask = [data.to(cfg.device) for data in data_batch['encoder_input_mask']]
            encoder_input_bboxes = [[torch.tensor(page).to(cfg.device).float() for page in data] for data in
                                    data_batch['bboxes']]
            transcripts = data_batch['transcripts']
            image_size = data_batch['image_size']
            pdf_paths = data_batch['pdfs']
            batch_lines = data_batch['lines']

    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    #         loss_dict = model(images, targets)
    #
    #         losses = sum(loss for loss in loss_dict.values())
    #         loss_value = losses.item()
    #
    #         loss_hist.send(loss_value)
    #
    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()
    #
    #         if itr % 50 == 0:
    #             print(f"Iteration #{itr} loss: {loss_value}")
    #
    #         itr += 1
    #
    #     # update the learning rate
    #     if lr_scheduler is not None:
    #         lr_scheduler.step()
    #
    #     print(f"Epoch #{epoch} loss: {loss_hist.value}")

    # 计算并打印损失
    # loss = compute_loss(model_output, batch_data)
    # print("Loss:", loss)
