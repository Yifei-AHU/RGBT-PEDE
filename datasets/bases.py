from typing import List
from torch.utils.data import Dataset
import torchvision.transforms as T
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import spacy
import math
import numpy as np
from openai import OpenAI

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, rgb_img_paths, t_img_paths, transform=None):
        self.image_pids = image_pids
        self.rgb_img_paths = rgb_img_paths
        self.t_img_paths = t_img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, rgb_img_path, t_img_path = self.image_pids[index], self.rgb_img_paths[index], self.t_img_paths[index]
        rgb_img = read_image(rgb_img_path,gray=False)
        t_img = read_image(t_img_path,gray=True)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            t_img = self.transform(t_img)
        return pid, rgb_img, t_img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        # 使用 对应单词的token
        self.color_words = [736, 4287, 4481, 1901, 1746, 5496, 3360, 2866, 7048, 1449, 1579, 20501, 5598, 12054, 23650, 
                            19899, 3467, 2209, 21002, 26677, 13919, 4852, 20032, 22821] # v1
        self.tokenizer = SimpleTokenizer()
        self.nlp = spacy.load("en_core_web_sm")
        self.flip_transform = T.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.dataset)

    def format_sentence(self, sentence):
        # 在每个句号后添加空格
        sentence = sentence.replace('.', '. ')
        # 在每个逗号后添加空格
        sentence = sentence.replace(',', ', ')
        # 去掉多余的空格
        return ' '.join(sentence.split())

    def replace_colors(self, sentence):
        # 将句子中的颜色单词替换为空字符串，并去除多余空格
        result = re.sub(r'\b(?:' + '|'.join(self.color_words) + r')\b', '<|mask|>', sentence, flags=re.IGNORECASE)
        return ' '.join(result.split())  # 去除多余的空格

    def __getitem__(self, index):
        pid, image_id, rgb_img_path, t_img_path, caption, no_color_caption = self.dataset[index]
        rgb_img = read_image(rgb_img_path, gray=False)
        t_img = read_image(t_img_path, gray=True)
        if self.transform is not None:
            if torch.rand(1).item() > 0.5:
                rgb_img = self.flip_transform(rgb_img)
                t_img = self.flip_transform(t_img)
            rgb_img = self.transform(rgb_img)
            t_img =  self.transform(t_img)

        caption_process = self.nlp(caption)
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        nouns = [token.text for token in caption_process if token.pos_ == "NOUN"]

        mask_ratio = 0.15

        num_to_mask2 = min(len(nouns), max(1, math.ceil(len(nouns) * mask_ratio)))  
        if num_to_mask2 > 0:
            nouns_to_mask2 = random.sample(nouns, num_to_mask2)
        else:
            nouns_to_mask2 = []

        nouns_to_mask2 = ' '.join(nouns_to_mask2)

        nouns_to_mask_tokens_2 = tokenize(nouns_to_mask2, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        start_index = (nouns_to_mask_tokens_2 == 49406).nonzero(as_tuple=True)[0].item() + 1
        end_index = (nouns_to_mask_tokens_2 == 49407).nonzero(as_tuple=True)[0].item()
        nouns_token2 = nouns_to_mask_tokens_2[start_index:end_index].tolist()  # 被mask的noun token
       
        delete_color_caption_tokens = tokenize(no_color_caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        
        delete_color_caption_tokens_clone = delete_color_caption_tokens.clone()

        delete_color_random_mask_tokens, _ = self._build_random_masked_tokens_and_labels(delete_color_caption_tokens.cpu().numpy(), mask_ratio) 
        delete_color_mask_noun_tokens, mnm_lables = self._build_masked_tokens_and_labels_from_list(delete_color_caption_tokens_clone, nouns_token2) 
        
        ori_caption_tokens_1 = caption_tokens.clone()  
        ori_caption_tokens_2 = caption_tokens.clone()  
        ori_caption_tokens_3 = caption_tokens.clone()   

        color_tokens = [value for value in self.color_words if value in ori_caption_tokens_1.cpu().numpy()] 
        num_to_mask3 = min(len(color_tokens), max(1, math.ceil(len(color_tokens) * 2 * mask_ratio)))
        if num_to_mask3 > 0:
            color_to_mask3 = random.sample(color_tokens, num_to_mask3)
        else:
            color_to_mask3 = []
            
        random_mask_tokens, _ = self._build_random_masked_tokens_and_labels(ori_caption_tokens_1.cpu().numpy(), mask_ratio) 
        color_mask_tokens, mcm_lables = self._build_masked_tokens_and_labels_from_list(ori_caption_tokens_3.cpu().numpy(), color_to_mask3)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy(), mask_ratio)  
     
        ret = {
            'pids': pid,
            # 'text_caption': caption,
            'image_ids': image_id,
            'rgb_images': rgb_img,
            't_images': t_img,
            'ori_caption_ids': ori_caption_tokens_2,
            'ori_delete_color_caption': delete_color_caption_tokens_clone,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels,
            'delete_color_random_mask': delete_color_random_mask_tokens, 
            'delete_color_mask_nouns': delete_color_mask_noun_tokens,  
            'random_mask': random_mask_tokens,
            'mask_color': color_mask_tokens,  # RGB   mask_color_sentence_tokens
            'mnm_lables': mnm_lables,
            'mcm_lables': mcm_lables,
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens, p):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < p:
                    prob /= p

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)
    
    def _build_random_masked_tokens_and_labels_v2(self, tokens, k): # 根据数量掩码不是比例
        """
        Mask exactly k random tokens for Language Model task.
        :param tokens: list of int, tokenized sentence.
        :param k: int, number of tokens to mask.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        # 过滤掉值为 49405 的 token，并记录可掩码的 token 索引
        candidate_indices = [i for i, token in enumerate(tokens) if 0 < token < 49405]

        # 如果可掩码的 token 数量小于 k，则掩码所有可掩码的 token
        num_to_mask = min(k, len(candidate_indices))

        # 随机选择 num_to_mask 个 token 进行掩码
        masked_indices = random.sample(candidate_indices, num_to_mask)

        labels = [0] * len(tokens)  # 初始化 labels，默认值为 0（表示不需要预测）
        for i in masked_indices:
            original_token = tokens[i]
            tokens[i] = mask
            # 记录原始 token 到 labels 中
            labels[i] = original_token

        return torch.tensor(tokens), torch.tensor(labels)
    
    def _build_masked_tokens_and_labels_from_list(self, tokens, tokens_list):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        
        labels = []
        
        labels = np.where(np.isin(tokens, tokens_list), tokens, 0)
        tokens = np.where(np.isin(tokens, tokens_list), mask, tokens)

        return torch.tensor(tokens), torch.tensor(labels)
