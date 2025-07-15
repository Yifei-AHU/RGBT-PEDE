import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class RGBTPEDES(BaseDataset):
    """
    RGBT-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'RGBT-PEDES'

    def __init__(self, root='', verbose=True):
        super(RGBTPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.rgb_img_dir = op.join(self.dataset_dir, 'RGB/') # RGB-night-v9/
        self.t_img_dir = op.join(self.dataset_dir, 'TGray/') # TGray

        self.anno_path = op.join(self.dataset_dir, 'caption_2_no_color.json')

        # 挑战子集
        # self.anno_path = op.join(self.dataset_dir, 'challenge_json/strong_light.json') # abnormal_illumination  low_light    test_challenge 

        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RGBT-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                anno['file_path'] = anno['file_path'].split('/')[-1]
                rgb_img_path = op.join(self.rgb_img_dir, anno['file_path'])
                t_img_path = op.join(self.t_img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                no_color_captions = anno['no_color_caption']
                # for caption in captions:
                #     if caption:
                #         dataset.append((pid, image_id, rgb_img_path, t_img_path, caption))
                for caption, no_color_captions in zip(captions, no_color_captions):
                    if caption:
                        dataset.append((pid, image_id, rgb_img_path, t_img_path, caption, no_color_captions))
                image_id += 1
            # for idx, pid in enumerate(pid_container):
            #     assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            rgb_img_paths = []
            t_img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                anno['file_path'] = anno['file_path'].split('/')[-1]
                rgb_img_path = op.join(self.rgb_img_dir, anno['file_path'])
                t_img_path = op.join(self.t_img_dir, anno['file_path'])
                rgb_img_paths.append(rgb_img_path)
                t_img_paths.append(t_img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    if caption:
                        captions.append(caption)
                        caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "rgb_img_paths": rgb_img_paths,
                "t_img_paths": t_img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.rgb_img_dir):
            raise RuntimeError("'{}' is not available".format(self.rgb_img_dir))
        if not op.exists(self.t_img_dir):
            raise RuntimeError("'{}' is not available".format(self.t_img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
