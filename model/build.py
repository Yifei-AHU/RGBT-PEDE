from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, build_T_Encoder_from_openai_pretrained, Gran_Decoder
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.cuda.amp import autocast
from .feature_fusion import FusionModule
import torch.nn.functional as F

class DCAlign(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        self.fuse = FusionModule(self.embed_dim)  #### Stage II

        if 'la' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64) # 64 -> 128
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
    
    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats#[:, 0, :]#.float()

    def encode_t_image(self, image):
        image_feats = self.base_model.encode_image(image) # 共享编码器
        return image_feats#[:, 0, :]
    
    def fuse(self, rgb, t):   # Stage I Fuse
       fuser_image_feats = rgb + t # 
       return fuser_image_feats, fuser_image_feats[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        rgb_images = batch['rgb_images']
        t_images = batch['t_images'] 

        caption_ids = batch['mlm_ids'] 

        delete_color_random_mask_captions = batch['delete_color_random_mask'] 
        complete_random_mask_captions = batch['random_mask'] 

        mask_nouns_caption_ids = batch['delete_color_mask_nouns'] 
        mask_color_caption_ids = batch['mask_color'] 
 
        rgb_image_feats, t_image_features, text_feats, delete_color_rd_mask_text_feats, complete_rd_mask_text_feats, mask_nouns_text_feats, mask_color_text_feats = self.base_model(rgb_images, t_images, caption_ids, delete_color_random_mask_captions, complete_random_mask_captions, mask_nouns_caption_ids, mask_color_caption_ids)

        rgb_i_feats = rgb_image_feats[:, 0, :].float()
        t_i_feats = t_image_features[:, 0, :].float() 
        
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        no_color_rd_mask_t_feats = delete_color_rd_mask_text_feats[torch.arange(delete_color_rd_mask_text_feats.shape[0]), delete_color_random_mask_captions.argmax(dim=-1)].float()
        complete_rd_mask_t_feats = complete_rd_mask_text_feats[torch.arange(complete_rd_mask_text_feats.shape[0]), complete_random_mask_captions.argmax(dim=-1)].float()
        
        # RGB T Fuse
        fuser_image_feats, fuser_i_feats = self.fuse(rgb_image_feats, t_image_features) 
 
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        #### Stage I : ga+la Stage II: ga

        if 'ga' in self.current_task:  # Global Align
            # Stage I 
            ret.update({'uib_loss':self.args.uib_weights*objectives.compute_uib_loss(t_i_feats, no_color_rd_mask_t_feats)})  
            ret.update({'sdm_loss':self.args.sdm_weights*objectives.compute_sdm(fuser_i_feats, t_feats, batch['pids'], logit_scale)})
            ret.update({'bia_loss':self.args.bia_weights*objectives.compute_infonce(rgb_i_feats, complete_rd_mask_t_feats)}) 

        # Stage I
        if 'la' in self.current_task: # Local Align

            fuser_image_feats = fuser_image_feats.half()

            with torch.no_grad(): 
                ori_x = self.base_model.encode_text(batch['ori_caption_ids']) 
                ori_delete_color_x = self.base_model.encode_text(batch['ori_delete_color_caption']) 

            mask_noun_indice = (mask_nouns_caption_ids == 49405).float()
            mask_color_indice =  (mask_color_caption_ids == 49405).float()   

            recover_color = self.cross_former(mask_color_text_feats, rgb_image_feats, rgb_image_feats) 
            recover_noun = self.cross_former(mask_nouns_text_feats, t_image_features, t_image_features)  

            crs_loss = objectives.compute_specific_location_simi_loss(ori_x, recover_color, mask_color_indice)
            cus_loss = objectives.compute_specific_location_simi_loss(ori_delete_color_x, recover_noun, mask_noun_indice)

            x = self.cross_former(text_feats, fuser_image_feats, fuser_image_feats)  
            x_score = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores2 = x_score.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'ar_loss': self.args.ar_weights*objectives.compute_mlm(scores2, mlm_labels)})  
            ret.update({'crs_loss': self.args.crs_weights*crs_loss})
            ret.update({'cus_loss': self.args.cus_weights*cus_loss})

        return ret


def build_model(args, num_classes=11003):
    model = DCAlign(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
