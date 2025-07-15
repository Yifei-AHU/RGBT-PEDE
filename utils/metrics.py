from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("IRRA.eval")

    # def __init__(self, train_loader):
    #     self.train_loader = train_loader 
    #     self.logger = logging.getLogger("IRRA.eval")

    def denormalize(self, tensor_img, mean, std):
        device = tensor_img.device
        tensor_img = tensor_img.squeeze(0) 
        mean = torch.tensor(mean, device=device).view(3, 1, 1)
        std = torch.tensor(std, device=device).view(3, 1, 1)
        img = tensor_img * std + mean
        img = img.clamp(0, 1)  # ensure in [0,1]
        img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
        return img

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        # ## 画热力图的代码
        # for _, batch in enumerate(self.train_loader):
        #     # batch = {k: v.to(device) for k, v in batch.items()}
        #     batch = {
        #         k: v if k == 'text_caption' else v.to(device)
        #         for k, v in batch.items()
        #     }
        #     rgb_images = batch['rgb_images']
        #     t_images = batch['t_images'] 
        #     captions = batch['ori_caption_ids']
        #     rgb_img_feat = model.encode_image(rgb_images)
        #     t_img_feat = model.encode_image(t_images)
        #     text_feat = model.encode_text(captions) # 文本的class token

        #     fuser_img_feats, _ = model.fuse(rgb_img_feat, t_img_feat)

        #     text_cls = F.normalize(text_feat, dim=-1)
        #     rgb_local_patches = F.normalize(rgb_img_feat[:,1:,:], dim=-1).float()
        #     t_local_patches = F.normalize(t_img_feat[:,1:,:], dim=-1).float()
        #     fuser_local_patches = F.normalize(fuser_img_feats[:,1:,:], dim=-1).float()

        #     text_rgb_similarity = torch.bmm(
        #         text_cls.unsqueeze(1),                     # [B, 1, 512]
        #         rgb_local_patches.transpose(1, 2)              # [B, 512, 192]
        #         ).squeeze(1)
        #     text_t_similarity = torch.bmm(
        #         text_cls.unsqueeze(1),                     # [B, 1, 512]
        #         t_local_patches.transpose(1, 2)              # [B, 512, 192]
        #         ).squeeze(1)
        #     text_fuser_similarity = torch.bmm(
        #         text_cls.unsqueeze(1),                     # [B, 1, 512]
        #         fuser_local_patches.transpose(1, 2)              # [B, 512, 192]
        #         ).squeeze(1)
        #     rgb_heatmaps = text_rgb_similarity.view(-1, 24, 8) 
        #     t_heatmaps= text_t_similarity.view(-1, 24, 8) 
        #     fuser_heatmnaps = text_fuser_similarity.view(-1, 24, 8) 

        #     # 上采样热图
        #     rgb_heatmaps = rgb_heatmaps.cpu().detach().numpy()
        #     rgb_heatmaps = F.interpolate(
        #         torch.tensor(rgb_heatmaps).unsqueeze(0),  # [1,1,12,16]
        #         size=(384, 128),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze().numpy()

        #     t_heatmaps = t_heatmaps.cpu().detach().numpy()
        #     t_heatmaps = F.interpolate(
        #         torch.tensor(t_heatmaps).unsqueeze(0),  # [1,1,12,16]
        #         size=(384, 128),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze().numpy()

        #     fuser_heatmnaps = fuser_heatmnaps.cpu().detach().numpy()
        #     fuser_heatmnaps = F.interpolate(
        #         torch.tensor(fuser_heatmnaps).unsqueeze(0),  # [1,1,12,16]
        #         size=(384, 128),
        #         mode='bilinear',
        #         align_corners=False
        #     ).squeeze().numpy()     

        #     rgb_heatmaps = (rgb_heatmaps - rgb_heatmaps.min()) / (rgb_heatmaps.max() - rgb_heatmaps.min() + 1e-8)
        #     rgb_heatmaps = 1.0 - rgb_heatmaps
        #     t_heatmaps = (t_heatmaps - t_heatmaps.min()) / (t_heatmaps.max() - t_heatmaps.min() + 1e-8)
        #     t_heatmaps = 1.0 - t_heatmaps
        #     fuser_heatmnaps = (fuser_heatmnaps - fuser_heatmnaps.min()) / (fuser_heatmnaps.max() - fuser_heatmnaps.min() + 1e-8)

        #     # 反归一化图像
        #     mean = [0.48145466, 0.4578275, 0.40821073]
        #     std = [0.26862954, 0.26130258, 0.27577711]
        #     oir_rgb_img_np = self.denormalize(rgb_images, mean, std)
        #     oir_t_img_np = self.denormalize(t_images, mean, std)

        #     caption = batch['text_caption'][0]
        #     import re
        #     img_name = re.sub(r'[\\/*?:"<>|]', "_", caption)
        #     img_name = img_name[:50]
        #     rgb_save_path = os.path.join('/data/dengyifei/Data/RGBT-PEDES/rgb_heatmap2/', f"{img_name}_result.png")
        #     t_save_path = os.path.join('/data/dengyifei/Data/RGBT-PEDES/t_heatmap2/', f"{img_name}_result.png")

        #     import matplotlib.pyplot as plt
        #     H, W = 384, 128
        #     plt.figure(figsize=(W / 25.6, H / 25.6), dpi=100)
        #     plt.imshow(oir_rgb_img_np)
        #     plt.imshow(rgb_heatmaps, cmap='jet', alpha=0.3)
        #     plt.axis('off')
        #     plt.savefig(rgb_save_path, bbox_inches='tight', pad_inches=0)
        #     # plt.savefig('test_hetamap.jpg', bbox_inches='tight', pad_inches=0)

        #     plt.figure(figsize=(W / 25.6, H / 25.6), dpi=100)
        #     plt.imshow(oir_t_img_np)
        #     plt.imshow(t_heatmaps, cmap='jet', alpha=0.3)
        #     plt.axis('off')
        #     plt.savefig(t_save_path, bbox_inches='tight', pad_inches=0)
        ##########################

        qids, gids, qfeats, gfeats = [], [], [], []

        # text
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat)
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # rgb  t 
        for pid, rgb_img, t_img in self.img_loader:
            rgb_img = rgb_img.to(device)
            t_img = t_img.to(device)
            with torch.no_grad():
                rgb_img_feat = model.encode_image(rgb_img)
                t_img_feat = model.encode_image(t_img)
                _, img_feat = model.fuse(rgb_img_feat, t_img_feat) 
                # img_feat = model.fuse(rgb_img_feat, t_img_feat).float()
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        # #### 计算 FLOPs #####
        # from thop import profile
        # flops, params = profile(model, inputs=(rgb_img[0].unsqueeze(0),t_img[0].unsqueeze(0),caption[0].unsqueeze(0)))
        # flops_g = flops / 1e9  # 转换为G
        # print(f"FLOPs: {flops_g:.2f} G")

        # start_event = torch.cuda.Event(enable_timing=True)
        # end_event = torch.cuda.Event(enable_timing=True)

        # with torch.no_grad():
        #     start_event.record()
        #     output = model(rgb_img[0].unsqueeze(0),t_img[0].unsqueeze(0),caption[0].unsqueeze(0))
        #     end_event.record()

        # # 等待 CUDA 计算完成
        # torch.cuda.synchronize()

        # inference_time_ms = start_event.elapsed_time(end_event)  # 结果单位本来就是 ms
        # print(f"Inference time: {inference_time_ms:.3f} ms") 

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features rgb

        similarity = qfeats @ gfeats.t() 

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
