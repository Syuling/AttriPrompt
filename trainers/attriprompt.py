import os.path as osp
import csv
import torch
from tqdm import tqdm

import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss
import os
import pickle
import numpy as np
import json
from PIL import Image
from sklearn.metrics import f1_score

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
# from lavis.models import load_model_and_preprocess

from clip1 import clip
import random
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.N_CTX_VISION,
                          "language_ctx": cfg.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model

def _get_base_text_features(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    if clip_model.dtype == torch.float16:
        text_encoder = text_encoder.cuda()
    
    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    with torch.no_grad():
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            if clip_model.dtype == torch.float16:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))  # not support float16 on cpu
            else:
                text_embeddings.append(text_encoder(embeddings.cuda(), tokens.cuda()))
    text_embeddings = torch.stack(text_embeddings).mean(1)
    # text_encoder = text_encoder.to(device)
    return text_embeddings.to(device)

class KLLoss(_Loss):
    def __init__(self, T, alpha=1.):
        super(KLLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, stu_logits, tea_logits):
        tea_logits = self.alpha * tea_logits+ (1 - self.alpha) * stu_logits
        
        tea_prob = F.softmax(tea_logits / self.T, dim=-1)
        kl_loss = -tea_prob * F.log_softmax(stu_logits / self.T,
                                            -1) * self.T * self.T
        kl_loss = kl_loss.sum(1).mean()

        return kl_loss

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.ATTRIPROMPT.N_CTX
        ctx_init = cfg.TRAINER.ATTRIPROMPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.ATTRIPROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, cfg.MODEL.NUM_PROMPT, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.text_prompt = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        # template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        # classnames = [template.format(name.replace("_", " ")) for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) #c*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS c*1*512
        self.register_buffer("token_suffix", embedding[:, 1+(n_ctx*cfg.MODEL.TOP_K):, :])  # CLS, EOS c*n*512

        # 构造无类别信息的 prompt
        nc_prompts = [prompt_prefix+'.' for _ in classnames]
        nc_tokenized_prompts = torch.cat([clip.tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts #1*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts).type(dtype)
            # print(embedding.size())
        self.register_buffer('nc_token_prefix', embedding[:, :1,:])  #没有类别信息的前缀 1*1*512
        self.register_buffer('nc_token_suffix', embedding[:, 1+(n_ctx*cfg.MODEL.TOP_K):,:]) #没有类别信息的后缀 1*n*512
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.ATTRIPROMPT.CLASS_TOKEN_POSITION

    def forward(self, indices, prob=None, label=None, infer=False):
        
        batch = indices.shape[0]
        if label != None:
            # print(self.text_prompt[label].size(),self.text_prompt[label][indices].size()) #[32, 10, 16, 512]
            if prob != None:
                ctx = (prob.unsqueeze(-1).unsqueeze(-1) * self.text_prompt[label][torch.arange(batch).unsqueeze(1),indices]).view(batch, -1, self.ctx_dim)
            else:
                ctx = self.text_prompt[label][torch.arange(batch).unsqueeze(1),indices].view(batch, -1, self.ctx_dim) #b*(k*16)*512

            prefix = self.nc_token_prefix[label] #32*c*1*512 [32, 47, 1, 512]
            suffix = self.nc_token_suffix[label] #32*c*n*512 [32, 47, 28, 512]

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            tokenized_prompts = self.nc_tokenized_prompts.view(self.n_cls,-1)[label.cpu()]
            
            self.prompts = prompts
            self.prompts_token = tokenized_prompts

        else:
            # print(self.text_prompt.view(-1,self.n_ctx,self.ctx_dim).size(),self.text_prompt.view(-1,self.n_ctx,self.ctx_dim)[indices].size())
            # ctx = self.text_prompt.view(-1,self.n_ctx,self.ctx_dim)[indices].view(batch,-1,self.ctx_dim) #b*(k*16)*512
            index_expanded = indices.unsqueeze(-1).unsqueeze(-1)
            
            if prob != None:
                ctx = (prob.unsqueeze(-1).unsqueeze(-1) * self.text_prompt.unsqueeze(0).repeat(batch,1,1,1,1).gather(2, index_expanded.expand(-1, -1, -1, self.n_ctx, self.ctx_dim))).view(batch, self.n_cls,-1,self.ctx_dim) 
            else:
                ctx = self.text_prompt.unsqueeze(0).repeat(batch,1,1,1,1).gather(2, index_expanded.expand(-1, -1, -1, self.n_ctx, self.ctx_dim)).view(batch, self.n_cls,-1,self.ctx_dim) 
            # print(ctx.size())

            # if ctx.dim() == 2:
            #     ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            # print('ctx: ',ctx.shape) [32, 48, 512]
            prefix = self.nc_token_prefix.unsqueeze(0).repeat(batch,1,1,1) #32*c*1*512 [32, 47, 1, 512]
            # print('prefix: ',prefix.shape)
            suffix = self.nc_token_suffix.unsqueeze(0).repeat(batch,1,1,1) #32*c*n*512 [32, 47, 28, 512]
            # print('suffix: ',suffix.shape)
            # ctx = ctx.unsqueeze(1).repeat(1, self.n_cls, 1, 1) #32*c*(k*16)*512 [32, 47, 48, 512]
            # print('ctx: ',ctx.shape) 

            if self.class_token_position == "end":
                
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=2,
                )

            
            # print('prompts: ',prompts.size()) #[32, 47, 77, 512]
            prompts = prompts.view(batch*self.n_cls, -1, self.ctx_dim)  #
            # # print('prompts: ',prompts.size())
            tokenized_prompts = self.nc_tokenized_prompts.view(self.n_cls,-1)
            tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch,1,1).view(batch*self.n_cls, -1)
            
            
            # print('prompts: ',prompts.size())
            # tokenized_prompts = self.tokenized_prompts.view(self.n_cls,-1)
            # tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch,1,1)
            self.prompts = prompts
            self.prompts_token = tokenized_prompts
        
        return prompts, tokenized_prompts

    def only_prefix(self, label):

        ctx = self.text_prompt[label].view(-1, self.n_ctx, self.ctx_dim) #(c*10)*16*512
        prompt_size = ctx.shape[0] 
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1) #(c*10)*77
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1) #(c*10)*1*512
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1) #(c*10)*n*512
        nc_prompts = torch.cat([prefix, ctx, suffix],dim=1)  #(c*10)*77*512
        return nc_prompts, nc_tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, train_loader):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        n_cls = len(classnames)
        ctx_dim = clip_model.visual.proj.shape[1]


        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        org_clip_model = load_clip_to_cpu(cfg, True)
        self.org_image_encoder = org_clip_model.visual
        self.org_text_encoder = TextEncoder(org_clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.class_text_features = _get_base_text_features(cfg, classnames, clip_model, self.org_text_encoder).cuda()
        
        org_class_text_features = self.class_text_features.clone()
        self.org_class_text_features = org_class_text_features / org_class_text_features.norm(dim=-1, keepdim=True)
        

        text_key = torch.empty(n_cls, self.cfg.MODEL.NUM_PROMPT, ctx_dim, dtype=self.dtype).cuda() #C*10*512
        nn.init.normal_(text_key, std=0.02)
        self.text_key = nn.Parameter(text_key)
        
        # self.text_key_param = nn.Parameter(torch.tensor(1.0))
        self.text_key_fc = nn.Linear(ctx_dim, n_cls,dtype=self.dtype,bias=False).cuda()
        self.text_key_fc.weight.data = self.class_text_features
        
        self.text_key_local = nn.Sequential(
            nn.Linear(clip_model.visual.proj.shape[0], ctx_dim,dtype=self.dtype,bias=False),
            # nn.Linear(ctx_dim, ctx_dim,dtype=self.dtype)
        ).cuda()


        self.logit_scale = clip_model.logit_scale
        self.n_cls = n_cls
        self.one_hot = torch.eye(n_cls).cuda()

    def forward(self, image, label=None, test=True): #32*512  32

        batch = len(image)
        if test:
            image_features, local_img = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            
            
            logits_img = logit_scale * self.text_key_fc(image_features)
            pred_label = torch.argmax(logits_img, dim=1)
            local_img = self.text_key_local(local_img)
            local_img = local_img / local_img.norm(dim=-1, keepdim=True)
            local_img = torch.cat((image_features.unsqueeze(1),local_img),1)

            ###b*c*k
            probability = F.cosine_similarity(local_img.unsqueeze(2).unsqueeze(2), \
                                              self.text_key.unsqueeze(0).unsqueeze(0),-1).mean(1)
            

            prob_values, indices = probability.topk(k=min(self.cfg.MODEL.TOP_K, probability.shape[2]), dim=2, largest=True) #32*k
            # print(indices[:2])
            text_prompt, tokenized_prompts = self.prompt_learner(indices,prob=prob_values,infer=True)
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], self.n_cls, -1)  #32*c*512

            text_features = text_features +  self.class_text_features.unsqueeze(0).repeat(batch,1,1)
            image_features = image_features.unsqueeze(1) #32*1*512
            logits =  logit_scale * (image_features * text_features).sum(-1) #32*c


            return logits, text_features, prob_values, indices
        
        else:
            # mask_image = PatchEnhance(image,0.1)
            with torch.no_grad():
                org_image_features,_ = self.org_image_encoder(image.type(self.dtype))
                org_image_features = org_image_features / org_image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                tea_logits = logit_scale * org_image_features @ self.org_class_text_features.t()

            image_features, local_img = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits_img = logit_scale * self.text_key_fc(image_features)

           
            local_img = self.text_key_local(local_img)
            local_img = local_img / local_img.norm(dim=-1, keepdim=True)
            local_img = torch.cat((image_features.unsqueeze(1),local_img),1)
            
            probability = F.cosine_similarity(local_img.unsqueeze(2), self.text_key[label].unsqueeze(1),-1).mean(1) #32*196*47*10 --> 32*47*10
            #
            
            prob_values, indices = probability.topk(k=min(self.cfg.MODEL.TOP_K, probability.shape[1]), dim=1, largest=True)  #32*47*k
            text_prompt, tokenized_prompts = self.prompt_learner(indices,prob=prob_values,label=label)
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.unsqueeze(1).repeat(1,self.n_cls,1)  #32*c*512 



            text_features = text_features + self.class_text_features.unsqueeze(0).repeat(batch,1,1)
            # print(text_features)
            image_features = image_features.unsqueeze(1) #32*1*512
            logit_scale = self.logit_scale.exp()
            logits = (image_features * text_features).sum(-1) #32*c


           

            loss_t = F.l1_loss(self.org_class_text_features, self.class_text_features,
                                      reduction='mean')
            
           
            return logits, tea_logits, logits_img, loss_t


@TRAINER_REGISTRY.register()
class AttriPrompt(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ATTRIPROMPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)


        if cfg.TRAINER.ATTRIPROMPT.PREC == "fp32" or cfg.TRAINER.ATTRIPROMPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            # self.blip_model.float()
        self.clip_model = clip_model

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, self.train_loader_x)
        self.kl_loss = KLLoss(0.07)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "text_key" in name or "text_prompt" in name or "VPT" in name:
                param.requires_grad_(True)
                print(name)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of learable params:', n_parameters)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.logit_scale = self.model.logit_scale.exp()
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("attri_prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ATTRIPROMPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.ATTRIPROMPT.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, tea_output, logits_img, loss_t = self.model(image,label=label,test=False)
            loss_main = F.cross_entropy(output, label)
            loss_img_main = F.cross_entropy(logits_img, label)
            loss_kl = self.kl_loss(logits_img,tea_output) #+ self.kl_loss(logits_text,tea_output)
            
            loss = loss_main + self.cfg.LOSS.ALPHA  * loss_img_main + self.cfg.LOSS.BETA * (loss_kl + loss_t) #+ 0.1 * (select_entropy)#+ self.cfg.LOSS.ALPHA *  loss_tt
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            'loss_main':loss_main.item(),
            'loss_img':loss_img_main.item(),
            'loss_t':loss_t.item(),
            'loss_kl':loss_kl.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    @torch.no_grad()
    def test(self, split=None, text_features=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # Initialize a dictionary to store per-class accuracy
        
        
        if self.cfg.cont_dis:
            n_max = self.cfg.Head
            n_min = 1
            alpha =0.6
            cls_idx = np.arange(1, self.n_cls + 1)
            cls_counts_cont = 1 / (cls_idx ** alpha)

            # 归一化到 [1, head]
            cls_counts = (cls_counts_cont - cls_counts_cont.min()) / (cls_counts_cont.max() - cls_counts_cont.min())
            cls_counts = cls_counts * (n_max - n_min) + n_min
            cls_counts = np.round(cls_counts).astype(int)

            # 修正首尾
            cls_counts[0] = n_max
            cls_counts[-1] = n_min
            np.random.seed(self.cfg.SEED)
            np.random.shuffle(cls_counts)
            shot_index = cls_counts
        else:
        
            rand_index = np.random.randint(0, 5, size=(self.n_cls,))
            shots = np.array([1, 2, 4, 8] + [self.cfg.Head])
            shot_index = shots[rand_index]
        
        per_class_correct = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}
        per_it_sim = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}
        per_class_total = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}



        text_sim = 0
        count = 0
        t_id = 0


        self.model.eval()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, ntext_features, prob_values, indices = self.model_inference(input)
            output_soft = F.softmax(output,1)


            t_id += len(label)
            count += 1
            self.evaluator.process(output, label)

            # Calculate per-class accuracy
            for i in range(len(label)):
                true_label = label[i].item()
                pred_label = output[i].argmax().item()
                text_sim += output_soft[i][true_label]

                if true_label not in per_class_correct:
                    per_class_correct[true_label] = 0
                    per_class_total[true_label] = 0

                per_class_total[true_label] += 1
                
                if pred_label == true_label:
                    per_class_correct[true_label] += 1

                if shot_index[true_label] in [1]:
                    per_class_total['sample_1'] += 1
                    per_it_sim['sample_1'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_1'] += 1

                elif shot_index[true_label] in [2]:
                    per_class_total['sample_2'] += 1
                    per_it_sim['sample_2'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_2'] += 1

                elif shot_index[true_label] in [4]:
                    per_class_total['sample_4'] += 1
                    per_it_sim['sample_4'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_4'] += 1

                elif shot_index[true_label] in [8]:
                    per_class_total['sample_8'] += 1
                    per_it_sim['sample_8'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_8'] += 1
                elif shot_index[true_label] in [16]:
                    per_class_total['sample_16'] += 1
                    per_it_sim['sample_16'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_16'] += 1
                elif shot_index[true_label] in [32]:
                    per_class_total['sample_32'] += 1
                    per_it_sim['sample_32'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_32'] += 1
                elif shot_index[true_label] in [64]:
                    per_class_total['sample_64'] += 1
                    per_it_sim['sample_64'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_64'] += 1
                else:
                    per_class_total['sample_128'] += 1
                    per_it_sim['sample_128'] += output_soft[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_128'] += 1


        results = self.evaluator.evaluate()

        text_sim = text_sim/count

        # Calculate and log per-class accuracy
        per_class_accuracy = {k: per_class_correct[k] / per_class_total[k] if per_class_total[k] > 0 else 0
                            for k in per_class_correct.keys()}
        
        per_it_sim = {k: per_it_sim[k] / per_class_total[k] if per_class_total[k] > 0 else 0
                            for k in per_it_sim.keys()}

        # Log overall results
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        for k, v in per_class_accuracy.items():
            tag = f"{split}/per_class_accuracy/{k}"
            self.write_scalar(tag, v, self.epoch)
            print(f"Class {k} accuracy: {v:.4f}")

        file_path = f'./output_csv/reesults.csv'
        last_result = list(results.values())[0]
        data = [self.cfg.DATASET.NAME, self.cfg.DATASET.NUM_SHOTS, self.cfg.SEED, self.cfg.MODEL.NUM_PROMPT,self.cfg.MODEL.TOP_K,self.cfg.LOSS.ALPHA,self.cfg.LOSS.BETA, \
                per_class_accuracy['sample_1'], per_class_accuracy['sample_2'],per_class_accuracy['sample_4'],per_class_accuracy['sample_8'],per_class_accuracy['sample_16'],per_class_accuracy['sample_32'],per_class_accuracy['sample_64'],per_class_accuracy['sample_128'],\
                per_it_sim['sample_1'], per_it_sim['sample_2'],per_it_sim['sample_4'],per_it_sim['sample_8'],per_it_sim['sample_16'],per_it_sim['sample_32'],per_it_sim['sample_64'],per_it_sim['sample_128'],\
                    last_result,results["macro_f1"],text_sim.item()]
        file_exists = osp.isfile(file_path)

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Dataset Name', 'Shots', 'Seed', 'Num_prompt','Top_k','ALPHA','BETA','Acc1','Acc2','Acc4','Acc8','Acc16','Acc32','Acc64','Acc128','Sim1','Sim2','Sim4','Sim8','Sim16','Sim32','Sim64','Sim128','Test Result','UF1','Text Simalarity'])
            writer.writerow(data)

        return list(results.values())[0]


    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
