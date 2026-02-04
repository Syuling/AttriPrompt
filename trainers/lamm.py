import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import pickle
import os
import csv
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from helpers import distillation
import copy

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

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

class FeatureLoss(nn.Module):
    def __init__(self,):
        super(FeatureLoss, self).__init__()

    def forward(self, text_features, fix_label_features, label):

        loss_num = 0
        loss = torch.tensor(0.).to(text_features.device)

        for i in range(label.size(0)):
            if torch.sum(label !=  label[i]) > 0:
                index = label[i]    
                dist_ap = (1- F.cosine_similarity(text_features[index].unsqueeze(0), fix_label_features[index].unsqueeze(0).detach()).squeeze())
                loss_num += 1
                loss += dist_ap
                    
        if loss_num != 0:
            return loss / loss_num
        else:
            return loss
        
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        origin_clip = cfg.origin_clip
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
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        if origin_clip:
            self.ctx.requires_grad = False
            
        self.random_init = cfg.random_init

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        
        self.name_lens = name_lens
        self.min_len = min(self.name_lens) # 1
        if self.min_len > 1:
            print("origin len is ", name_lens)
            classnames = self.revise_classnames(classnames, name_lens, self.min_len)
            name_lens = [len(_tokenizer.encode(name)) for name in classnames] 
            print("later len is ", name_lens)
            
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.eval_only = cfg.eval_only
        self.n_cls = n_cls
        print(f"Number of classes: {n_cls}")
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.base = (cfg.DATASET.SUBSAMPLE_CLASSES == "base" or cfg.DATASET.SUBSAMPLE_CLASSES == "all" )
        
        self._init_suffix_dict(classnames, clip_model, dtype)
        self._get_token_classes(dtype)
        
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.base or not self.eval_only:
            self.register_buffer("token_suffix", embedding[:, 1 + self.min_len + n_ctx:, :])  # EOS
            self.register_buffer("token_suffix_test", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
    def revise_classnames(self, classnames, name_lens, min_len):
        if min(name_lens) < min_len:
            for i in range(len(classnames)):
                if name_lens[i] < min_len:
                    classnames[i] = ("<|startoftext|> "*(min_len - name_lens[i])) + classnames[i]
        return classnames        
    
    def _init_suffix_dict(self, classnames, clip_model, dtype):
        print("class_num: ", len(classnames))
       
        self.suffix_classes = [clip_model.token_embedding(clip.tokenize(name)).type(dtype) for name in classnames]

    def _get_token_classes(self, dtype):
            
        if self.base or not self.eval_only:
            # print(len(self.suffix_classes))
            self.token_classes_all = torch.cat(self.suffix_classes).type(dtype)
            # print(self.token_classes_all.shape)         
            self.token_classes = self.token_classes_all[:, 1:self.min_len+1, :]
            # print(self.token_classes.shape)
            if self.random_init:
                nn.init.normal_(self.token_classes, std=0.02)
            self.token_classes = nn.Parameter(self.token_classes)
            self.fix_token = copy.deepcopy(self.token_classes)
            self.fix_token.requires_grad = False
        else:
            pass
        
    def construct_prompts(self, ctx, prefix, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts

    def construct_prompts_v2(self, ctx, prefix, classes, suffix):
        
        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    classes, # (dim0, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        else:
            raise ValueError
        
        return prompts
             
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.base or not self.eval_only:
            classes = self.token_classes
            # print(ctx.shape, prefix.shape, classes.shape, suffix.shape)
            prompts = self.construct_prompts_v2(ctx, prefix, classes, suffix)
        else:
            prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts

    def forward_test(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix_test
    
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.triplet_loss = cfg.triplet_loss
        self.shots = cfg.shots
        self.loss_feature_fn = FeatureLoss()
        
    def _get_origin_feature(self):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner.forward_test()
        fix_label_features = self.text_encoder(prompts, tokenized_prompts)
        
        return fix_label_features
    
    def forward(self, image, label=None):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            if self.triplet_loss:
                
                fix_label_features = self._get_origin_feature()
                fix_label_features = fix_label_features / fix_label_features.norm(dim=-1, keepdim=True)
                
                fix_logists = logit_scale * image_features @ fix_label_features.t()
                loss_logits = distillation(logits, fix_logists, T=2.0)
                
                loss_feature = self.loss_feature_fn(text_features, fix_label_features, label)
                loss_paramter = F.mse_loss(self.prompt_learner.token_classes, self.prompt_learner.fix_token.detach(), reduction="sum")
                
                entorpy_loss = F.cross_entropy(logits, label)
                
                return entorpy_loss + 1 / self.shots * loss_paramter + loss_feature + 0.05 * loss_logits
            else:
                return F.cross_entropy(logits, label)

        return logits


@TRAINER_REGISTRY.register()
class LAMM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        num_params = sum(p.numel() for p in self.model.prompt_learner.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params}")
        
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        
        prec = self.cfg.TRAINER.COOP.PREC
        
        if prec == "amp":
            with autocast():
                loss = model(image, label) 
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss = model(image, label)
            self.model_backward_and_update(loss)
        loss_summary = {"loss": loss.item()}

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
        per_class_correct = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}
        per_it_sim = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}
        per_class_total = {'sample_1':0,'sample_2':0,'sample_4':0,'sample_8':0,'sample_16':0,'sample_32':0,'sample_64':0,'sample_128':0}

        rand_index = np.random.randint(0, 5, size=(self.n_cls,))
        shots = np.array([1, 2, 4, 8, self.cfg.Head])
        shot_index = shots[rand_index]


        for batch_idx, batch in enumerate(tqdm(data_loader)):
            
            input, label = self.parse_batch_test(batch)
            output= self.model_inference(input)
            output = F.softmax(output,1)


            self.evaluator.process(output, label)

            # Calculate per-class accuracy
            for i in range(len(label)):
                true_label = label[i].item()
                pred_label = output[i].argmax().item()
                # text_sim += output[i][true_label].item()

                if true_label not in per_class_correct:
                    per_class_correct[true_label] = 0
                    per_class_total[true_label] = 0

                per_class_total[true_label] += 1
                if pred_label == true_label:
                    per_class_correct[true_label] += 1
                
                if shot_index[true_label] in [1]:
                    per_class_total['sample_1'] += 1
                    per_it_sim['sample_1'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_1'] += 1

                elif shot_index[true_label] in [2]:
                    per_class_total['sample_2'] += 1
                    per_it_sim['sample_2'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_2'] += 1

                elif shot_index[true_label] in [4]:
                    per_class_total['sample_4'] += 1
                    per_it_sim['sample_4'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_4'] += 1

                elif shot_index[true_label] in [8]:
                    per_class_total['sample_8'] += 1
                    per_it_sim['sample_8'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_8'] += 1
                elif shot_index[true_label] in [16]:
                    per_class_total['sample_16'] += 1
                    per_it_sim['sample_16'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_16'] += 1
                elif shot_index[true_label] in [32]:
                    per_class_total['sample_32'] += 1
                    per_it_sim['sample_32'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_32'] += 1
                elif shot_index[true_label] in [64]:
                    per_class_total['sample_64'] += 1
                    per_it_sim['sample_64'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_64'] += 1
                else:
                    per_class_total['sample_128'] += 1
                    per_it_sim['sample_128'] += output[i][true_label].item()
                    if pred_label == true_label:
                        per_class_correct['sample_128'] += 1

        
        results = self.evaluator.evaluate()

        # Calculate and log per-class accuracy
        per_class_accuracy = {k: per_class_correct[k] / per_class_total[k] if per_class_total[k] > 0 else 0
                            for k in per_class_correct.keys()}

        per_it_sim = {k: per_it_sim[k] / per_class_total[k] if per_class_total[k] > 0 else 0
                            for k in per_it_sim.keys()}
        
        # Log overall results
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # Log and print per-class accuracy
        for k, v in per_class_accuracy.items():
            tag = f"{split}/per_class_accuracy/{k}"
            self.write_scalar(tag, v, self.epoch)
            print(f"Class {k} accuracy: {v:.4f}")


        file_path = f'./output_csv/imbalance_results-LAMM.csv'
        last_result = list(results.values())[0]
        data = [self.cfg.DATASET.NAME, self.cfg.DATASET.NUM_SHOTS, self.cfg.SEED, \
                per_class_accuracy['sample_1'], per_class_accuracy['sample_2'],per_class_accuracy['sample_4'],per_class_accuracy['sample_8'],per_class_accuracy['sample_16'],per_class_accuracy['sample_32'],per_class_accuracy['sample_64'],per_class_accuracy['sample_128'],\
                per_it_sim['sample_1'], per_it_sim['sample_2'],per_it_sim['sample_4'],per_it_sim['sample_8'],per_it_sim['sample_16'],per_it_sim['sample_32'],per_it_sim['sample_64'],per_it_sim['sample_128'],\
                    last_result,results["macro_f1"]]
        file_exists = osp.isfile(file_path)

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Dataset Name', 'Shots', 'Seed', 'Acc1','Acc2','Acc4','Acc8','Acc16','Acc32','Acc64','Acc128','Sim1','Sim2','Sim4','Sim8','Sim16','Sim32','Sim64','Sim128','Test Result','UF1'])
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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
                
                
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
