import os.path as osp
from tqdm import tqdm
from collections import OrderedDict
import math
import copy
import csv
import os
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import pickle

# from lavis.models import load_model_and_preprocess
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_clip_to_cpu1(cfg):
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

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits, text_features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # self.blip_model, self.processor, _ = load_model_and_preprocess(
        #     name="blip_caption", model_type="large_coco", is_eval=True, device=self.device
        # )
       
        self.clip_model = load_clip_to_cpu1(cfg)
        self.clip_model.to(self.device)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            # self.blip_model.float()


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of learable params:', n_parameters)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

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
        shots = np.array([1,2,4,8,self.cfg.Head])
        shot_index = shots[rand_index]
        
       
        text_sim = 0
       

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            
            input, label = self.parse_batch_test(batch)
            output, newtext_features = self.model_inference(input)
            output = F.softmax(output,1)

           
            self.evaluator.process(output, label)

            # Calculate per-class accuracy
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
        # text_sim = text_sim/count
        # print('text similarity: ',text_sim)

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

        file_path = f'./output_csv/imbalance_results-{self.cfg.TRAINER.NAME}.csv'
        last_result = list(results.values())[0]
        data = [self.cfg.DATASET.NAME, self.cfg.DATASET.NUM_SHOTS, self.cfg.SEED, \
                per_class_accuracy['sample_1'], per_class_accuracy['sample_2'],per_class_accuracy['sample_4'],per_class_accuracy['sample_8'],per_class_accuracy['sample_16'],per_class_accuracy['sample_32'],per_class_accuracy['sample_64'],per_class_accuracy['sample_128'],\
                per_it_sim['sample_1'], per_it_sim['sample_2'],per_it_sim['sample_4'],per_it_sim['sample_8'],per_it_sim['sample_16'],per_it_sim['sample_32'],per_it_sim['sample_64'],per_it_sim['sample_128'],\
                    last_result,results["macro_f1"],text_sim]
        file_exists = osp.isfile(file_path)

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Dataset Name', 'Shots', 'Seed', 'Acc1','Acc2','Acc4','Acc8','Acc16','Acc32','Acc64','Acc128','Sim1','Sim2','Sim4','Sim8','Sim16','Sim32','Sim64','Sim128','Test Result','UF1','Text Simalarity'])
            writer.writerow(data)

        return list(results.values())[0]
    
    def after_epoch(self):
            last_epoch = (self.epoch + 1) == self.max_epoch
            do_test = not self.cfg.TEST.NO_TEST
            meet_checkpoint_freq = (
                (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
            )

            # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            if last_epoch:
                self.save_model(self.epoch, self.output_dir)
                # curr_result = self.test(split="test")
                # file_path = './MAPLE'
                # if not os.path.exists(file_path):
                #     os.makedirs(file_path)
                # file_path = './MAPLE/maple_result_vit16.csv'
                # data = [self.cfg.DATASET.NAME, self.cfg.DATASET.NUM_SHOTS, self.cfg.SEED, curr_result]
                # file_exists = osp.isfile(file_path)

                # with open(file_path, 'a', newline='') as file:
                #     writer = csv.writer(file)
                #     if not file_exists:
                #         writer.writerow(['Dataset Name', 'Shots', 'Seed', 'Test Result'])
                #     writer.writerow(data)
                # is_best = curr_result > self.best_result
                # if is_best:
                #     self.best_result = curr_result
                #     self.save_model(
                #         self.epoch,
                #         self.output_dir,
                #         val_result=curr_result,
                #         model_name="model-best.pth.tar"
                #     )

            # if meet_checkpoint_freq or last_epoch:
                # self.save_model(self.epoch, self.output_dir)