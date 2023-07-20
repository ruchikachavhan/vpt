#!/usr/bin/env python3
"""
a trainer class
"""
import datetime
import time
import torch
import torch.nn as nn
import os
import torchvision
from fvcore.common.config import CfgNode
from fvcore.common.checkpoint import Checkpointer

from ..engine.evaluator import Evaluator
from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss
from ..utils import logging
from ..utils.train_utils import AverageMeter, gpu_mem_usage

import numpy as np
import json

logger = logging.get_logger("visual_prompt")


class Trainer():
    """
    a trainer with below logics:

    1. Build optimizer, scheduler
    2. Load checkpoints if provided
    3. Train and eval at each epoch
    """
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        evaluator: Evaluator,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        logger.info("\tSetting up the optimizer...")
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.checkpointer = Checkpointer(
            self.model,
            save_dir=cfg.OUTPUT_DIR,
            save_to_disk=True
        )

        for name, param in self.model.named_parameters():
            logger.info(f"{name}: {param.requires_grad}")

        if len(cfg.MODEL.WEIGHT_PATH) > 0:
            # only use this for vtab in-domain experiments
            checkpointables = [key for key in self.checkpointer.checkpointables if key not in ["head.last_layer.bias",  "head.last_layer.weight"]]
            self.checkpointer.load(cfg.MODEL.WEIGHT_PATH, checkpointables)
            logger.info(f"Model weight loaded from {cfg.MODEL.WEIGHT_PATH}")

        self.evaluator = evaluator
        self.cpu_device = torch.device("cpu")
        self.augmented = cfg.DATA.AUGMENTED
        self.mode = cfg.DATA.MODE



    def forward_one_batch(self, inputs, targets, is_train):
        """Train a single (full) epoch on the model using the given
        data loader.

        Args:
            X: input dict
            targets
            is_train: bool
        Returns:
            loss
            outputs: output logits
        """
        # move data to device
        if self.augmented:
            inputs, aug_inputs = inputs
            aug_inputs = aug_inputs.to(self.device, non_blocking=True)
            aug_indices = targets[1].view(aug_inputs.shape[0], -1)
            assert aug_indices is not None
            if aug_indices is not None:
                assert len(aug_indices) == inputs.shape[0] # ONly one combination of transforms per image is supported
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets[0].to(self.device, non_blocking=True)  # (batchsize, )
        
        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape} {aug_inputs.shape}")
            logger.info(f"shape of targets: {targets.shape} {aug_indices.shape}")
            if is_train:
                torchvision.utils.save_image(aug_inputs, "aug_input_image.png")
                torchvision.utils.save_image(inputs, "input_image.png")


        # forward
        with torch.set_grad_enabled(is_train):
            if is_train:
                # only one augmented image is passed
                assert len(aug_inputs.size()) == 4
                (_, _, orig_cls_token, orig_features), orig_output = self.model(inputs, aug_indices, vis = True, return_feature=True, get_logits=True)
                (_, _, aug_cls_token, aug_features), aug_output = self.model(aug_inputs, aug_indices, vis = True, return_feature=True, get_logits=True)

            else:
                #  All 
                assert len(aug_inputs.size()) == 5
                aug_cls_token, aug_features = [], []
                orig_cls_token, orig_features = [], []
                orig_output, aug_output = [], []
                for b in range(aug_indices.shape[1]):
                    (_, _, orig_cls_tkn, orig_feats), orig_out = self.model(inputs, aug_indices[:, b], vis = True, return_feature=True, get_logits=True)
                    (_, _, aug_cls_tkn, aug_feats), aug_out = self.model(aug_inputs[:, b, :, :, :], aug_indices[:, b], vis = True, return_feature=True, get_logits=True)
                    # do transpose of all features
                
                    aug_cls_token.append(aug_cls_tkn)
                    aug_features.append(aug_feats)
                    orig_cls_token.append(orig_cls_tkn)
                    orig_features.append(orig_feats)
                    orig_output.append(orig_out)
                    aug_output.append(aug_out)

                # Stack everything 
                aug_cls_token = torch.stack(aug_cls_token).transpose(0, 1)
                aug_features = torch.stack(aug_features).transpose(0, 1)
                orig_cls_token = torch.stack(orig_cls_token).transpose(0, 1)
                orig_features = torch.stack(orig_features).transpose(0, 1)
                orig_output = torch.stack(orig_output).transpose(0, 1)
                aug_output = torch.stack(aug_output).transpose(0, 1)

            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        orig_features.shape, targets.shape))
            if self.cfg.DBG:
                        print("orig_out: ", orig_output.shape)
                        print("aug_out: ", aug_output.shape)

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                if self.cfg.SOLVER.LOSS == 'mse':
                    if self.cfg.SOLVER.USE_CLS_TOKEN:
                        pred = aug_cls_token
                        tgt = orig_cls_token
                    else:
                        pred = aug_features
                        tgt = orig_features
                    loss = self.cls_criterion(
                        pred, tgt, self.cls_weights
                    )
                elif self.cfg.SOLVER.LOSS == 'kl':
                    loss = self.cls_criterion(
                        orig_output, aug_output, self.cls_weights)

                elif self.cfg.SOLVER.LOSS == 'cross_entropy':
                    loss = 0.0
                    if len(orig_output.shape) == 2:
                        loss = self.cls_criterion(
                            orig_output, targets, self.cls_weights, None)
                    elif len(orig_output.shape) == 3:
                        loss = 0.0
                        for i in range(orig_output.shape[1]):
                                loss += self.cls_criterion(
                                    orig_output[:, i, :], targets, self.cls_weights, None)
                       
                    

            if loss == float('inf'):
                logger.info(
                    "encountered infinite loss, skip gradient updating for this batch!"
                )
                return -1, -1
            elif torch.isnan(loss).any():
                logger.info(
                    "encountered nan loss, skip gradient updating for this batch!"
                )
                return -1, -1

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        

        return loss, orig_features, aug_features, orig_cls_token, aug_cls_token, aug_indices

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        if self.augmented:
            aug_inputs = data["augmented"].float()
            aug_indices = data["aug_indices"]
            inputs = [inputs, aug_inputs]
            labels = [labels, aug_indices]
        return inputs, labels

    def train_classifier(self, train_loader, val_loader, test_loader):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()
        self.save_prompt(0)

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH
        total_data = len(train_loader)
        best_epoch = -1
        best_metric = 0
        log_interval = self.cfg.SOLVER.LOG_EVERY_N

        losses = AverageMeter('Loss', ':.4e')
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')

        self.cls_weights = train_loader.dataset.get_class_weights(
            self.cfg.DATA.CLASS_WEIGHTS_TYPE)
        logger.info(f"class weights: {self.cls_weights}")
        patience = 0  # if > self.cfg.SOLVER.PATIENCE, stop training

        #  Checks
        if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
            if self.cfg.NUM_GPUS > 1:
                if self.model.module.enc.transformer.prompt_config.NUM_TOKENS_PER_TYPE != -1:
                    assert self.model.module.enc.transformer.prompt_config.NUM_INVAR_TYPES == len(train_loader.dataset.transform_combinations)
            else:
                if self.model.enc.transformer.prompt_config.NUM_TOKENS_PER_TYPE != -1:
                    assert self.model.enc.transformer.prompt_config.NUM_INVAR_TYPES == len(train_loader.dataset.transform_combinations)

        for epoch in range(total_epoch):
            # reset averagemeters to measure per-epoch results
            losses.reset()
            batch_time.reset()
            data_time.reset()

            lr = self.scheduler.get_lr()[0]
            logger.info(
                "Training {} / {} epoch, with learning rate {}".format(
                    epoch + 1, total_epoch, lr
                )
            )

            # Enable training mode
            self.model.train()
            end = time.time()

            if self.cfg.MODEL.TRANSFER_TYPE == 'measure_invariance':
                #  Only eval for one epoch
                for k, p in self.model.module.head.named_parameters():
                    print("Freezing head parameters for invariance measurement", k)
                    p.requires_grad = False
                self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
                return
                    
            for idx, input_data in enumerate(train_loader):
                if self.cfg.DBG and idx == 20:
                    # if debugging, only need to see the first few iterations                    
                    break
                
                X, targets = self.get_input(input_data)
                # logger.info(X.shape)
                # logger.info(targets.shape)
                # measure data loading time
                data_time.update(time.time() - end)

                train_loss, _, _, _, _, _ = self.forward_one_batch(X, targets, True)

                if train_loss == -1:
                    # continue
                    return None

                if self.augmented:
                    losses.update(train_loss.item(), X[0].shape[0])
                else:
                    losses.update(train_loss.item(), X.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # log during one batch
                if (idx + 1) % log_interval == 0:
                    seconds_per_batch = batch_time.val
                    eta = datetime.timedelta(seconds=int(
                        seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                    logger.info(
                        "\tTraining {}/{}. train loss: {:.4f},".format(
                            idx + 1,
                            total_data,
                            train_loss
                        )
                        + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                            seconds_per_batch,
                            data_time.val,
                            str(eta),
                        )
                        + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                    )
            logger.info(
                "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                    data_time.avg, batch_time.avg)
                + "average train loss: {:.4f}".format(losses.avg))
             # update lr, scheduler.step() must be called after optimizer.step() according to the docs: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate  # noqa
            self.scheduler.step()

            # Enable eval mode
            self.model.eval()

            self.save_prompt(epoch + 1)

            # eval at each epoch for single gpu training
            self.evaluator.update_iteration(epoch)
            self.eval_classifier(val_loader, "val", epoch == total_epoch - 1)
            if test_loader is not None:
                self.eval_classifier(
                    test_loader, "test", epoch == total_epoch - 1)
            

            # check the patience
            # t_name = "val_" + val_loader.dataset.name
            # try:
            #     if self.cfg.DATA.MODE == 'classification':
            #         curr_acc = self.evaluator.results[f"epoch_{epoch}"]['classification'][t_name]["top1"]
            #     elif self.cfg.DATA.MODE == 'pose_estimation':
            #         print("Seeing results for pose estimation")
            #         curr_acc = self.evaluator.results[f"epoch_{epoch}"]['pose_estimation'][t_name]["pka"]
            # except KeyError:
            #     return

            # if curr_acc > best_metric:
            #     best_metric = curr_acc
            #     best_epoch = epoch + 1
            #     logger.info(
            #         f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
            #     patience = 0
            # else:
            #     patience += 1
            # if patience >= self.cfg.SOLVER.PATIENCE:
            #     logger.info("No improvement. Breaking out of loop.")
            #     break

        # save the last checkpoints
        # if self.cfg.MODEL.SAVE_CKPT:
        #     Checkpointer(
        #         self.model,
        #         save_dir=self.cfg.OUTPUT_DIR,
        #         save_to_disk=True
        #     ).save("last_model")

    @torch.no_grad()
    def save_prompt(self, epoch):
        # only save the prompt embed if below conditions are satisfied
        if self.cfg.MODEL.PROMPT.SAVE_FOR_EACH_EPOCH:
            if self.cfg.MODEL.TYPE == "vit" and "prompt" in self.cfg.MODEL.TRANSFER_TYPE:
                if self.cfg.NUM_GPUS > 1:
                    prompt_embds = self.model.module.enc.transformer.prompt_embeddings.cpu().numpy()
                else:
                    prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    if self.cfg.NUM_GPUS > 1:
                        deep_embds = self.model.module.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    else:
                        deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"invariance_prompt_imagenet_ep{epoch}.pth"))

    @torch.no_grad()
    def eval_classifier(self, data_loader, prefix, save=False):
        """evaluate classifier"""
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')

        log_interval = self.cfg.SOLVER.LOG_EVERY_N
        test_name = prefix + "_" + data_loader.dataset.name
        total = len(data_loader)

        # initialize features and target
        total_logits = []
        total_targets = []
        orig_cls_tokens, orig_features = [[] for _ in range(len(data_loader.dataset.transform_combinations))], [[] for _ in range(len(data_loader.dataset.transform_combinations))]
        aug_cls_tokens = [[] for _ in range(len(data_loader.dataset.transform_combinations))]
        aug_features = [[] for _ in range(len(data_loader.dataset.transform_combinations))]
        for idx, input_data in enumerate(data_loader):
            end = time.time()
            if idx == 20 and self.cfg.DBG:
                # if debugging, only need to see the first few iterations
                break
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X[0].shape))
            loss, orig_feats, aug_feats, orig_cls_tkn, aug_cls_tkn, aug_indices = self.forward_one_batch(X, targets, False)
            #  orig_cls_tkn, orig_feats: (bs, 768)
            #  aug_cls_tkn, aug_features: (bs, num_transforms, 768)

            if prefix == 'test':
                aug_cls_tkn = None
                aug_feats = None
                orig_cls_tkn = None
                orig_feats = None

            if loss == -1:
                return
            if self.augmented:
                losses.update(loss, X[0].shape[0])
            else:
                losses.update(loss, X.shape[0])

            # measure elapsed time
            batch_time.update(time.time() - end)

            if (idx + 1) % log_interval == 0:
                logger.info(
                    "\tTest {}/{}. loss: {:.3f}, {:.4f} s / batch. (data: {:.2e})".format(  # noqa
                        idx + 1,
                        total,
                        losses.val,
                        batch_time.val,
                        data_time.val
                    ) + "max mem: {:.5f} GB ".format(gpu_mem_usage())
                )

            # targets: List[int]
            # total_targets.extend(list(targets.numpy()))
            # total_logits.append(outputs)

            if aug_cls_tkn is not None:
                assert aug_features is not None # Check for sanity
                
                for k in range(len(aug_cls_tokens)):  # Iterate over all combinations of transforms
                    aug_cls_tokens[k].extend(aug_cls_tkn[:, k, :])
                    aug_features[k].extend(aug_feats[:, k, :])
                    orig_cls_tokens[k].extend(orig_cls_tkn[:, k, :])
                    orig_features[k].extend(orig_feats[:, k, :])

        if self.augmented and prefix == 'val':
            #  Measure invariances and save the the measurements
            features_sim = {}
            token_sim = {}
            iter = 0
            for k in range(0, len(data_loader.dataset.transform_combinations)):
                aug = data_loader.dataset.transform_combinations[k]
                #  Join all string elements in aug
                aug = "_".join(aug)
                print(aug)
                features_sim[aug] = self.measure_invariances(orig_features[iter], aug_features[iter]).item()
                token_sim[aug] = self.measure_invariances(orig_cls_tokens[iter], aug_cls_tokens[iter]).item()
                iter += 1

        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.module.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.module.side_alpha))

            
        if self.augmented and prefix == 'val':
            out = {"features_sim": features_sim, "token_sim": token_sim}
            print(out)
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_invariances.json")
            with open(out_path, "w") as f:
                json.dump(out, f)
            logger.info(
                f"Saved invariances for {test_name} at {out_path}")


    def measure_invariances(self, clean_feats, aug_feats):
        clean_feats = torch.stack(clean_feats)
        aug_feats = torch.stack(aug_feats)

        mean_feature = torch.mean(clean_feats, dim=0)
        #  Calculate covariance matrix, pytorch takes input matrix where rows are variables and columns are observations
        try:
            cov_matrix = torch.cov(clean_feats.T) + 1e-6 * torch.eye(clean_feats.shape[1]).to(self.device)
            inv_cov_matrix = torch.linalg.inv(cov_matrix)
            cholesky_matrix = torch.linalg.cholesky(inv_cov_matrix) 
        except:
            cov_matrix = torch.cov(clean_feats.T) + 1e-4 * torch.eye(clean_feats.shape[1]).to(self.device)
            inv_cov_matrix = torch.linalg.inv(cov_matrix)
            cholesky_matrix = torch.linalg.cholesky(inv_cov_matrix)
 

        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        a = (mean_feature - clean_feats) @ cholesky_matrix
        b = (mean_feature - aug_feats) @ cholesky_matrix

        sim = cosine_sim(a, b)
        return sim.mean()

        
        