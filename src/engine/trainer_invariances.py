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
        inputs = inputs.to(self.device, non_blocking=True)    # (batchsize, 2048)
        targets = targets.to(self.device, non_blocking=True)  # (batchsize, )

        if self.cfg.DBG:
            logger.info(f"shape of inputs: {inputs.shape}")
            logger.info(f"shape of targets: {targets.shape}")

        # forward
        with torch.set_grad_enabled(is_train):
            (_, _, orig_cls_token, orig_features), outputs = self.model(inputs, vis = True, return_feature=True, get_logits=True)
            
            if self.cfg.DBG:
                logger.info(
                    "shape of model output: {}, targets: {}".format(
                        orig_features.shape, targets.shape))

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            else:
                loss = self.cls_criterion(
                    outputs, targets, self.cls_weights)

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

        if self.augmented:
            with torch.no_grad():
                aug_cls_tokens = {}
                aug_features = {}
                for aug_index in range(aug_inputs.shape[1]):
                    aug_im = aug_inputs[:, aug_index, :, :, :]
                    (_, _, aug_cls_token, aug_feat), _ = self.model(aug_im, vis = True, return_feature=True, get_logits=True)
                    aug_cls_tokens[aug_index] = aug_cls_token
                    aug_features[aug_index] = aug_feat
            
                return loss, outputs, aug_cls_tokens, aug_features, orig_cls_token, orig_features
        
        return loss, outputs, None, None, None, None

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        if self.augmented:
            aug_inputs = data["augmented"].float()
            inputs = [inputs, aug_inputs]
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
            t_name = "val_" + val_loader.dataset.name
            try:
                if self.cfg.DATA.MODE == 'classification':
                    curr_acc = self.evaluator.results[f"epoch_{epoch}"]['classification'][t_name]["top1"]
                elif self.cfg.DATA.MODE == 'pose_estimation':
                    print("Seeing results for pose estimation")
                    curr_acc = self.evaluator.results[f"epoch_{epoch}"]['pose_estimation'][t_name]["pka"]
            except KeyError:
                return

            if curr_acc > best_metric:
                best_metric = curr_acc
                best_epoch = epoch + 1
                logger.info(
                    f'Best epoch {best_epoch}: best metric: {best_metric:.3f}')
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg.SOLVER.PATIENCE:
                logger.info("No improvement. Breaking out of loop.")
                break

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
                prompt_embds = self.model.enc.transformer.prompt_embeddings.cpu().numpy()
                out = {"shallow_prompt": prompt_embds}
                if self.cfg.MODEL.PROMPT.DEEP:
                    deep_embds = self.model.enc.transformer.deep_prompt_embeddings.cpu().numpy()
                    out["deep_prompt"] = deep_embds
                torch.save(out, os.path.join(
                    self.cfg.OUTPUT_DIR, f"prompt_ep{epoch}.pth"))

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
        orig_cls_tokens, orig_features = [], []
        aug_cls_tokens = [[] for _ in range(len(data_loader.dataset.transform_dict.keys()))]
        aug_features = [[] for _ in range(len(data_loader.dataset.transform_dict.keys()))]
        for idx, input_data in enumerate(data_loader):
            end = time.time()
            X, targets = self.get_input(input_data)
            # measure data loading time
            data_time.update(time.time() - end)

            if self.cfg.DBG:
                logger.info("during eval: {}".format(X.shape))
            loss, outputs, aug_cls_tkn, aug_feats, orig_cls_tkn, orig_feats = self.forward_one_batch(X, targets, False)
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
            total_targets.extend(list(targets.numpy()))
            total_logits.append(outputs)

            if aug_cls_tkn is not None:
                assert aug_features is not None # Check for sanity
                orig_cls_tokens.extend(orig_cls_tkn)
                orig_features.extend(orig_feats)
                for k in range(len(aug_cls_tokens)):
                    aug_cls_tokens[k].extend(aug_cls_tkn[k])
                    aug_features[k].extend(aug_feats[k])

        if self.augmented and prefix == 'val':
            #  Measure invariances and save the the measurements
            features_sim = {}
            token_sim = {}
            iter = 0
            for k in data_loader.dataset.transform_dict.keys():
                features_sim[k] = self.measure_invariances(orig_features, aug_features[iter]).item()
                token_sim[k] = self.measure_invariances(orig_cls_tokens, aug_cls_tokens[iter]).item()
                iter += 1

        logger.info(
            f"Inference ({prefix}):"
            + "avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                data_time.avg, batch_time.avg)
            + "average loss: {:.4f}".format(losses.avg))
        if self.model.side is not None:
            logger.info(
                "--> side tuning alpha = {:.4f}".format(self.model.side_alpha))
        # total_testimages x num_classes
        joint_logits = torch.cat(total_logits, dim=0).cpu().numpy()
        if self.mode == 'classification':
            self.evaluator.classify(
                joint_logits, total_targets,
                test_name, self.cfg.DATA.MULTILABEL,
            )
        elif self.mode == 'pose_estimation':
            total_targets = np.array(total_targets)
            self.evaluator.regress(
                joint_logits, total_targets, test_name
            )

        # save the probs and targets
        if save and self.cfg.MODEL.SAVE_CKPT:
            out = {"targets": total_targets, "joint_logits": joint_logits}
            out_path = os.path.join(
                self.cfg.OUTPUT_DIR, f"{test_name}_logits.pth")
            torch.save(out, out_path)
            logger.info(
                f"Saved logits and targets for {test_name} at {out_path}")
            
        if self.augmented and prefix == 'val':
            out = {"features_sim": features_sim, "token_sim": token_sim}
            if data_loader.dataset.predict_rotation:
                 out_path = os.path.join(
                    self.cfg.OUTPUT_DIR, f"{test_name}_rot_invariances.json")
            else:
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

        
        