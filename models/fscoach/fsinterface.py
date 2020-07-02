#coding: utf-8
"""
    @author: cjliux@gmail.com
    borrowed heavily from https://github.com/zliucr/coach
"""
import os
import sys
import tqdm
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import logging
import pickle
logger = logging.getLogger()
from collections import defaultdict

from .conll2002_metrics import *
from .data import (get_dataloader_for_fs_train, 
                    get_dataloader_for_fs_eval, get_dataloader_for_fs_test)
from .model import BinarySLUTagger, SlotNamePredictor, SentRepreGenerator, IntentPredictor
from utils.vocab import Vocab
from utils.scaffold import get_output_dir, init_logger

torch.nn.Module.dump_patches = True


class SLUTrainer(object):
    def __init__(self, params, 
            binary_slu_tagger, slotname_predictor, intent_predictor,
            sent_repre_generator=None, optimizer=None):
        self.params = params
        os.makedirs(self.params.dump_path, exist_ok=True)

        self.binary_slu_tagger = binary_slu_tagger
        self.slotname_predictor = slotname_predictor
        self.intent_predictor = intent_predictor

        self.lr = params.lr
        self.use_label_encoder = params.tr
        self.num_domain = params.num_domain

        # read vocab
        self.domain_set = Vocab.from_file(os.path.join(params.data_path, "domains.txt"))
        self.intent_set = Vocab.from_file(os.path.join(params.data_path, "intents.txt"))
        # self.slot_list = Vocab.from_file(os.path.join(params["data_path"], "slots.txt"))
        self.vocab = Vocab.from_file(os.path.join(params.data_path, "token_vocab.txt"))
        self.y2_set = Vocab.from_file(os.path.join(params.data_path, "label_vocab.txt"))
        self.y1_set = Vocab.from_file(os.path.join(params.data_path, "bin_label_vocab.txt"))
        with open(os.path.join(params.data_path, "dom2slots.json"), 'r', encoding='utf8') as fd:
            self.domain2slot = json.load(fd)

        # opt
        if self.use_label_encoder:
            self.sent_repre_generator = sent_repre_generator
            self.loss_fn_mse = nn.MSELoss()
            model_parameters = [
                {"params": self.binary_slu_tagger.parameters()},
                {"params": self.slotname_predictor.parameters()},
                {"params": self.sent_repre_generator.parameters()},
                {"params": self.intent_predictor.parameters()},
            ]
        else:
            model_parameters = [
                {"params": self.binary_slu_tagger.parameters()},
                {"params": self.slotname_predictor.parameters()},
                {"params": self.intent_predictor.parameters()},
            ]
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr)
        if optimizer is not None:
            self.optimizer = optimizer

        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0

        self.stop_training_flag = False
    
    def train_step(self, batch, epoch=None):
        model_input = batch["model_input"]
        X, lengths = model_input["padded_seqs"].cuda(), model_input["lengths"].cuda()
        y_bin, y_final, y_dm = batch["y1"], batch["y2"], model_input["domains"]
        y_int = model_input["intents"].cuda()
        if self.params.tr:
            templates, tem_lengths = model_input["padded_templates"].cuda(), model_input["tem_lengths"].cuda()

        self.binary_slu_tagger.train()
        self.slotname_predictor.train()
        if self.use_label_encoder:
            self.sent_repre_generator.train()
        
        batch_size = X.size(0)
        meta_loss = 0

        slutagger_snapshot = copy.deepcopy(self.binary_slu_tagger.state_dict())
        slnmpred_snapshot = copy.deepcopy(self.slotname_predictor.state_dict())
        if self.params.tr:
            sreprgen_snapshot = copy.deepcopy(self.sent_repre_generator.state_dict())
        optim_snapshot = copy.deepcopy(self.optimizer.state_dict())

        for i_sam in range(batch_size):
            # support set
            sup_batch = batch["support"][i_sam]
            sup_input = sup_batch["model_input"]
            sup_X, sup_lens = sup_input["padded_seqs"].cuda(), sup_input["lengths"].cuda()
            sup_ybin, sup_yfin, sup_ydm = sup_batch["y1"], sup_batch["y2"], sup_input["domains"]
            sup_int = sup_input["intents"].cuda()
            if self.params.tr:
                sup_tem, sup_tlens = sup_input["padded_templates"].cuda(), sup_input["tem_lengths"].cuda()

            bin_preds, lstm_hiddens = self.binary_slu_tagger(sup_X)

            logits_int = self.intent_predictor(lstm_hiddens, sup_lens)
            loss_int = self.loss_fn(logits_int, sup_int)
            self.optimizer.zero_grad()
            loss_int.backward(retain_graph=True)
            self.optimizer.step()

            ## optimize binary_slu_tagger
            loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, sup_lens, sup_ybin)
            self.optimizer.zero_grad()
            loss_bin.backward(retain_graph=True)
            self.optimizer.step()

            ## optimize slotname_predictor
            pred_slotname_list, gold_slotname_list = self.slotname_predictor(
                sup_ydm, lstm_hiddens, binary_golds=sup_ybin, final_golds=sup_yfin)

            for j_sam, (pred_slotname_each_sample, gold_slotname_each_sample) in enumerate(zip(pred_slotname_list, gold_slotname_list)):
                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                if pred_slotname_each_sample.size(0) == 0: continue
                loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                self.optimizer.zero_grad()
                if j_sam == sup_X.size(0) - 1 and not self.use_label_encoder:
                    loss_slotname.backward()
                else:
                    loss_slotname.backward(retain_graph=True)
                self.optimizer.step()

            if self.use_label_encoder:
                templates_repre, input_repre = self.sent_repre_generator(
                    sup_tem, sup_tlens, lstm_hiddens, sup_lens)

                input_repre = input_repre.detach()
                template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
                template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
                template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
                input_repre.requires_grad = True

                self.optimizer.zero_grad()
                template0_loss.backward(retain_graph=True)
                template1_loss.backward(retain_graph=True)
                if epoch <= 3:
                    template2_loss.backward()
                else:
                    template2_loss.backward(retain_graph=True)
                self.optimizer.step()

                if epoch > 3:
                    templates_repre = templates_repre.detach()
                    input_loss0 = self.loss_fn_mse(input_repre, templates_repre[:, 0, :])
                    input_loss1 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 1, :])
                    input_loss2 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 2, :])
                    templates_repre.requires_grad = True

                    self.optimizer.zero_grad()
                    input_loss0.backward(retain_graph=True)
                    input_loss1.backward(retain_graph=True)
                    # input_loss2.backward(retain_graph=True)
                    input_loss2.backward()
                    self.optimizer.step()
            
            ## qry
            X_sam, len_sam = X[i_sam:i_sam+1], lengths[i_sam:i_sam+1]
            ybin_sam, yfin_sam, ydm_sam = y_bin[i_sam:i_sam+1], y_final[i_sam:i_sam+1], y_dm[i_sam:i_sam+1]
            yint_sam = y_int[i_sam:i_sam+1]
            if self.params.tr:
                tem_sam, tlen_sam = templates[i_sam:i_sam+1], tem_lengths[i_sam:i_sam+1]

            bin_preds, lstm_hiddens = self.binary_slu_tagger(X_sam)

            logits_int = self.intent_predictor(lstm_hiddens, len_sam)
            loss_int = self.loss_fn(logits_int, yint_sam)
            meta_loss += loss_int

            loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, len_sam, ybin_sam)
            meta_loss += loss_bin
            # self.optimizer.zero_grad()
            # loss_bin.backward(retain_graph=True)
            # self.optimizer.step()

            pred_slotname_list, gold_slotname_list = self.slotname_predictor(
                ydm_sam, lstm_hiddens, binary_golds=ybin_sam, final_golds=yfin_sam)

            for pred_slotname_each_sample, gold_slotname_each_sample in zip(pred_slotname_list, gold_slotname_list):
                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                if pred_slotname_each_sample.size(0) == 0: continue
                loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                meta_loss += loss_slotname
                # self.optimizer.zero_grad()
                # loss_slotname.backward(retain_graph=True)
                # self.optimizer.step()

            if self.use_label_encoder:
                templates_repre, input_repre = self.sent_repre_generator(
                    tem_sam, tlen_sam, lstm_hiddens, len_sam)

                input_repre = input_repre.detach()
                template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
                template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
                template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
                input_repre.requires_grad = True

                meta_loss += template0_loss + template1_loss + template2_loss
                # self.optimizer.zero_grad()
                # template0_loss.backward(retain_graph=True)
                # template1_loss.backward(retain_graph=True)
                # template2_loss.backward(retain_graph=True)
                # self.optimizer.step()

                if epoch > 3:
                    templates_repre = templates_repre.detach()
                    input_loss0 = self.loss_fn_mse(input_repre, templates_repre[:, 0, :])
                    input_loss1 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 1, :])
                    input_loss2 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 2, :])
                    templates_repre.requires_grad = True

                    meta_loss += input_loss0 + input_loss1 + input_loss2
                    # self.optimizer.zero_grad()
                    # input_loss0.backward(retain_graph=True)
                    # input_loss1.backward(retain_graph=True)
                    # input_loss2.backward(retain_graph=True)
                    # self.optimizer.step()

            self.binary_slu_tagger.load_state_dict(slutagger_snapshot)
            self.slotname_predictor.load_state_dict(slnmpred_snapshot)
            if self.use_label_encoder:
                self.sent_repre_generator.load_state_dict(sreprgen_snapshot)
            self.optimizer.load_state_dict(optim_snapshot)

        meta_loss = meta_loss / batch_size
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()

        if self.use_label_encoder:
            return loss_bin.item(), loss_slotname.item(), template0_loss.item(), template1_loss.item()
        else:
            return loss_bin.item(), loss_slotname.item()
    
    def evaluate(self, dataloader, istestset=False):
        binary_preds, binary_golds = [], []
        final_preds, final_golds = [], []
        intent_preds = []

        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:
            model_input = batch["model_input"]
            X, lengths = model_input["padded_seqs"].cuda(), model_input["lengths"].cuda()
            y_bin, y_final, y_dm = batch["y1"], batch["y2"], model_input["domains"]
            y_int = model_input["intents"].cuda()
            # if self.params.tr:
            #     templates, tem_lengths = model_input["padded_templates"].cuda(), model_input["tem_lengths"].cuda()

            binary_golds.extend(y_bin)
            final_golds.extend(y_final)

            batch_size = X.size(0)
            slutagger_snapshot = copy.deepcopy(self.binary_slu_tagger.state_dict())
            slnmpred_snapshot = copy.deepcopy(self.slotname_predictor.state_dict())
            if self.params.tr:
                sreprgen_snapshot = copy.deepcopy(self.sent_repre_generator.state_dict())
            optim_snapshot = copy.deepcopy(self.optimizer.state_dict())

            for i_sam in range(batch_size):
                self.binary_slu_tagger.train()
                self.slotname_predictor.train()

                # support set
                sup_batch = batch["support"][i_sam]
                sup_input = sup_batch["model_input"]
                sup_X, sup_lens = sup_input["padded_seqs"].cuda(), sup_input["lengths"].cuda()
                sup_ybin, sup_yfin, sup_ydm = sup_batch["y1"], sup_batch["y2"], sup_input["domains"]
                sup_int = sup_input["intents"].cuda()
                if self.params.tr:
                    sup_tem, sup_tlens = sup_input["padded_templates"].cuda(), sup_input["tem_lengths"].cuda()

                bin_preds, lstm_hiddens = self.binary_slu_tagger(sup_X)

                logits_int = self.intent_predictor(lstm_hiddens, sup_lens)
                loss_int = self.loss_fn(logits_int, sup_int)
                self.optimizer.zero_grad()
                loss_int.backward(retain_graph=True)
                self.optimizer.step()

                ## optimize binary_slu_tagger
                loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, sup_lens, sup_ybin)
                self.optimizer.zero_grad()
                loss_bin.backward(retain_graph=True)
                self.optimizer.step()

                ## optimize slotname_predictor
                pred_slotname_list, gold_slotname_list = self.slotname_predictor(
                    sup_ydm, lstm_hiddens, binary_golds=sup_ybin, final_golds=sup_yfin)

                for j_sam, (pred_slotname_each_sample, gold_slotname_each_sample) in enumerate(zip(pred_slotname_list, gold_slotname_list)):
                    assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                    if pred_slotname_each_sample.size(0) == 0: continue
                    loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                    self.optimizer.zero_grad()
                    if j_sam == sup_X.size(0) - 1 and not self.use_label_encoder:
                        loss_slotname.backward()
                    else:
                        loss_slotname.backward(retain_graph=True)
                    self.optimizer.step()

                if self.use_label_encoder:
                    templates_repre, input_repre = self.sent_repre_generator(
                        sup_tem, sup_tlens, lstm_hiddens, sup_lens)

                    input_repre = input_repre.detach()
                    template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
                    template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
                    template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
                    input_repre.requires_grad = True

                    self.optimizer.zero_grad()
                    template0_loss.backward(retain_graph=True)
                    template1_loss.backward(retain_graph=True)
                    template2_loss.backward()
                    self.optimizer.step()

                # qry
                self.binary_slu_tagger.eval()
                self.slotname_predictor.eval()
                with torch.no_grad():
                    X_sam, len_sam = X[i_sam:i_sam+1], lengths[i_sam:i_sam+1]
                    ybin_sam, yfin_sam, ydm_sam = y_bin[i_sam:i_sam+1], y_final[i_sam:i_sam+1], y_dm[i_sam:i_sam+1]
                    # if self.params.tr:
                    #     tem_sam, tlen_sam = templates[i_sam:i_sam+1], tem_lengths[i_sam:i_sam+1]

                    bin_preds_batch, lstm_hiddens = self.binary_slu_tagger(X_sam)

                    pred_int = self.intent_predictor.predict(ydm_sam, lstm_hiddens, len_sam)
                    intent_preds.extend(pred_int)

                    bin_preds_batch = self.binary_slu_tagger.crf_decode(bin_preds_batch, len_sam)
                    binary_preds.extend(bin_preds_batch)

                    slotname_preds_batch = self.slotname_predictor(
                        ydm_sam, lstm_hiddens, binary_preditions=bin_preds_batch, 
                        binary_golds=None, final_golds=None)
                    
                    final_preds_batch = self.combine_binary_and_slotname_preds(
                        ydm_sam, bin_preds_batch, slotname_preds_batch)
                    final_preds.extend(final_preds_batch)
                
                self.binary_slu_tagger.load_state_dict(slutagger_snapshot)
                self.slotname_predictor.load_state_dict(slnmpred_snapshot)
                if self.use_label_encoder:
                    self.sent_repre_generator.load_state_dict(sreprgen_snapshot)
                self.optimizer.load_state_dict(optim_snapshot)

        # binary predictions
        binary_preds = np.concatenate(binary_preds, axis=0)
        binary_preds = list(binary_preds)
        binary_golds = np.concatenate(binary_golds, axis=0)
        binary_golds = list(binary_golds)

        # final predictions
        final_preds = np.concatenate(final_preds, axis=0)
        final_preds = list(final_preds)
        final_golds = np.concatenate(final_golds, axis=0)
        final_golds = list(final_golds)

        bin_lines, final_lines = [], []
        for bin_pred, bin_gold, final_pred, final_gold in zip(binary_preds, binary_golds, final_preds, final_golds):
            bin_slot_pred = self.y1_set[bin_pred]
            bin_slot_gold = self.y1_set[bin_gold]
            
            final_slot_pred = self.y2_set[final_pred]
            final_slot_gold = self.y2_set[final_gold]
            
            bin_lines.append("w" + " " + bin_slot_pred + " " + bin_slot_gold)
            final_lines.append("w" + " " + final_slot_pred + " " + final_slot_gold)
            
        bin_result = conll2002_measure(bin_lines)
        bin_f1 = bin_result["fb1"]
        
        final_result = conll2002_measure(final_lines)
        final_f1 = final_result["fb1"]
        
        if istestset == False:  # dev set
            if final_f1 > self.best_f1:
                self.best_f1 = final_f1
                self.no_improvement_num = 0
                logger.info("Found better model!!")
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
            
            if self.no_improvement_num >= self.early_stop:
                self.stop_training_flag = True
        
        return bin_f1, final_f1, self.stop_training_flag
        
    def combine_binary_and_slotname_preds(self, dm_id_batch, binary_preds_batch, slotname_preds_batch):
        """
        Input:
            dm_id_batch: (bsz)
            binary_preds: (bsz, seq_len)
            slotname_preds: (bsz, num_slotname, slot_num)
        Output:
            final_preds: (bsz, seq_len)
        """
        final_preds = []
        for i in range(len(dm_id_batch)):
            dm_id = dm_id_batch[i]
            binary_preds = binary_preds_batch[i]
            slotname_preds = slotname_preds_batch[i]
            slot_list_based_dm = self.domain2slot[self.domain_set[dm_id]]
            
            i = -1
            final_preds_each = []
            for bin_pred in binary_preds:
                # values of bin_pred are 0 (O), or 1(B) or 2(I)
                if bin_pred.item() == 0:
                    final_preds_each.append(0)
                elif bin_pred.item() == 1:
                    i += 1
                    pred_slot_id = torch.argmax(slotname_preds[i])
                    slotname = "B-" + slot_list_based_dm[pred_slot_id]
                    final_preds_each.append(self.y2_set.word2index[slotname])
                elif bin_pred.item() == 2:
                    if i == -1:
                        final_preds_each.append(0)
                    else:
                        pred_slot_id = torch.argmax(slotname_preds[i])
                        slotname = "I-" + slot_list_based_dm[pred_slot_id]
                        if slotname not in self.y2_set:
                            final_preds_each.append(0)
                        else:
                            final_preds_each.append(self.y2_set.word2index[slotname])
                
            assert len(final_preds_each) == len(binary_preds)
            final_preds.append(final_preds_each)

        return final_preds
    
    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        torch.save({
                "binary_slu_tagger": self.binary_slu_tagger.state_dict(),
                "slotname_predictor": self.slotname_predictor.state_dict(),
                "intent_predictor": self.intent_predictor.state_dict(),
                "sent_repre_generator": self.sent_repre_generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)

    def predict(self, dataloader):
        self.binary_slu_tagger.eval()
        self.slotname_predictor.eval()

        # binary_preds, binary_golds = [], []
        # final_preds, final_golds = [], []
        # final_slvals = []
        final_items = defaultdict(list)

        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:
            model_input = batch["model_input"]
            X, lengths = model_input["padded_seqs"].cuda(), model_input["lengths"].cuda()
            y_bin, y_final, y_dm = batch["y1"], batch["y2"], model_input["domains"]
            # y_int = model_input["intents"]

            texts, tokens, domains, ids = batch["text"], batch["token"], batch["domain"], batch["id"]

            batch_size = X.size(0)
            slutagger_snapshot = copy.deepcopy(self.binary_slu_tagger.state_dict())
            slnmpred_snapshot = copy.deepcopy(self.slotname_predictor.state_dict())
            if self.params.tr:
                sreprgen_snapshot = copy.deepcopy(self.sent_repre_generator.state_dict())
            optim_snapshot = copy.deepcopy(self.optimizer.state_dict())

            for i_sam in range(batch_size):
                self.binary_slu_tagger.train()
                self.slotname_predictor.train()

                # support set
                sup_batch = batch["support"][i_sam]
                sup_input = sup_batch["model_input"]
                sup_X, sup_lens = sup_input["padded_seqs"].cuda(), sup_input["lengths"].cuda()
                sup_ybin, sup_yfin, sup_ydm = sup_batch["y1"], sup_batch["y2"], sup_input["domains"]
                sup_int = sup_input["intents"].cuda()
                if self.params.tr:
                    sup_tem, sup_tlens = sup_input["padded_templates"].cuda(), sup_input["tem_lengths"].cuda()

                bin_preds, lstm_hiddens = self.binary_slu_tagger(sup_X)

                logits_int = self.intent_predictor(lstm_hiddens, sup_lens)
                loss_int = self.loss_fn(logits_int, sup_int)
                self.optimizer.zero_grad()
                loss_int.backward(retain_graph=True)
                self.optimizer.step()

                ## optimize binary_slu_tagger
                loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, sup_lens, sup_ybin)
                self.optimizer.zero_grad()
                loss_bin.backward(retain_graph=True)
                self.optimizer.step()

                ## optimize slotname_predictor
                pred_slotname_list, gold_slotname_list = self.slotname_predictor(
                    sup_ydm, lstm_hiddens, binary_golds=sup_ybin, final_golds=sup_yfin)

                for j_sam, (pred_slotname_each_sample, gold_slotname_each_sample) in enumerate(zip(pred_slotname_list, gold_slotname_list)):
                    assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                    if pred_slotname_each_sample.size(0) == 0: continue
                    loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                    self.optimizer.zero_grad()
                    if j_sam == sup_X.size(0) - 1 and not self.use_label_encoder:
                        loss_slotname.backward()
                    else:
                        loss_slotname.backward(retain_graph=True)
                    self.optimizer.step()

                if self.use_label_encoder:
                    templates_repre, input_repre = self.sent_repre_generator(
                        sup_tem, sup_tlens, lstm_hiddens, sup_lens)

                    input_repre = input_repre.detach()
                    template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
                    template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
                    template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
                    input_repre.requires_grad = True

                    self.optimizer.zero_grad()
                    template0_loss.backward(retain_graph=True)
                    template1_loss.backward(retain_graph=True)
                    template2_loss.backward()
                    self.optimizer.step()

                # qry
                self.binary_slu_tagger.eval()
                self.slotname_predictor.eval()
                with torch.no_grad():
                    X_sam, len_sam = X[i_sam:i_sam+1], lengths[i_sam:i_sam+1]
                    ybin_sam, yfin_sam, ydm_sam = y_bin[i_sam:i_sam+1], y_final[i_sam:i_sam+1], y_dm[i_sam:i_sam+1]
                    # if self.params.tr:
                    #     tem_sam, tlen_sam = templates[i_sam:i_sam+1], tem_lengths[i_sam:i_sam+1]

                    bin_preds_batch, lstm_hiddens = self.binary_slu_tagger(X_sam)

                    pred_ints = self.intent_predictor.predict(ydm_sam, lstm_hiddens, len_sam)

                    bin_preds_batch = self.binary_slu_tagger.crf_decode(bin_preds_batch, len_sam)
                    # binary_preds.extend(bin_preds_batch)

                    slotname_preds_batch = self.slotname_predictor(
                        ydm_sam, lstm_hiddens, binary_preditions=bin_preds_batch, 
                        binary_golds=None, final_golds=None)
                    
                    final_preds_batch = self.combine_binary_and_slotname_preds(
                        ydm_sam, bin_preds_batch, slotname_preds_batch)
                    # final_preds.extend(final_preds_batch)

                    x, l_x, fp = X_sam[0], len_sam[0], final_preds_batch[0]
                    i_x = pred_ints[0]

                    labels = [self.y2_set.index2word[i] for i in fp[:l_x]]
                    ents = self.collect_named_entities(labels)

                    slvals = {}
                    for etype, start, end in ents:
                        val = ''.join(tokens[i_sam][start:end+1])
                        if etype not in slvals:
                            slvals[etype] = val
                        elif isinstance(slvals[etype], str):
                            slvals[etype] = [slvals[etype], val]
                        else:
                            slvals[etype].append(val)

                    # final_slvals.append(slvals)
                    item = {}
                    item['domain'] = domains[i_sam]
                    item['text'] = texts[i_sam]
                    item['id'] = ids[i_sam]
                    item['slots'] = slvals
                    item['intent'] = i_x
                    final_items[item['domain']].append(item)

                self.binary_slu_tagger.load_state_dict(slutagger_snapshot)
                self.slotname_predictor.load_state_dict(slnmpred_snapshot)
                if self.use_label_encoder:
                    self.sent_repre_generator.load_state_dict(sreprgen_snapshot)
                self.optimizer.load_state_dict(optim_snapshot)
                
        return final_items

    def collect_named_entities(self, labels):
        named_entities = []
        start_offset, end_offset, ent_type = None, None, None
         
        for offset, token_tag in enumerate(labels):
            if token_tag == 'O':
                if ent_type is not None and start_offset is not None:
                    end_offset = offset - 1
                    named_entities.append((ent_type, start_offset, end_offset))
                    start_offset = None
                    end_offset = None
                    ent_type = None
            elif ent_type is None:
                ent_type = token_tag[2:]
                start_offset = offset
            elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):
                end_offset = offset - 1
                named_entities.append((ent_type, start_offset, end_offset))

                # start of a new entity
                ent_type = token_tag[2:]
                start_offset = offset
                end_offset = None

        # catches an entity that goes up until the last token
        if ent_type and start_offset and end_offset is None:
            named_entities.append((ent_type, start_offset, len(labels)-1))
        return named_entities


def train_model(params):
    params.dump_path = get_output_dir(params.dump_path, params.exp_name, params.exp_id)
    os.makedirs(params.dump_path, exist_ok=True)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))
    init_logger(os.path.join(params.dump_path, params.logger_filename))

    vocab = Vocab.from_file(os.path.join(params.data_path, "token_vocab.txt"))

    dataloader_tr = get_dataloader_for_fs_train(
        params.data_path, params.raw_data_path, params.evl_dm.split(','), 
        params.batch_size, params.tr, params.n_samples)
    dataloader_val = get_dataloader_for_fs_eval(
        params.data_path, params.raw_data_path, params.evl_dm.split(','), 
        params.batch_size, params.tr, params.n_samples)

    binary_slutagger = BinarySLUTagger(params, vocab)
    slotname_predictor = SlotNamePredictor(params)
    binary_slutagger, slotname_predictor = binary_slutagger.cuda(), slotname_predictor.cuda()
    intent_predictor = IntentPredictor(params).cuda()

    if params.tr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()

    if params.tr:
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor, 
            intent_predictor, sent_repre_generator=sent_repre_generator)
    else:
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor, intent_predictor)
    
    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_bin_list, loss_slotname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        pbar = tqdm.tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        if params.tr:
            for i, batch in pbar:
                # X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()
                loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(batch, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                loss_tem0_list.append(loss_tem0)
                loss_tem1_list.append(loss_tem1)

                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            for i, batch in pbar:
                loss_bin, loss_slotname = slu_trainer.train_step(batch)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))
            
        if params.tr:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format(
                (e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format(
                (e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_val, istestset=False)
        logger.info("Eval on dev set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        # bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_test, istestset=True)
        # logger.info("Eval on test set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        if stop_training_flag == True:
            break


def eval_model(params):
    # get dataloader
    dataloader_val = get_dataloader_for_fs_eval(
        params.data_path, params.raw_data_path, params.evl_dm.split(','), 
        params.batch_size, params.tr, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)
    
    vocab = Vocab.from_file(os.path.join(params.data_path, "token_vocab.txt"))
    binary_slu_tagger = BinarySLUTagger(params, vocab)
    slotname_predictor = SlotNamePredictor(params)
    binary_slu_tagger, slotname_predictor = binary_slu_tagger.cuda(), slotname_predictor.cuda()
    intent_predictor = IntentPredictor(params).cuda()
    sent_repre_generator = None
    if params.tr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()
        model_parameters = [
            {"params": binary_slu_tagger.parameters()},
            {"params": slotname_predictor.parameters()},
            {"params": sent_repre_generator.parameters()},
            {"params": intent_predictor.parameters()},
        ]
    else:
        model_parameters = [
            {"params": binary_slu_tagger.parameters()},
            {"params": slotname_predictor.parameters()},
            {"params": intent_predictor.parameters()},
        ]
    # Adam optimizer
    optimizer = torch.optim.Adam(model_parameters, params.lr)

    reloaded = torch.load(model_path)
    binary_slu_tagger.load_state_dict(reloaded["binary_slu_tagger"])
    slotname_predictor.load_state_dict(reloaded["slotname_predictor"])
    optimizer.load_state_dict(reloaded["optimizer"])
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()
    intent_predictor.load_state_dict(reloaded["intent_predictor"])
    if params.tr:
        sent_repre_generator.load_state_dict(reloaded["sent_repre_generator"])

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor, 
        intent_predictor, sent_repre_generator, optimizer=optimizer)

    bin_f1, fin_f1, _ = slu_trainer.evaluate(dataloader_val, istestset=False)
    print("Eval on dev set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, fin_f1))

    # _, f1_score, _ = slu_trainer.evaluate(dataloader_test, istestset=True)
    # print("Eval on test set. Final Slot F1 Score: {:.4f}.".format(f1_score))
    return bin_f1, fin_f1


def test_model(params):
    dataloader_test = get_dataloader_for_fs_test(
        params.data_path, params.raw_data_path, 
        params.batch_size, params.tr, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)
    save_dir = os.path.dirname(model_path)
    
    vocab = Vocab.from_file(os.path.join(params.data_path, "token_vocab.txt"))
    binary_slu_tagger = BinarySLUTagger(params, vocab)
    slotname_predictor = SlotNamePredictor(params)
    binary_slu_tagger, slotname_predictor = binary_slu_tagger.cuda(), slotname_predictor.cuda()
    intent_predictor = IntentPredictor(params).cuda()
    sent_repre_generator = None
    if params.tr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()
        model_parameters = [
            {"params": binary_slu_tagger.parameters()},
            {"params": slotname_predictor.parameters()},
            {"params": sent_repre_generator.parameters()},
            {"params": intent_predictor.parameters()},
        ]
    else:
        model_parameters = [
            {"params": binary_slu_tagger.parameters()},
            {"params": slotname_predictor.parameters()},
            {"params": intent_predictor.parameters()},
        ]
    # Adam optimizer
    optimizer = torch.optim.Adam(model_parameters, params.lr)

    reloaded = torch.load(model_path)
    binary_slu_tagger.load_state_dict(reloaded["binary_slu_tagger"])
    slotname_predictor.load_state_dict(reloaded["slotname_predictor"])
    optimizer.load_state_dict(reloaded["optimizer"])
    intent_predictor.load_state_dict(reloaded["intent_predictor"])
    if params.tr:
        sent_repre_generator.load_state_dict(reloaded["sent_repre_generator"])

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor, 
        intent_predictor, sent_repre_generator, optimizer=optimizer)
    # final_slvals = slu_trainer.predict(dataloader_test)
    final_items = slu_trainer.predict(dataloader_test)

    # dom_items = defaultdict(list)
    # for i_sam in range(len(final_slvals)):
    #     item = {}
    #     item['domain'] = dev_qry_data['raw_domains'][i_sam]
    #     item['text'] = dev_qry_data['raw_text'][i_sam]
    #     item['slots'] = final_slvals[i_sam]
    #     dom_items[item['domain']].append(item)
    
    for dom, pred in final_items.items():
        with open(os.path.join(save_dir, "predict_{}.json".format(dom)), 'w', encoding='utf8') as fd:
            json.dump(pred, fd, ensure_ascii=False, indent=2)


if __name__=='__main__':
    pass

