#coding: utf-8
"""
    @author: cjliux@gmail.com
    borrowed heavily from https://github.com/zliucr/coach
"""
import os
import sys
import tqdm
import json
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
from .model import BinarySLUTagger, SlotNamePredictor, SentRepreGenerator
from utils.vocab import Vocab
from utils.scaffold import get_output_dir, init_logger


class SLUTrainer(object):
    def __init__(self, params, 
            binary_slu_tagger, slotname_predictor, sent_repre_generator=None):
        self.params = params
        os.makedirs(self.params.dump_path, exist_ok=True)

        self.binary_slu_tagger = binary_slu_tagger
        self.slotname_predictor = slotname_predictor

        self.lr = params.lr
        self.use_label_encoder = params.tr
        self.num_domain = params.num_domain

        # read vocab
        self.domain_set = Vocab.from_file(os.path.join(params.data_path, "domains.txt"))
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
                {"params": self.sent_repre_generator.parameters()}
            ]
        else:
            model_parameters = [
                {"params": self.binary_slu_tagger.parameters()},
                {"params": self.slotname_predictor.parameters()}
            ]
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_f1 = 0

        self.stop_training_flag = False
    
    def train_step(self, batch, epoch=None):
        model_input = batch["model_input"]
        X, lengths = model_input["padded_seqs"].cuda(), model_input["lengths"].cuda()
        y_bin, y_final, y_dm = batch["y1"], batch["y2"], model_input["domains"]
        if self.params.tr:
            templates, tem_lengths = model_input["padded_templates"].cuda(), model_input["tem_lengths"].cuda()

        self.binary_slu_tagger.train()
        self.slotname_predictor.train()
        if self.use_label_encoder:
            self.sent_repre_generator.train()

        bin_preds, lstm_hiddens = self.binary_slu_tagger(X)

        ## optimize binary_slu_tagger
        loss_bin = self.binary_slu_tagger.crf_loss(bin_preds, lengths, y_bin)
        self.optimizer.zero_grad()
        loss_bin.backward(retain_graph=True)
        self.optimizer.step()

        ## optimize slotname_predictor
        pred_slotname_list, gold_slotname_list = self.slotname_predictor(y_dm, lstm_hiddens, binary_golds=y_bin, final_golds=y_final)

        for pred_slotname_each_sample, gold_slotname_each_sample in zip(pred_slotname_list, gold_slotname_list):
            assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
            if pred_slotname_each_sample.size(0) == 0: continue
            loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
            self.optimizer.zero_grad()
            loss_slotname.backward(retain_graph=True)
            self.optimizer.step()
        
        if self.use_label_encoder:
            templates_repre, input_repre = self.sent_repre_generator(templates, tem_lengths, lstm_hiddens, lengths)

            input_repre = input_repre.detach()
            template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
            template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
            template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
            input_repre.requires_grad = True

            self.optimizer.zero_grad()
            template0_loss.backward(retain_graph=True)
            template1_loss.backward(retain_graph=True)
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
                input_loss2.backward(retain_graph=True)
                self.optimizer.step()
        
        if self.use_label_encoder:
            return loss_bin.item(), loss_slotname.item(), template0_loss.item(), template1_loss.item()
        else:
            return loss_bin.item(), loss_slotname.item()
    
    def evaluate(self, dataloader, istestset=False):
        self.binary_slu_tagger.eval()
        self.slotname_predictor.eval()

        binary_preds, binary_golds = [], []
        final_preds, final_golds = [], []

        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in pbar:
            model_input = batch["model_input"]
            X, lengths = model_input["padded_seqs"].cuda(), model_input["lengths"].cuda()
            y_bin, y_final, y_dm = batch["y1"], batch["y2"], model_input["domains"]

            binary_golds.extend(y_bin)
            final_golds.extend(y_final)

            X, lengths = X.cuda(), lengths.cuda()
            bin_preds_batch, lstm_hiddens = self.binary_slu_tagger(X)
            bin_preds_batch = self.binary_slu_tagger.crf_decode(bin_preds_batch, lengths)
            binary_preds.extend(bin_preds_batch)

            slotname_preds_batch = self.slotname_predictor(y_dm, lstm_hiddens, binary_preditions=bin_preds_batch, binary_golds=None, final_golds=None)
            
            final_preds_batch = self.combine_binary_and_slotname_preds(y_dm, bin_preds_batch, slotname_preds_batch)
            final_preds.extend(final_preds_batch)
        
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
            "binary_slu_tagger": self.binary_slu_tagger,
            "slotname_predictor": self.slotname_predictor
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

            texts, tokens, domains, ids = batch["text"], batch["token"], batch["domain"], batch["id"]

            X, lengths = X.cuda(), lengths.cuda()
            bin_preds_batch, lstm_hiddens = self.binary_slu_tagger(X)
            bin_preds_batch = self.binary_slu_tagger.crf_decode(bin_preds_batch, lengths)
            
            slotname_preds_batch = self.slotname_predictor(y_dm, lstm_hiddens, binary_preditions=bin_preds_batch, binary_golds=None, final_golds=None)

            final_preds_batch = self.combine_binary_and_slotname_preds(y_dm, bin_preds_batch, slotname_preds_batch)
            
            for i_sam, (x, l_x, fp) in enumerate(zip(X, lengths, final_preds_batch)):
                # tokens = [self.vocab.index2word[i] for i in x[:l_x]]
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
                final_items[item['domain']].append(item)

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

    if params.tr:
        sent_repre_generator = SentRepreGenerator(params, vocab)
        sent_repre_generator = sent_repre_generator.cuda()

    if params.tr:
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor, sent_repre_generator=sent_repre_generator)
    else:
        slu_trainer = SLUTrainer(params, binary_slutagger, slotname_predictor)
    
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
    
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)

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
    
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)
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

