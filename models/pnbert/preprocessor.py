# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

import os, sys
import json
import collections
import logging
import numpy as np

logger = logging.getLogger()


def convert_examples_to_features(examples, domain_map, intent_map, slots_map, 
                           max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  features = []
  seg = pkuseg.pkuseg()
  for (ex_index, example) in enumerate(examples):
    ori_slots = slots_convert(example.text_a, example.slots)
    # tokens_a = tokenizer.tokenize(example.text_a)
    tokens_a = []
    tokens_slots = []
    for i, word in enumerate(example.text_a):
      token = tokenizer.tokenize(word)
      tokens_a.extend(token)
      if len(token) > 0:
        tokens_slots.append(ori_slots[i])
    if not len(tokens_a) == len(tokens_slots):
      logger.info("********** Take Care! ***********")
      print(tokens_a)
      print(tokens_slots)
    assert len(tokens_a) == len(tokens_slots)

    # tokens_b = None
    # if example.text_b:
    #   tokens_b = tokenizer.tokenize(example.text_b)

    # if tokens_b:
    #   # Modifies `tokens_a` and `tokens_b` in place so that the total
    #   # length is less than the specified length.
    #   # Account for [CLS], [SEP], [SEP] with "- 3"
    #   _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    # else:
      # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]
      tokens_slots = tokens_slots[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    slots_id = []
    segment_ids = []
    tokens.append("[CLS]")
    slots_id.append(slots_map["O"])
    segment_ids.append(2)
    for token, slots in zip(tokens_a, tokens_slots):
      tokens.append(token)
      slots_id.append(slots_map[slots])
      # segment_ids.append(1)
    
    # use third cut word tool to display segment id
    cut_seg = seg.cut(example.text_a)
    for word in cut_seg:
      for i in range(len(word)):
        if i == 0:
          segment_ids.append(0)
        else:
          segment_ids.append(1)
    
    if len(segment_ids) > max_seq_length - 1:
      segment_ids = segment_ids[0:(max_seq_length - 1)]

    assert len(tokens) == len(segment_ids)
    

    # print("segment id", segment_ids, example.text_a)
    tokens.append("[SEP]")
    slots_id.append(slots_map["O"])
    segment_ids.append(2)

    # if tokens_b:
    #   for token in tokens_b:
    #     tokens.append(token)
    #     segment_ids.append(1)
    #   tokens.append("[SEP]")
    #   segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(2)
      slots_id.append(slots_map["O"])

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slots_id) == max_seq_length

    domain_id, intent_id = 0, 0
    if example.domain:
      domain_id = domain_map[example.domain]
    if example.intent:
      intent_id = intent_map[example.intent]

    # if ex_index < 1:
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("guid: %s" % (example.guid))
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in tokens]))
    #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    features.append(InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        domain_id=domain_id,
        intent_id=intent_id,
        slots_id=slots_id,
        is_real_example=True))

  return features


