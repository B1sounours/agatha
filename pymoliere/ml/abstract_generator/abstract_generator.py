import torch
import pickle
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
import math
from random import shuffle
import pytorch_lightning as pl
from pymoliere.ml.abstract_generator import datasets
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb
import os
from argparse import Namespace
from pymoliere.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
from pymoliere.ml.util.kv_store_dataset import KVStoreDictDataset


class AbstractGenerator(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(AbstractGenerator, self).__init__()
    self.hparams = hparams
    self.init_tokenizer()
    def get_emb(num):
      return torch.nn.Embedding(
          num,
          self.hparams.embedding_dim,
          padding_idx=self.tokenizer.padding_idx
    )
    self.text_embedding = get_emb(self.tokenizer.len_text())
    self.mesh_embedding = get_emb(self.tokenizer.len_mesh())
    self.year_embedding = get_emb(self.tokenizer.len_year())

    # Positional encoding is (Max Sequence Length, 1, Embedding Dim)
    self.register_buffer(
        "positional_encoding",
        self.generate_positional_encoding(
          max_text_length=self.hparams.max_text_length,
          embedding_dim=self.hparams.embedding_dim,
        )
    )

    self.transformer = torch.nn.Transformer(
        d_model=self.hparams.embedding_dim,
        nhead=self.hparams.num_attention_heads,
        num_encoder_layers=self.hparams.num_encoder_layers,
        num_decoder_layers=self.hparams.num_decoder_layers,
        dim_feedforward=self.hparams.intermediate_feedforward_dim,
        dropout=self.hparams.intermediate_dropout,
    )

    # This mask has -inf for all values that follow the target input
    # Of size (text, text)
    self.register_buffer(
        "text_attention_mask",
        self.transformer.generate_square_subsequent_mask(self.hparams.max_text_length),
    )

    def get_pred(num):
      return torch.nn.Linear(self.hparams.embedding_dim, num)
    self.predict_text = get_pred(self.tokenizer.len_text())
    self.predict_pos = get_pred(self.tokenizer.len_pos())
    self.predict_dep = get_pred(self.tokenizer.len_dep())
    self.predict_ent = get_pred(self.tokenizer.len_entity_label())

    self.softmax = torch.nn.LogSoftmax(dim=2)
    self.loss_fn = torch.nn.NLLLoss()

  # These three functions are needed to not pickle the tokenizer
  def __getstate__(self):
    # We need to exclude the tokenizer
    state = self.__dict__.copy()
    if "tokenizer" in state:
      del state["tokenizer"]
    return state

  def __setstate(self, state):
    self.__dict__.update(state)
    self.init_tokenizer()

  def init_tokenizer(self):
    if not hasattr(self, "tokenizer"):
      self.tokenizer = AbstractGeneratorTokenizer(
          **self.hparams.tokenizer_kwargs
      )


  def forward(
      self,
      text:torch.LongTensor,
      year:torch.LongTensor,
      mesh:torch.LongTensor,
  ):
    all_tensors = [text, year, mesh]
    for t in all_tensors:
      assert len(t.shape) == 2, "Input tensor must be 2d"
      assert t.shape[1] ==all_tensors[0].shape[1], "Unequal batch size"
      assert t.shape[0] <= self.hparams.max_text_length, "Max seq len exceeded"
    assert year.shape[0] == 1, f"Year must be a seq of 1 elem, instead {year.shape[0]}"

    text_emb = self.text_embedding(text)
    year_emb = self.year_embedding(year)
    mesh_emb = self.mesh_embedding(mesh)

    text_len = text.shape[0]
    positional_text_emb = text_emb + self.positional_encoding[:text_len, :, :]
    text_padding_mask = (text==self.tokenizer.padding_idx).t_()

    context_emb = torch.cat((year_emb, mesh_emb))
    context_padding_mask = \
        (torch.cat((year, mesh)) == self.tokenizer.padding_idx).t_()

    encoded_text = self.transformer(
        src=context_emb,
        src_key_padding_mask=context_padding_mask,
        tgt=positional_text_emb,
        tgt_key_padding_mask=text_padding_mask,
        tgt_mask=self.text_attention_mask[:text_len,:text_len],
    )

    return {
        "text": self.softmax(self.predict_text(encoded_text)),
        "pos": self.softmax(self.predict_pos(encoded_text)),
        "dep": self.softmax(self.predict_dep(encoded_text)),
        "ent": self.softmax(self.predict_ent(encoded_text)),
    }

  def generate_positional_encoding(
      self,
      max_text_length:int,
      embedding_dim:int,
  )->torch.FloatTensor:
    # Dim must be even
    assert embedding_dim % 2 == 0

    # Returns a (seq_len, emb) tensor
    positional_encodings = []
    for pos in range(max_text_length):
      positional_encodings.append([])
      for i in range(0, int(embedding_dim / 2)):
        # Even index
        positional_encodings[-1].append(
          math.sin(pos / (10000 ** (2 * i / embedding_dim)))
        )
        # Odd index
        positional_encodings[-1].append(
          math.cos(pos / (10000 ** (2 * i / embedding_dim)))
        )
    result = torch.FloatTensor(
        positional_encodings,
    ).unsqueeze(1)
    assert result.shape == (max_text_length, 1, embedding_dim)
    return result

  def _get_masked_text_vectors(self, expected, predicted):
    mask = (expected['text'] != self.hparams.padding_idx)
    # matrix of probability vectors over vocab
    masked_predicted_text = (
        predicted['text'][mask]
        .view(-1, predicted['text'].shape[2])
    )
    # vector of indices from 0 to vocab size
    masked_expected_text = (
        expected['text'][mask]
        .view(-1)
        -self.hparams.vocab_start_idx
    )
    return masked_expected_text, masked_predicted_text

  # Added for pytorch-lightning
  def training_step(self, batch, batch_nb):
    self.init_tokenizer()
    model_in, shifted_text_features = batch
    predicted_text_features = self.forward(**model_in)
    for key in shifted_text_features:
      assert key in predicted_text_features
    assert "text" in shifted_text_features
    mask = shifted_text_features["text"] != self.tokenizer.padding_idx
    mask_count = mask.sum()
    partial_losses = {}
    partial_accuracies = {}
    for key in shifted_text_features:
      actual = predicted_text_features[key][mask].view(mask_count, -1)
      expected = shifted_text_features[key][mask].view(mask_count)
      partial_losses[key] = self.loss_fn(actual, expected)
      partial_accuracies[key] = (
          (actual.argmax(dim=1) == expected)
          .float().mean()
      ).detach()

    loss = sum(partial_losses.values())
    metrics = {}
    for key in partial_losses:
      metrics[f"{key}_loss"] = partial_losses[key]
      metrics[f"{key}_acc"] = partial_accuracies[key]
    metrics["loss"] = loss

    # Return dict of metrics
    return {
        'loss': loss,
        'progress_bar': metrics,
        'log': metrics,
    }

  def _collate(self, batch):
    return datasets.shift_text_features_for_training(
        datasets.collate_encoded_abstracts(batch)
    )
  @pl.data_loader
  def train_dataloader(self):
    self.init_tokenizer()
    abstracts = KVStoreDictDataset(self.hparams.training_data_dir)
    encoder = datasets.EncodedAbstracts(
        abstract_ds=abstracts,
        tokenizer_kwargs=self.hparams.tokenizer_kwargs,
        max_text_length=self.hparams.max_text_length+1, # add one because we shift
        max_mesh_length=self.hparams.max_text_length-1, # remove one because year
    )
    sampler=torch.utils.data.distributed.DistributedSampler(encoder)
    loader = torch.utils.data.DataLoader(
        dataset=encoder,
        sampler=sampler,
        batch_size=self.hparams.batch_size,
        collate_fn=self._collate,
        #num_workers=self.hparams.dataset_workers,
    )
    return loader


  def configure_optimizers(self):
    optimizer = Lamb(
        self.parameters(),
        # facebook paper says linear growth with batch size
        lr=self.hparams.learning_rate,
        weight_decay=0.01,
    )
    return optimizer

  def optimizer_step(
      self,
      epoch_idx,
      batch_idx,
      optimizer,
      optimizer_idx,
      second_order_closure=None
  ):
    # warm up lr
    if  self.trainer.global_step < self.hparams.warmup_steps:
      lr_scale = min(
          1.,
          float(self.trainer.global_step + 1)/float(self.hparams.warmup_steps)
      )
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale * self.hparams.learning_rate
    optimizer.step()
    optimizer.zero_grad()

  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size
    )
