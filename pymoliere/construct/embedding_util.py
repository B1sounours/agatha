from typing import Tuple, Iterable, ClassVar
import torch
import torch
from transformers import (
    BertModel,
    BertTokenizer,
)
import dask.bag as dbag
import dask.dataframe as ddf
import numpy as np
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence
from pymoliere.util.misc_util import iter_to_batches
import logging
from dask.distributed import Lock
from pymoliere.util.misc_util import Record
from pymoliere.construct import dask_process_global as dpg
from pymoliere.ml.sentence_classifier import (
    record_to_sentence_classifier_input,
    sentence_classifier_output_to_labels,
)

def get_pytorch_device_initalizer(
    disable_gpu:bool,
)->Tuple[str, dpg.Initializer]:
  def _init():
    if torch.cuda.is_available() and not disable_gpu:
      return torch.device("cuda")
    else:
      return torch.device("cpu")
  return "embedding_util:device", _init

def get_scibert_initializer(
    scibert_data_dir:Path,
)->Tuple[str, dpg.Initializer]:
  def _init():
    device = dpg.get("embedding_util:device")
    tokenizer = BertTokenizer.from_pretrained(scibert_data_dir)
    model = BertModel.from_pretrained(scibert_data_dir)
    model.eval()
    model.to(device)
    return (tokenizer, model)
  return "embedding_util:tokenizer,model", _init


def get_pretrained_model_initializer(
  name:str,
  model_class:ClassVar,
  data_dir:Path,
  **model_kwargs
)->Tuple[str, dpg.Initializer]:
  def _init():
    device = dpg.get("embedding_util:device")
    model = model_class(**model_kwargs)
    model.load_state_dict(
        torch.load(
          str(data_dir),
          map_location=device,
        )
    )
    model.eval()
    return model
  return f"embedding_util:{name}", _init


def apply_sentence_classifier_to_part(
    records:Iterable[Record],
    batch_size:int,
    sentence_classifier_name="sentence_classifier",
    predicted_type_suffix=":pred",
    sentence_type_field="sent_type",
)->Iterable[Record]:
  device = dpg.get("embedding_util:device")
  model = dpg.get(f"embedding_util:{sentence_classifier_name}")

  res = []
  for rec_batch in iter_to_batches(records, batch_size):
    model_input = torch.stack(
        [record_to_sentence_classifier_input(r) for r in rec_batch]
    ).to(device)
    predicted_labels = sentence_classifier_output_to_labels(model(model_input))
    for r, lbl in zip(rec_batch, predicted_labels):
      r[sentence_type_field] = lbl+predicted_type_suffix
      res.append(r)
  print(len(res))
  return res

def _bert_to_sentence_embeddings(
    bert_model:torch.nn.Module,
    sequences:torch.FloatTensor,
    tokenizer:BertTokenizer,
    sent_emb_method:str,
)->torch.FloatTensor:
  """
  For a set of sequences, produce sentence embeddings. Look into the 'bert for
  feature extraction' heading at this page:
  http://jalammar.github.io/illustrated-bert/ 

  Strategies:
    - last_layer: Return the final output of the model. This is a joint, pooled
      representation of the whole sentence.
    - mean_hidden_without_specials: Returns the average of the model's tokens,
      ignoring special tokens.
    - mean_hidden: Returns the average of the model's tokens, including
      specials, but ignoring padding.
  """
  # Valid types
  assert sent_emb_method in {
      "last_layer",
      "mean_hidden",
      "mean_hidden_without_specials"
  }
  weights = bert_model(sequences)
  if sent_emb_method == "last_layer":
    # The last layer of the bert model is a pooled vector for the whole
    # sentence.
    embedding = weights[-1]
  else: # one of the mean_hidden, these use weights[-2]
    embedding = weights[-2]
    bad_tokens = [tokenizer.pad_token_id]
    if sent_emb_method == "mean_hidden_without_specials":
      # additional special tokens
      bad_tokens += [
          tokenizer.unk_token_id,
          tokenizer.sep_token_id,
          tokenizer.cls_token_id,
          tokenizer.mask_token_id,
      ]

    # The mask tells us which sequence values are going to contribute to the
    # average. We start with all true and then set all invalid tokens to false.
    mask = torch.ones(sequences.shape, dtype=bool, device=sequences.device)
    for bad_tok in bad_tokens:
      # set all bad tokens to false
      mask &= (sequences != bad_tok)

    # This line is kind of magic. The mask starts as a 2d structure (batch size
    # by max sequence length) but is converted to a 3d structure (batch size,
    # sequence length, embedding length) such that if a sequence is marked
    # "true" in the 2d version, that value will be replaced by a vector of
    # "true" the same size as the embedding.
    mask = mask.unsqueeze(-1).expand_as(embedding)

    # Pairwise multiply the embedding by the mask. The 2'nd to last layer has
    # one embedding per token in the input. Note that X*false = 0 and X*true = 1
    embedding *= mask

    # The average is the 1'st axis sum (over the words in the sequence)
    # pairwise divided by the total number of words (sum of the mask values).
    embedding = embedding.sum(axis=1)
    embedding /= mask.sum(axis=1)
    del mask
  # makes sure that embeddings are of appropriate shape (one vector per seq)
  assert embedding.shape[0] == sequences.shape[0]
  assert len(embedding.shape) == 2
  return embedding.float()

def embed_records(
    records:Iterable[Record],
    batch_size:int,
    text_field:str,
    max_sequence_length:int,
    sentence_embedding_method:str,
    out_embedding_field:str="embedding",
)->Iterable[Record]:
  """
  Introduces an embedding field to each record, indicated the scibert embedding
  of the supplied text field.
  """

  device = dpg.get("embedding_util:device")
  tokenizer, model = dpg.get("embedding_util:tokenizer,model")

  res = []
  for batch in iter_to_batches(records, batch_size):
    texts = list(map(lambda x: x[text_field], batch))
    sequences = pad_sequence(
        [
          torch.tensor(
            tokenizer.encode(
              t,
              add_special_tokens=True
            )[:max_sequence_length]
          )
          for t in texts
        ],
        batch_first=True,
    ).to(device)
    try:
      embs = _bert_to_sentence_embeddings(
          bert_model=model,
          sequences=sequences,
          tokenizer=tokenizer,
          sent_emb_method=sentence_embedding_method,
      ).cpu().detach().numpy()
    except:
      print(texts)
      raise Exception("Something went wrong getting embeddings.")
    del sequences
    for record, emb in zip(batch, embs):
      record[out_embedding_field] = emb
      res.append(record)
  return res
