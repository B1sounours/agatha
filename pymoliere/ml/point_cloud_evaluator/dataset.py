import torch
from pathlib import Path
from pymoliere.ml.util.embedding_index import EmbeddingIndex
from pymoliere.util.sqlite3_graph import Sqlite3Graph
from pymoliere.ml.util.entity_index import EntityIndex
from typing import List
import random
from dataclasses import dataclass
import numpy as np

@dataclass
class PointCloudObservation:
  lemma_embeddings:List[np.array]

@dataclass
class PointCloudTensors:
  lemmas:torch.FloatTensor

def pointclouds_to_tensors(
    samples:List[PointCloudObservation]
)->PointCloudTensors:
  return PointCloudTensors(
      # Seq Leng X Batch size X emb dim
      lemmas=torch.nn.utils.rnn.pad_sequence([
        torch.FloatTensor(s.lemma_embeddings) for s in samples
      ]),
  )

def sample_lemma(examples:List[PointCloudObservation]):
  return random.choice(random.choice(examples).lemma_embeddings)

def generate_negative_scramble_batch(
    positive_examples:List[PointCloudObservation]
)->PointCloudTensors:
  return pointclouds_to_tensors([
      PointCloudObservation(
        # generate a random set of equal length
        lemma_embeddings=[
          sample_lemma(positive_examples)
          for _ in pos_ref.lemma_embeddings
        ]
      )
      # Duplicate the sizes from each positive example
      for pos_ref in positive_examples
  ])

def collate_point_clouds(
    positive_examples:List[PointCloudObservation],
    neg_scrambles_per:int,
)->List[PointCloudTensors]:
  """
  The first one is the positive sample, the rest are negatives
  """
  assert neg_scrambles_per > 0
  positive_examples = [
      p for p in positive_examples if len(p.lemma_embeddings) > 0
  ]
  res = [pointclouds_to_tensors(positive_examples)]
  for _ in range(neg_scrambles_per):
    res.append(generate_negative_scramble_batch(positive_examples))
  return res


class PointCloudDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      embedding_dim:int,
      entity_dir:Path,
      embedding_index:EmbeddingIndex,
      graph_index:Sqlite3Graph,
      source_type:str,
      neigh_type:str,
      max_neighbors:int,
  ):
    assert embedding_dim > 0
    self.max_neighbors = max_neighbors
    self.neigh_type = neigh_type
    self.embedding_index = embedding_index
    self.graph_index = graph_index
    self.sentence_names = EntityIndex(entity_dir, source_type)
    self.first_call = True

  def __len__(self):
    return len(self.sentence_names)

  def __getitem__(self, idx:int)->PointCloudObservation:
    if self.first_call:
      self.graph_index.__enter__()
    name = self.sentence_names[idx]
    if name not in self.graph_index:
      print("ERROR WITH", name)
      valid_neighbors = []
    else:
      valid_neighbors = [
          n for n in self.graph_index[name] if n[0] == self.neigh_type
      ]
      if len(valid_neighbors) > self.max_neighbors:
        valid_neighbors = random.sample(valid_neighbors, self.max_neighbors)
    return PointCloudObservation(
        lemma_embeddings=[
          self.embedding_index[n] for n in valid_neighbors
        ]
    )
