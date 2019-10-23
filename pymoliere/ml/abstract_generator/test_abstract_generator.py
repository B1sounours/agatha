import pymoliere.ml.abstract_generator.util as util
from transformers import BertTokenizer, AdamW
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from copy import copy

SENTENCE_1 = (
    "Hypothesis generation is becoming a crucial time-saving technique which "
    "allows biomedical researchers to quickly discover implicit connections "
    "between important concepts."
)

SENTENCE_2 = (
    "Typically, these systems operate on domain-specific fractions of public "
    "medical data."
)

SENTENCE_3 = (
    "MOLIERE, in contrast, utilizes information from over 24.5 million "
    "documents."
)



def test_bert_generator_forward():
  """
  This test ensures that the modified version of bert can successfully do a
  single forward pass given two sentences.  The model should convert the hidden
  layers following the model to per-embedding logits.
  """
  model = util.AbstractGenerator.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  batch = pad_sequence(
      sequences=[
        torch.tensor(
          tokenizer.encode(
            SENTENCE_1,
            text_pair=SENTENCE_2,
            add_special_tokens=True
          )
        ),
        torch.tensor(
          tokenizer.encode(
            SENTENCE_2,
            text_pair=SENTENCE_3,
            add_special_tokens=True
          )
        ),
      ],
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  # batch of x, many tokens, softmax over voccab
  expected_shape = (batch.shape[0], batch.shape[1], tokenizer.vocab_size)
  res = model(batch)
  actual_shape = res.shape

  # Make sure that the above doesn't break
  assert expected_shape == actual_shape

def test_bert_generator_backwards():
  model = util.AbstractGenerator.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  loss_fn = torch.nn.NLLLoss()
  optim = AdamW(model.parameters(), lr=1)
  batch = pad_sequence(
      sequences=[
        torch.tensor(
          tokenizer.encode(
            SENTENCE_1,
            text_pair=SENTENCE_2,
            add_special_tokens=True
          )
        ),
        torch.tensor(
          tokenizer.encode(
            SENTENCE_2,
            text_pair=SENTENCE_3,
            add_special_tokens=True
          )
        ),
      ],
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  # We're going to test that the last layer updates
  assert model.last_hidden2voccab.weight.requires_grad == True
  init_hidden_values = model.last_hidden2voccab.weight.detach().numpy().copy()

  # Super simple update model
  predicted_logits = model(batch)
  optim.zero_grad()
  # Loss expects a vec of nll vectors, and a vec of ints
  loss = loss_fn(
      predicted_logits.view(-1, tokenizer.vocab_size),
      batch.view(-1),
  )
  loss.backward()
  optim.step()

  final_hidden_values = model.last_hidden2voccab.weight.detach().numpy()
  print(init_hidden_values.shape)
  # We want to make sure that the weights have changed
  assert not np.allclose(init_hidden_values, final_hidden_values)


def test_mask_sentence():
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  original_sequence = tokenizer.encode(
      text=SENTENCE_1,
      text_pair=SENTENCE_2,
      add_special_tokens=True
  )
  mask = [False] * len(original_sequence)
  mask[2] = mask[-2] = True
  # do the expected manually
  expected = copy(original_sequence)
  expected[-2] = expected[2] = tokenizer.mask_token_id
  expected[2] = tokenizer.mask_token_id
  # get actual
  actual = util.mask_sequence(
      tokenizer=tokenizer,
      sequence=original_sequence,
      mask=mask
  )
  # require that actual is a modified copy
  assert original_sequence != actual
  assert actual == expected


def generate_sentence_mask():
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  first_sent_seq = tokenizer.encode(SENTENCE_1, add_special_tokens=True)
  full_sequence = tokenizer.encode(
      text=SENTENCE_1,
      text_pair=SENTENCE_2,
      add_special_tokens=True
  )
  valid_mask_values = set(range(len(first_sent_seq), len(full_sequence)))
  # MASK EVERYTHING
  mask = util.generate_sentence_mask(tokenizer, full_sequence, 1)
  assert len(mask) == len(full_sequence)
  for idx, is_masked in enumerate(mask):
    if is_masked:
      assert idx in valid_mask_values


def test_group_sentences_into_pairs_sorted():
  records = [{
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_unsorted():
  # This sort order is difficult because when we see "first" we have to
  # generate both pairs at that time
  records = [{
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_bad_abstract():
  records = [{
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "ignore me!",
        "sent_idx": 0,
        "sent_total": 1,
        "pmid": 987,
        "version": 1,
      }, {
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_separate_versions():
  records = [{
        "sent_text": "v1 first",
        "sent_idx": 1,
        "sent_total": 2,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "v1 title",
        "sent_idx": 0,
        "sent_total": 2,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "v2 title",
        "sent_idx": 0,
        "sent_total": 2,
        "pmid": 123,
        "version": 2,
      }, {
        "sent_text": "v2 first",
        "sent_idx": 1,
        "sent_total": 2,
        "pmid": 123,
        "version": 2,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("v1 title", "v1 first"),
      ("v2 title", "v2 first"),
  ]
  assert set(actual) == set(expected)
