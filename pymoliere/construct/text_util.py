from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict, Callable, Iterable
import spacy
from copy import copy
from nltk.tokenize import sent_tokenize
from pymoliere.util import db_key_util
import logging
from pymoliere.construct import dask_process_global as dpg

from pymoliere.util.misc_util import Record, Edge

GLOBAL_NLP_OBJS = {
    "analyze_sentence": None,
}

################################################################################

def _op_g_pre(graph:bool)->str:
  "helper for graph ids"
  return f"{db_key_util.GRAPH_TYPE}:" if graph else ""

def sentence_to_id(sent:Record, graph:bool=False)->str:
  return get_sentence_id(
      pmid=sent["pmid"],
      version=sent["version"],
      sent_idx=sent["sent_idx"],
      graph=graph,
  )

def get_sentence_id(
    pmid:int,
    version:int,
    sent_idx:int,
    graph:bool=False
)->str:
  typ = db_key_util.SENTENCE_TYPE
  return f"{_op_g_pre(graph)}{typ}:{pmid}:{version}:{sent_idx}"

def token_to_id(
    token:Record,
    graph:bool=False,
)->str:
  typ = db_key_util.LEMMA_TYPE
  lem = token["lemma"]
  pos = token["pos"]
  return f"{_op_g_pre(graph)}{typ}:{pos}:{lem}".lower()

def entity_to_id(
    entity:Record,
    sentence:Record,
    token_field:str="tokens",
    graph:bool=False,
)->str:
  ent = get_entity_text(
      entity=entity,
      sentence=sentence,
      token_field=token_field
  )
  typ = db_key_util.ENTITY_TYPE
  return f"{_op_g_pre(graph)}{typ}:{ent}".lower()

def get_entity_text(
    entity:Record,
    sentence:Record,
    token_field:str="tokens",
)->str:
  return "_".join([
      sentence[token_field][tok_idx]["lemma"]
      for tok_idx in range(entity["tok_start"], entity["tok_end"])
  ])

def mesh_to_id(
    mesh_code:str,
    graph:bool=False,
)->str:
  typ = db_key_util.MESH_TERM_TYPE
  return f"{_op_g_pre(graph)}{typ}:{mesh_code}".lower()

################################################################################

def get_scispacy_initalizer(
    scispacy_version:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    return spacy.load(scispacy_version)
  return "text_util:nlp", _init

def get_stopwordlist_initializer(
    stopword_path:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    with open(stopword_path, 'r') as f:
      return {line.strip().lower() for line in f}
  return "text_util:stopwords", _init

################################################################################

def split_sentences(
    document_elem:Record,
    text_data_field:str="text_data",
    id_field:str="id",
    min_sentence_len:Optional[int]=None,
    max_sentence_len:Optional[int]=None,
)->List[Record]:
  """
  Splits a document into its collection of sentences. In order of text field
  elements, we split sentences and create new elements for the result. All
  fields from the original document, as well as the text field (minus the
  actual text itself) are copied over.

  If min/max sentence len are specified, we do NOT consider sentences that fail
  to match the range.

  `id_field` will be set with {SENTENCE_TYPE}:{pmid}:{version}:{sent_idx}

  For instance:

  {
    "status": "Published",
    "umls": ["C123", "C456"],
    "text_fields": [{
        "text": "Title 1",
        "type": "title"
      }, {
        "text": "This is an abstract. This is another sentence.",
        "type": "abstract:raw",
      }]
  }

  becomes:

  [{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "Title 1",
    "sent_type": "title",
    "sent_idx": 0,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is an abstract.",
    "sent_type": "abstract:raw",
    "sent_idx": 1,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is another sentence.",
    "sent_type": "abstract:raw",
    "sent_idx": 2,
    "sent_total": 3,
  }]
  """
  assert text_data_field in document_elem
  sent_text_key = f"sent_text"
  sent_idx_key = f"sent_idx"
  sent_total_key = f"sent_total"

  res = []

  # Get all non-textual data
  doc_non_text_elem = copy(document_elem)
  doc_non_text_elem.pop(text_data_field)

  sent_idx = 0
  for text_data in document_elem[text_data_field]:
    assert "text" in text_data
    # Holds all additional text features per-sentence in this field
    non_text_attr = copy(doc_non_text_elem)
    for key, val in text_data.items():
      if key != "text":
        key = f"sent_{key}"
        assert key not in non_text_attr
        non_text_attr[key] = val
    for sentence_text in sent_tokenize(text_data["text"]):
      if min_sentence_len is not None and len(sentence_text) < min_sentence_len:
        continue
      if max_sentence_len is not None and len(sentence_text) > max_sentence_len:
        continue
      sent_rec = copy(non_text_attr)
      assert sent_text_key not in sent_rec
      sent_rec[sent_text_key] = sentence_text
      assert sent_idx_key not in sent_rec
      sent_rec[sent_idx_key] = sent_idx
      sent_idx += 1
      res.append(sent_rec)

  # set total
  for r in res:
    assert sent_total_key not in r
    r[sent_total_key] = sent_idx
    r[id_field] = sentence_to_id(r)
  return res

def analyze_sentence(
    sent_rec:Record,
    text_field:str,
    nlp:Any=None,
    token_field:str="tokens",
    entity_field:str="entities",
    #vector_field:str="vector"
)->Record:
  "Splits tokens into useful components"
  assert text_field in sent_rec
  assert token_field not in sent_rec
  assert entity_field not in sent_rec
  #assert vector_field not in sent_rec

  if nlp is None:
    nlp = dpg.get("text_util:nlp")
  stopwords = dpg.get("text_util:stopwords")

  sent_rec = copy(sent_rec)
  try:
    doc = nlp(sent_rec[text_field])
    #sent_rec[vector_field] = doc.vector.tolist()
    sent_rec[entity_field] = [
      {
        "tok_start": ent.start,
        "tok_end": ent.end,
        "cha_start": ent.start_char,
        "cha_end": ent.end_char,
        #"umls": ent._.umls_ents[0][0] if len(ent._.umls_ents) > 0 else None,
        #"vector": ent.vector.tolist(),
      }
      for ent in doc.ents
    ]
    sent_rec[token_field] = [
        {
          "cha_start": tok.idx,
          "cha_end": tok.idx + len(tok),
          "lemma": tok.lemma_,
          "pos": tok.pos_,
          "tag": tok.tag_,
          "dep": tok.dep_,
          #"vector": tok.vector.tolist() if not tok.is_stop else None,
          "stop": \
              tok.lemma_ in stopwords or tok.text.strip().lower() in stopwords
        }
        for tok in doc
    ]
  except Exception as e:
    sent_rec["ERR"] = str(e)
    logging.info(e)
  return sent_rec


def add_bow_to_analyzed_sentence(
    record:Record,
    bow_field="bow",
    token_field="tokens",
    entity_field="entities",
    mesh_heading_field="mesh_headings"
)->Record:
  assert bow_field not in record
  bow = []
  for lemma in record[token_field]:
    if lemma["pos"] in {"NOUN", "VERB", "ADJ"} and not lemma["stop"]:
      bow.append(lemma["lemma"])
  for entity in record[entity_field]:
    is_stop_ent = True
    for t in record[token_field][entity["tok_start"]:entity["tok_end"]]:
      if not t["stop"]:
        is_stop_ent = False
        break
    # A stop-entity is one comprised of only stopwords such as "et_al."
    if not is_stop_ent:
      ent_text = get_entity_text(
          entity=entity,
          sentence=record,
          token_field=token_field
      )
      bow.append(ent_text)
  bow += record["mesh_headings"]
  record["bow"] = bow
  return record


def get_edges_from_sentence_part(
    sentences:Iterable[Record],
    token_field:str="tokens",
    entity_field:str="entities",
    mesh_field:str="mesh_headings",
    sent_idx_field:str="sent_idx",
    sent_total_field:str="sent_total",
    id_field:str="id",
)->Iterable[Edge]:
  res = []
  for sent_rec in sentences:
    sent_k = db_key_util.to_graph_key(sent_rec[id_field])
    for token in sent_rec[token_field]:
      if token["stop"] or token["pos"] in ("PUNCT"):
        continue
      tok_k = token_to_id(token, graph=True)
      res.append(db_key_util.to_edge(source=sent_k, target=tok_k))
      res.append(db_key_util.to_edge(target=sent_k, source=tok_k))
    for entity in sent_rec[entity_field]:
      ent_k = entity_to_id(
          entity=entity,
          sentence=sent_rec,
          token_field=token_field,
          graph=True
      )
      res.append(db_key_util.to_edge(source=sent_k, target=ent_k))
      res.append(db_key_util.to_edge(target=sent_k, source=ent_k))
    for mesh_code in sent_rec[mesh_field]:
      mesh_k = mesh_to_id(mesh_code, graph=True)
      res.append(db_key_util.to_edge(source=sent_k, target=mesh_k))
      res.append(db_key_util.to_edge(target=sent_k, source=mesh_k))
    # Adj sentence edges. We only need to make edges for "this" sentence,
    # because the other sentences will get the other sides of each connection.
    if sent_rec[sent_idx_field] > 0:
      res.append(
        db_key_util.to_edge(
          source=sent_k,
          target=get_sentence_id(
            pmid=sent_rec["pmid"],
            version=sent_rec["version"],
            sent_idx=sent_rec["sent_idx"]-1,
            graph=True,
          )
        )
      )
    if sent_rec[sent_idx_field] < sent_rec[sent_total_field]-1:
      res.append(
        db_key_util.to_edge(
          source=sent_k,
          target=get_sentence_id(
            pmid=sent_rec["pmid"],
            version=sent_rec["version"],
            sent_idx=sent_rec["sent_idx"]+1,
            graph=True,
          )
        )
      )
  return res
