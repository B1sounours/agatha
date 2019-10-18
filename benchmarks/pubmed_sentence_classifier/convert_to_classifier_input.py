#!/usr/bin/env python3

"""
This script converts the raw textual data provided in raw_data into the
expected input of the sentence classifier model. This entails embedding each
record, and converting the results into a mocked checkpoint directory.
"""

from argparse import ArgumentParser
from pathlib import Path
from pymoliere.ml.sentence_classifier import util as sent_class_util
from pymoliere.construct import embedding_util
from pymoliere.util.misc_util import Record
from typing import Iterable
from pymoliere.construct import dask_process_global as dpg
import pickle


def parse_raw_file(raw_file_path:Path)->Iterable[Record]:
  docs = []
  with open(raw_file_path) as f:
    for line in f:
      line = line.strip()
      if len(line) == 0:
        continue
      if line[0] == "#":  # abstract header indicator
        docs.append([])
        continue
      docs[-1].append(line)

  res = []
  for doc in docs:
    for idx, line in enumerate(doc):
      try:
        label, text = line.split(" ", 1)
        label = f"abstract:{label.lower()}"
        res.append({
          "text": text,
          "sent_type": label,
          "sent_idx": (idx+1),
          "sent_total": len(doc),
          "date": "9999-99-99"
        })
      except:
        print(f"Err: '{line}'")
  return res


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--bert_data_dir", type=Path)
  parser.add_argument("--raw_data_in", type=Path)
  parser.add_argument("--eval_data_out", type=Path)
  parser.add_argument("--disable_gpu", action="store_true")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--max_sequence_length", type=int, default=500)
  args = parser.parse_args()

  assert args.raw_data_in.is_file()
  args.eval_data_out.mkdir(parents=True, exist_ok=True)

  print("Prepping embedding")
  preloader = dpg.WorkerPreloader()
  preloader.register(*embedding_util.get_pytorch_device_initalizer(
      disable_gpu=args.disable_gpu,
  ))
  preloader.register(*embedding_util.get_scibert_initializer(
      scibert_data_dir=args.bert_data_dir,
  ))
  dpg.add_global_preloader(preloader=preloader)

  print("Converting to records.")
  records = parse_raw_file(args.raw_data_in)

  # Step 3: Embed Records
  embedded_records = embedding_util.embed_records(
      records,
      batch_size=args.batch_size,
      text_field="text",
      max_sequence_length=args.max_sequence_length,
      show_pbar=True,
  )

  # Step 4: Records to training data via util
  print("Converting to sentence_classifier.util.TrainingData")
  training_data = [
      sent_class_util.record_to_training_tuple(r)
      for r in embedded_records
  ]

  print("Saving as mock ckpt")
  done_file = args.eval_data_out.joinpath("__done__")
  part_file = args.eval_data_out.joinpath("part-0.pkl")
  with open(part_file, 'wb') as f:
    pickle.dump(training_data, f)
  with open(done_file, 'w') as f:
    f.write(f"{part_file}\n")
