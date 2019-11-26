import dask
from dask.distributed import Client
import horovod.torch as hvd
from nltk.tokenize import sent_tokenize
from pathlib import Path
import pickle
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.construct import dask_checkpoint, file_util, text_util, ftp_util
from pymoliere.ml.model_summary import print_model_summary
from pymoliere.ml.abstract_generator.misc_util import HashedIndex, OrderedIndex
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb
from pymoliere.ml.abstract_generator.generation_util import (
    generate_new_text,
)
from pymoliere.ml.abstract_generator.abstract_generator import (
    INTERESTING_SENTENCE_LABLES,
    AbstractGeneratorTokenizer,
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.batch_generator import (
    AbstractWindowGenerator
)
from pymoliere.ml.train_model import (
    train_model,
    split_partitions_across_ranks,
    split_list_by_rank,
)
from pymoliere.util.misc_util import Record, iter_to_batches
import sentencepiece as spm
import sys
import torch
from typing import Iterable, List, Dict
import random
import os
from tqdm import tqdm

MODES = ["train", "evaluate", "prep"]

# Taken from transformers module. This function (get_linear_schedule_with_warmup)
# is under the Apache2 License
def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
  """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
  """
  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) \
        / float(max(1, num_training_steps - num_warmup_steps)))
  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def items_to_hashed_index(collection:Iterable[str], max_index:int)->HashedIndex:
  res = HashedIndex(max_index=max_index)
  for elem in collection:
    res.add(elem)
  return res

def items_to_ordered_index(collection:Iterable[str])->OrderedIndex:
  res = OrderedIndex()
  for elem in collection:
    res.add(elem)
  return res

def connect_to_dask_cluster(config:cpb.AbstractGeneratorConfig)->None:
  # Potential cluster
  if config.cluster.run_locally or config.cluster.address == "localhost":
    print("Running dask on local machine!")
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
    if config.cluster.restart:
      print("\t- Restarting cluster...")
      dask_client.restart()

def get_paths(config:cpb.AbstractGeneratorConfig):
  """
  Returns all the relevant paths based on data from the config.
  """
  # Location we can find the existing data
  assert config.cluster.HasField("shared_scratch")
  scratch_root_dir = Path(config.cluster.shared_scratch)
  pmc_download_dir = scratch_root_dir.joinpath("pmc_raw")
  pmc_download_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_dir = scratch_root_dir.joinpath("dask_checkpoints")
  model_root_dir = scratch_root_dir.joinpath("models").joinpath("abstract_generator")
  if config.HasField("model_path"):
    model_path = Path(config.model_path)
  else:
    model_path = model_root_dir.joinpath("model.pt")
  model_ckpt_dir = model_root_dir.joinpath("dask_checkpoints")
  model_extra_data_path = model_root_dir.joinpath("extra_data.pkl")
  tokenizer_training_data_dir = \
      model_ckpt_dir.joinpath("tokenizer_training_data")
  tokenizer_model_path = model_root_dir.joinpath("tokenizer.model")
  tokenizer_vocab_path = model_root_dir.joinpath("tokenizer.vocab")

  # List of all directories
  dir_paths = [
      path for name, path in locals().items()
      if name.split("_")[-1]=="dir"
  ]
  # Make sure all dirs are present
  for dir_path in dir_paths:
    dir_path.mkdir(parents=True, exist_ok=True)

  # Return all paths, provided they end in "_dir" or "_path"
  return {
      name: path
      for name, path in locals().items()
      if name.split("_")[-1] in ["dir", "path"]
  }

def init_everything_for_hvd():
  seed = 42
  hvd.init()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.set_num_threads(4)
  torch.cuda.set_device(hvd.local_rank())

def get_tokenizer_from_config(
    config:cpb.AbstractGeneratorConfig
)->AbstractGeneratorTokenizer:
  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]
  assert tokenizer_model_path.is_file()
  assert extra_data_path.is_file()
  return AbstractGeneratorTokenizer(
      tokenizer_model_path=tokenizer_model_path,
      extra_data_path=extra_data_path,
  )

def get_model_from_config(
    config:cpb.AbstractGeneratorConfig,
    tokenizer:AbstractGeneratorTokenizer,
)->AbstractGenerator:
  return AbstractGenerator(
      tokenizer=tokenizer,
      embedding_dim=config.embedding_dim,
      max_text_length=config.text_length,
      num_attention_heads=config.num_attention_heads,
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers,
      intermediate_dropout=0.1,
      intermediate_feedforward_dim=config.hidden_fc_size,
  )

def get_device(config:cpb.AbstractGeneratorConfig)->torch.device:
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def evaluate(config:cpb.AbstractGeneratorConfig):
  init_everything_for_hvd()
  paths = get_paths(config)

  testing_data_dir = paths["model_ckpt_dir"].joinpath("testing_data")
  assert testing_data_dir.is_dir()

  device = get_device(config)
  tokenizer = get_tokenizer_from_config(config)
  model = get_model_from_config(config, tokenizer)
  model.load_state_dict(torch.load(paths["model_path"])["model_state_dict"])
  model.to(device)
  model.eval()

  testing_data = split_partitions_across_ranks(
      testing_data_dir,
      rank=hvd.rank(),
      size=10 if config.debug else hvd.size(),
  )

  batch_generator = AbstractWindowGenerator(
      tokenizer=tokenizer,
      records=testing_data,
      batch_size=1,
      text_size=config.text_length,
      return_training_data=False,
      only_first_window_per_abstract=True,
  )

  for record in testing_data:
    pmid = record["pmid"]
    year = int(record["date"].split("-")[0])
    title_rec = record["text_data"][0]
    assert title_rec["type"] == "title"
    title=title_rec["text"]
    mesh_headings=record["mesh_headings"]

    title_tokens = \
        [tokenizer.start_symbol_idx] + tokenizer.encode_text(title)

    context = tokenizer.encode_context_sequence(
        year=year,
        mesh_headings=mesh_headings,
    )
    text_generator = generate_new_text(
        model=model,
        tokenizer=tokenizer,
        context=torch.LongTensor(context).unsqueeze(1).to(device),
        text=torch.LongTensor(title_tokens).unsqueeze(1).to(device),
    )
    texts = []
    for idx in range(200):
      te = next(text_generator)
      texts.append(te)
    prediction=tokenizer.decode_text(texts)

    print(
        f"{pmid},{year},'{','.join(mesh_headings)}','{title}','{prediction}'"
    )

def distribute_training_partitions(
    partition_files:List[Path],
    rank:int,
    size:int,
    max_result_size:int,
)->List[Dict[str, torch.Tensor]]:
  if hvd.rank() == 0:
    print(f"Splitting {len(partition_files)} paths across {size} machines")
  # everyone needs to setup the index tensor
  indices = torch.randperm(len(partition_files))
  # reset everyone to the tensor owned by rank 0
  indices = hvd.broadcast(indices, root_rank=0, name="indices").tolist()
  # split the indies list up
  indices = split_list_by_rank(indices, rank, size)
  #print(f"I'm responsible for {len(indices)} files:", indices)
  res = []
  for idx in indices:
    with open(partition_files[idx], 'rb') as f:
      res += pickle.load(f)
  if max_result_size / len(res) < 0.75:
    print(f"Warning, only selecting {max_result_size} out of {len(res)}")
  random.shuffle(res)
  return res[:max_result_size]

def train(config:cpb.AbstractGeneratorConfig):
  init_everything_for_hvd()
  paths = get_paths(config)

  training_data_dir = paths["model_ckpt_dir"].joinpath("training_data_windows")
  assert training_data_dir.is_dir()

  all_training_files = list(training_data_dir.glob("*.pkl"))
  assert len(all_training_files) > 0
  all_training_files.sort()

  effective_batch_size = config.sys.batch_size*hvd.size()
  if hvd.rank() == 0:
    print("Effective batch size:", effective_batch_size)


  if hvd.rank() == 0:
    print("Preparing model")

  device = get_device(config)
  tokenizer = get_tokenizer_from_config(config)
  model = get_model_from_config(config, tokenizer)


  #print_model_summary(model)
  model.to(device)

  loss_fn = torch.nn.NLLLoss()
  optimizer = Lamb(
      model.parameters(),
      # facebook paper says linear growth with batch size
      lr=config.sys.learning_rate,
      weight_decay=0.01,
  )

  # Prep for horovod
  optimizer = hvd.DistributedOptimizer(
      optimizer,
      named_parameters=model.named_parameters(),
  )

  schedule = get_linear_schedule_with_warmup(
      optimizer=optimizer,
      num_warmup_steps=config.num_warmup_steps,
      num_training_steps=config.sys.steps_per_epoch * config.sys.num_epochs,
  )

  start_epoch = 0

  # load checkpoint if found
  if config.HasField("restore_from_checkpoint"):
    if hvd.rank() == 0:
      print("Loading checkpoint", config.restore_from_checkpoint)
    ckpt = torch.load(config.restore_from_checkpoint)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    schedule.load_state_dict(ckpt["schedule_state_dict"])
    start_epoch = ckpt["epoch"]

  # Update everybody
  # in the case we loaded from a checkpoint, this is very important
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)

  if config.debug:
    print_model_summary(model)

  # At this point, we just need to run train.
  # To do so, we're going to define a bunch of callback functions

  def loss_wrapper(predicted, expected):
    mask = expected["text"] != tokenizer.padding_idx
    def part(pre, exp, start_idx):
      assert len(pre.shape) == 3
      assert len(exp.shape) == 2
      assert pre.shape[0] == exp.shape[0]
      assert pre.shape[1] == exp.shape[1]
      return loss_fn(
          pre[mask].view(-1, pre.shape[2]),
          exp[mask].view(-1)-start_idx,
      )
    return part(predicted["text"], expected["text"], tokenizer.vocab_start_idx) \

  def after_loss_calculation(loss):
      loss.backward()
      optimizer.synchronize()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      with optimizer.skip_synchronize():
        optimizer.step()
      schedule.step()
      optimizer.zero_grad()

  def on_epoch_start(epoch):
    if hvd.rank() == 0:
      print()
      print()
      print("Epoch:", epoch)
    if (
        epoch > 0
        and hvd.rank() == 0
    ):
      print("Saving model checkpoint")
      torch.save({
          "model_state_dict": model.state_dict(),
          "optimizer_state_dict": optimizer.state_dict(),
          "schedule_state_dict": schedule.state_dict(),
          "epoch": epoch,
        },
        f"{paths['model_path']}.{epoch}"
      )

  def generator_wrapper(epoch):
    training_data = distribute_training_partitions(
        all_training_files,
        rank=hvd.rank(),
        size=100 if config.debug else hvd.size(),
        max_result_size=config.sys.steps_per_epoch * config.sys.batch_size
    )
    for batch_data in iter_to_batches(training_data, config.sys.batch_size):
      vals = {
          f: (
            AbstractWindowGenerator
            .field_to_long_tensor(batch_data, f)
            .to(device)
          )
          for f in batch_data[0]
      }
      yield (
          {"text": vals["text"], "context": vals["context"]},
          {"text": vals["shifted_text"]},
      )


  def partial_accuracy(predicted, expected, value, start_idx):
    mask = expected != tokenizer.padding_idx
    predicted = predicted[value].argmax(dim=2)
    expected = expected[value] - start_idx
    assert predicted.shape[0] == expected.shape[0]
    assert predicted.shape[1] == expected.shape[1]
    return (predicted[mask] == expected[mask]).float().mean()

  def text_accuracy(predicted, expected):
    return partial_accuracy(predicted, expected, "text", tokenizer.vocab_start_idx)

  def on_phase_end(phase, metric2average):
    if phase == "train":
      text_acc = hvd.allreduce(
          metric2average["text_acc"],
          name="text_acc"
      )
      if hvd.rank() == 0:
        print("Epoch end text accuracy,", float(text_acc))

  train_model(
      model=model,
      loss_fn=loss_wrapper,
      num_epochs=config.sys.num_epochs,
      on_epoch_start=on_epoch_start,
      batch_generator=generator_wrapper,
      after_loss_calculation=after_loss_calculation,
      on_phase_end=on_phase_end,
      disable_plots=True,
      disable_batch_report=hvd.rank() != 0,
      num_batches=config.sys.steps_per_epoch,
      metrics = [
        ("text_acc", text_accuracy),
      ],
      start_at_epoch=start_epoch,
  )

  if hvd.rank() == 0:
    print("Saving model")
    torch.save(model.state_dict(), paths["model_path"])


def prep(config:cpb.AbstractGeneratorConfig):
  # all important paths
  paths = get_paths(config)
  # print("Downloading PMC")
  # with ftp_util.ftp_connect(
      # address="ftp.ncbi.nlm.nih.gov",
      # workdir="/pub/pmc/oa_bulk/",
  # ) as conn:
    # xml_paths = ftp_util.ftp_retreive_all(
        # conn=conn,
        # pattern="^.*\.xml\.tar\.gz$",
        # directory=paths["pmc_download_dir"],
        # show_progress=True,
    # )
  connect_to_dask_cluster(config)
  def ckpt(val, name, overwrite=False):
    print("Checkpoint", name)
    return dask_checkpoint.checkpoint(
        val,
        name=name,
        checkpoint_dir=paths["model_ckpt_dir"],
        overwrite=overwrite,
    )


  # Get the full set of abstracts
  abstracts = file_util.load(
      paths["checkpoint_dir"]
      .joinpath("medline_documents")
  )

  def all_text_fields_labeled(record:Record)->bool:
    for field in record["text_data"]:
      if field["type"] not in INTERESTING_SENTENCE_LABLES:
        return False
    return True

  interesting_abstracts = (
      abstracts
      # don't want the ones that are title-only
      .filter(lambda rec: len(rec["text_data"]) > 1)
      # Allow unlabeled abstracts
      #.filter(all_text_fields_labeled)
  )
  interesting_abstracts = ckpt(interesting_abstracts, "interesting_abstracts")

  is_test_data = (
      interesting_abstracts
      .map(lambda rec: (random.random() <= config.sys.test_ratio, rec))
  )
  is_test_data = ckpt(is_test_data, "is_test_data")

  testing_data = (
      is_test_data
      .filter(lambda b_r: b_r[0])
      .map(lambda b_r: b_r[1])
  )
  testing_data = ckpt(testing_data, "testing_data")

  training_data = (
      is_test_data
      .filter(lambda b_r: not b_r[0])
      .map(lambda b_r: b_r[1])
  )
  training_data = ckpt(training_data, "training_data")

  print("Collecting all mesh headings")
  all_mesh_headings = (
      training_data
      .map(lambda rec: rec["mesh_headings"])
      .flatten()
      .frequencies()
      .filter(lambda mesh_freq: mesh_freq[1] >= config.min_mesh_term_support)
      .map(lambda mesh_freq: mesh_freq[0])
      .compute()
  )
  print(f"Indexing all {len(all_mesh_headings)} mesh headings")
  mesh_index = items_to_ordered_index(all_mesh_headings)

  ###

  print("Getting oldest year")
  oldest_year = (
      training_data
      .filter(lambda rec: rec["date"] is not None)
      .map(lambda rec: int(rec["date"].split("-")[0]))
      .filter(lambda year: year > 1000)  # some invalid years are crazy
      .min()
      .compute()
  )
  print("\t-", oldest_year)

  ###

  print("Collecting training data for tokenizer")
  training_data_files = (
      training_data
      # Only collect 30% of abstracts
      .random_sample(0.3)
      .map_partitions(text_util.split_sentences)
      # Only need the text. We are doing a case-insensitive model.
      .map(lambda rec: rec["sent_text"])
      .map(lambda text: text.lower() if config.lowercase else text)
      # Only take 10% of sentences, ultimately,'re subsetting again
      .random_sample(0.1)
      # Reduce the total number of files
      .repartition(20)
      # Store results in textfiles
      .to_textfiles(f"{paths['tokenizer_training_data_dir']}/*.txt")
  )
  print("Training tokenizer")
  # need to place files in tokenizer_model_path
  spm.SentencePieceTrainer.train(
      f"--input={','.join(training_data_files)} "
      f"--model_prefix={paths['tokenizer_model_path'].parent}/tokenizer "
      f"--vocab_size={config.vocab_size} "
      f"--character_coverage=1.0 "
      f"--model_type=unigram "
      f"--input_sentence_size={config.max_tokenizer_sentences} "
      f"--shuffle_input_sentence=true "
  )
  assert paths["tokenizer_model_path"].is_file()
  assert paths["tokenizer_vocab_path"].is_file()

  extra_data = {
      "mesh_index": mesh_index,
      "oldest_year": oldest_year,
  }
  with open(paths["model_extra_data_path"], 'wb') as f:
    pickle.dump(extra_data, f)
  print("\t- Written:", paths["model_extra_data_path"])

  def abstracts_to_windows(records):
    tokenizer = AbstractGeneratorTokenizer(
      tokenizer_model_path = paths["tokenizer_model_path"],
      extra_data_path = paths["model_extra_data_path"],
    )
    generator = AbstractWindowGenerator(
      tokenizer=tokenizer,
      records=list(records),
      batch_size=config.sys.batch_size,
      text_size=config.text_length,
      lowercase=config.lowercase,
    )
    return list(generator.iterate_data_across_abstracts())

  ckpt(
    training_data.map_partitions(abstracts_to_windows),
    "training_data_windows",
    overwrite=True,
  )



if __name__ == "__main__":
  config = cpb.AbstractGeneratorConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)

  assert config.mode in MODES
  if config.mode == "prep":
    prep(config)
  if config.mode == "train":
    train(config)
  if config.mode == "evaluate":
    evaluate(config)
