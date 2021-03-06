from pathlib import Path
from typing import Callable, Any, Tuple, Optional, List
import dask.bag as dbag
import json
from dask.distributed import Lock
import string
import random
import pickle
import dask
from tqdm import tqdm

EXT = ".pkl"
DONE_FILE = "__done__"

def get_random_ascii_str(str_len:int)->str:
  return "".join([random.choice(string.ascii_letters) for _ in range(str_len)])
import pickle

def copy_to_local_scratch(src:Path, local_scratch_dir:Path)->Path:
  local_scratch_dir.mkdir(parents=True, exist_ok=True)
  assert src.is_file()
  assert local_scratch_dir.is_dir()
  dest  = local_scratch_dir.joinpath(src.name)
  dest.write_bytes(src.read_bytes())
  return dest

def prep_scratches(
    local_scratch_root:Path,
    shared_scratch_root:Path,
    task_name:str,
)->Tuple[Path, Path]:
  local = local_scratch_root.joinpath(task_name)
  shared = shared_scratch_root.joinpath(task_name)
  local.mkdir(parents=True, exist_ok=True)
  shared.mkdir(parents=True, exist_ok=True)
  return local, shared

def load_to_memory(dir_path:Path, disable_pbar:bool=False)->List[Any]:
  "Performs loading right now, without dask"
  assert is_result_saved(dir_path)
  result = []
  for path in tqdm(get_part_files(dir_path), disable=disable_pbar):
    if path.is_file():
      result += load_part(path)
    else:
      raise Exception(f"Invalid stored bag {dir_path}. Missing {path}.")
  return result


def get_part_files(dir_path:Path)->List[Path]:
  assert is_result_saved(dir_path)
  with open(dir_path.joinpath(DONE_FILE)) as f:
    return [Path(line.strip()) for line in f]


def load_part(path:Path, allow_failure:bool=False)->List[Any]:
  try:
    with open(path, 'rb') as f:
      return pickle.load(f)
  except Exception as e:
    print("Encountered an issue with", path)
    if allow_failure:
      return []
    else:
      raise e

def load(dir_path:Path, allow_failure:bool=False)->dbag.Bag:
  assert is_result_saved(dir_path)
  done_path = dir_path.joinpath(DONE_FILE)
  load_tasks = []
  with open(done_path) as f:
    for line in f:
      path = Path(line.strip())
      load_tasks.append(dask.delayed(load_part)(path, allow_failure))
  return dbag.from_delayed(load_tasks)


def save_part(part:List[Any], path:Path)->Path:
  "Stores that partition at `path`, returns `path`"
  # Turns out that some complex functions might have non-list partitions
  # For instance, a partition containing numpy arrays will be _MapChunk
  part = list(part)
  with open(path, 'wb') as f:
    pickle.dump(part, f, protocol=4)
  return path


def write_done_file(parts:List[str], part_dir:Path)->Path:
  done_path = part_dir.joinpath(DONE_FILE)
  with open(done_path, 'w') as f:
    for part in parts:
      f.write(f"{part}\n")
  return done_path

def save(bag:dbag.Bag, path:Path, keep_partial_result:bool=False)->dask.delayed:
  path.mkdir(parents=True, exist_ok=True)
  save_tasks = []
  for part_idx, part in enumerate(bag.to_delayed()):
    part_path = path.joinpath(f"part-{part_idx}{EXT}")
    # if the partial result is not present, or we're not keeping partials
    if not part_path.is_file() or not keep_partial_result:
      save_tasks.append(dask.delayed(save_part)(part, part_path))
    else:
      # introduces a no-op that keeps __done__ file correct
      save_tasks.append(dask.delayed(part_path))
  return dask.delayed(write_done_file)(save_tasks, path)

def is_result_saved(path:Path)->bool:
  done_path = path.joinpath(DONE_FILE)
  return done_path.is_file()

def touch_random_unused_file(base_dir:Path, ext:Optional[str]=None)->Path:
  assert base_dir.is_dir()
  if ext is None:
    ext = ""
  elif ext[0] != ".":
    ext = f".{ext}"
  lock = Lock(f"dir_lock:{base_dir.name}")
  while(not lock.acquire(timeout=5)):
    pass
  # THREADSAFE
  name = f"{get_random_ascii_str(10)}{ext}"
  path = base_dir.joinpath(name)
  while path.is_file():
    name = f"{get_random_ascii_str(10)}{ext}"
    path = base_dir.joinpath(name)
  path.touch()
  # End Threadsafe
  lock.release()
  return path


def save_value(value:Any, path:Path)->None:
  """
  Saves an arbitrary object.
  """
  with open(path, 'wb') as f:
    pickle.dump(value, f)


def load_value(path:Path)->Any:
  """
  Loads an arbitrary object.
  """
  with open(path, 'rb') as f:
    return pickle.load(f)

def load_random_sample_to_memory(
    data_dir:Path,
    value_sample_rate:float=1,
    partition_sample_rate:float=1,
    disable_pbar:bool=False,
)->List[Any]:
  assert value_sample_rate > 0
  assert value_sample_rate <= 1
  assert partition_sample_rate > 0
  assert partition_sample_rate <= 1
  assert is_result_saved(data_dir)
  res = []
  for part in tqdm(get_part_files(data_dir), disable=disable_pbar):
    if random.random() < partition_sample_rate:
      for rec in load_part(part):
        if random.random() < value_sample_rate:
          res.append(rec)
  return res
