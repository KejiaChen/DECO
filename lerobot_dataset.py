"""Local, single-camera LeRobot v2.1 reader for the repository training interface."""

from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml


DEFAULT_CACHE = Path(__file__).resolve().parent / ".cache/datasets/red_cup"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config/red_cup_data.yaml"


def load_reader_config(overrides=None):
    """Use the shared YAML settings, with optional per-dataset overrides."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())["data"]["reader"]
    config.update(overrides or {})
    return config


def _expect(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name}: expected {expected}, got {actual}")


def _validate_metadata(info, episodes, image_key, config):
    _expect(info["codebase_version"], config["codebase_version"], "codebase_version")
    feature_shapes = {
        image_key: config["image_shape"],
        "observation.state": config["state_shape"],
        "action": config["action_shape"],
    }
    for key, expected in feature_shapes.items():
        _expect(info["features"][key]["shape"], expected, f"{key} shape")
    _expect(len(episodes), info["total_episodes"], "Episode count")
    _expect(len({episode["episode_index"] for episode in episodes}), len(episodes), "Unique episode count")
    _expect(sum(episode["length"] for episode in episodes), info["total_frames"], "Total frame count")
    if info["fps"] <= 0 or info["chunks_size"] <= 0:
        raise ValueError("fps and chunks_size must be positive")
    if any(episode["length"] <= 0 for episode in episodes):
        raise ValueError("Empty episodes are unsupported")


def _write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    temporary.replace(path)


def source_checksums(paths, root):
    """Hash only source metadata, Parquet and videos; never write to the source."""
    checksums = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checksums[str(path.relative_to(root))] = digest.hexdigest()
    return checksums


def _source(data_dir, image_key, config):
    root = Path(data_dir).resolve()
    info_path = root / "meta/info.json"
    episodes_path = root / "meta/episodes.jsonl"
    tasks_path = root / "meta/tasks.jsonl"
    info = json.loads(info_path.read_text())
    candidates = [key for key, feature in info["features"].items()
                  if feature["dtype"] == "video"]
    if image_key is None:
        image_key, = candidates
    if image_key not in candidates:
        raise ValueError(f"Unknown video feature: {image_key}")
    episodes = sorted((json.loads(line) for line in episodes_path.read_text().splitlines()
                       if line.strip()), key=lambda item: item["episode_index"])
    _validate_metadata(info, episodes, image_key, config)
    tasks = {json.loads(line)["task_index"] for line in tasks_path.read_text().splitlines()
             if line.strip()}
    paths = [info_path, episodes_path, tasks_path]
    for episode in episodes:
        values = dict(episode_index=episode["episode_index"],
                      episode_chunk=episode["episode_index"] // info["chunks_size"],
                      video_key=image_key)
        paths.extend(root / info[key].format(**values) for key in ("data_path", "video_path"))
    checksums = source_checksums(paths, root)
    fingerprint = hashlib.sha256(json.dumps(checksums, sort_keys=True).encode()).hexdigest()
    return root, info, episodes, tasks, image_key, checksums, fingerprint


def episode_paths(root, info, episode_id, image_key):
    values = dict(episode_index=episode_id, episode_chunk=episode_id // info["chunks_size"],
                  video_key=image_key)
    return tuple(root / info[key].format(**values) for key in ("data_path", "video_path"))


def read_episode(root, info, episode, image_key, task_ids, config):
    path, _ = episode_paths(root, info, episode["episode_index"], image_key)
    table = pq.read_table(path)
    length = episode["length"]
    _expect(table.num_rows, length, f"Parquet length: {path}")
    arrays = {}
    feature_shapes = {"observation.state": config["state_shape"], "action": config["action_shape"]}
    for key, shape in feature_shapes.items():
        values = np.asarray(table[key].to_pylist(), dtype=np.float32)
        _expect(values.shape, (length, *shape), f"{key} shape: {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values for {key}: {path}")
        arrays[key] = values
    for key, expected in (("frame_index", np.arange(length)),
                          ("episode_index", np.full(length, episode["episode_index"]))):
        if not np.array_equal(table[key].to_numpy(), expected):
            raise ValueError(f"Invalid {key}: {path}")
    times = table["timestamp"].to_numpy()
    if not np.allclose(times, np.arange(length) / info["fps"], **config["timestamp_tolerance"]):
        raise ValueError(f"Invalid frame timestamps: {path}")
    tasks = table["task_index"].to_numpy().astype(np.int64)
    if not set(tasks.tolist()).issubset(task_ids):
        raise ValueError(f"Unknown task_index: {path}")
    return arrays["observation.state"], arrays["action"], tasks, times


def _valid_frame_cache(path, shape):
    if not path.is_file():
        return False
    try:
        frames = np.load(path, mmap_mode="r", allow_pickle=False)
        return frames.shape == shape and frames.dtype == np.uint8
    except (ValueError, OSError):
        return False


def _cache_video(video_path, output_path, shape, timestamps, config):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(dir=output_path.parent, suffix=".npy")
    os.close(descriptor)
    temporary = Path(filename)
    frames = None
    try:
        frames = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
        count = 0
        with av.open(str(video_path)) as container:
            for frame in container.decode(video=0):
                if count >= shape[0]:
                    raise ValueError(f"Video has more frames than Parquet: {video_path}")
                if frame.time is None or not np.isclose(frame.time, timestamps[count], **config["timestamp_tolerance"]):
                    raise ValueError(f"Video/Parquet timestamp mismatch: {video_path}, frame {count}")
                rgb = frame.to_ndarray(format="rgb24")
                _expect(rgb.shape, tuple(config["image_shape"]), f"Video dimensions: {video_path}")
                frames[count] = rgb
                count += 1
        if count != shape[0]:
            raise ValueError(f"Video has fewer frames than Parquet: {video_path}")
        frames.flush()
        del frames
        frames = None
        temporary.replace(output_path)
    finally:
        del frames
        temporary.unlink(missing_ok=True)


def prepare_lerobot_v21(data_dir, cache_dir=DEFAULT_CACHE, *, image_key=None,
                       split_seed=42, train_ratio=0.9, show_progress=False, reader=None):
    """Prepare shared RGB caches, an episode split and unpadded training-only stats."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    config = load_reader_config(reader)
    root, info, episodes, task_ids, image_key, checksums, fingerprint = _source(data_dir, image_key, config)
    cut = int(len(episodes) * train_ratio)
    if cut == 0 or cut == len(episodes):
        raise ValueError("Both training and validation must contain episodes")
    cache = Path(cache_dir).resolve()
    if cache == root or root in cache.parents:
        raise ValueError("Place the cache outside the read-only source dataset")
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = cache / "manifest.json"
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous and (previous["source_fingerprint"] != fingerprint or previous["image_key"] != image_key):
            raise ValueError("Cache belongs to different or modified source data; use a fresh cache_dir")
        labels = {}
        for episode in tqdm(episodes, desc="Prepare RGB episodes", disable=not show_progress):
            index = episode["episode_index"]
            labels[index] = read_episode(root, info, episode, image_key, task_ids, config)
            _, video = episode_paths(root, info, index, image_key)
            shape = (episode["length"], *config["image_shape"])
            output = cache / "frames" / f"episode_{index:06d}.npy"
            if not previous or not _valid_frame_cache(output, shape):
                _cache_video(video, output, shape, labels[index][3], config)
        if source_checksums([root / name for name in checksums], root) != checksums:
            raise RuntimeError("Source data changed while preparing the cache")
        manifest = dict(schema_version=1, source_root=str(root), source_fingerprint=fingerprint,
                        source_checksums=checksums, image_key=image_key, fps=info["fps"],
                        image_shape=info["features"][image_key]["shape"],
                        total_episodes=len(episodes), total_frames=info["total_frames"])
        _write_json(manifest_path, manifest)
        ids = [episode["episode_index"] for episode in episodes]
        random.Random(split_seed).shuffle(ids)
        train_ids, val_ids = sorted(ids[:cut]), sorted(ids[cut:])
        split = dict(source_fingerprint=fingerprint, split_seed=split_seed, train_ratio=train_ratio,
                     train_episode_indices=train_ids, val_episode_indices=val_ids,
                     train_frames=sum(len(labels[index][0]) for index in train_ids),
                     val_frames=sum(len(labels[index][0]) for index in val_ids))
        stats = dict(source_fingerprint=fingerprint, training_episode_indices=train_ids,
                     num_frames=split["train_frames"], std_ddof=config["std_ddof"],
                     std_epsilon=config["std_epsilon"])
        for name, column in (("observation", 0), ("action", 1)):
            values = np.concatenate([labels[index][column] for index in train_ids]).astype(np.float64)
            for suffix, result in (("mean", values.mean(0)), ("std", values.std(0, ddof=config["std_ddof"])),
                                   ("min", values.min(0)), ("max", values.max(0))):
                stats[f"{name}_{suffix}"] = result.tolist()
        partition = cache / "splits" / f"seed-{split_seed}-train-{train_ratio:g}"
        _write_json(partition / "split.json", split)
        _write_json(partition / "stats.json", stats)
    return dict(manifest=manifest, split=split, stats=stats, cache_dir=str(cache),
                split_path=str(partition / "split.json"), stats_path=str(partition / "stats.json"))


class LeRobotV21Dataset(Dataset):
    """Return the existing eight-field tuple; unavailable modalities are empty tensors.

    With a transform, image preprocessing is delegated to the training entrypoint.
    Without one, images are CHW float32 in [0, 1] at their original resolution.
    The mask is True for real action steps and False for repeated tail padding.
    """

    def __init__(self, data_dir, train=True, transform=None, *, cache_dir=DEFAULT_CACHE,
                 image_key=None, split_seed=42, train_ratio=0.9, chunk_size=32,
                 norm_type="mean_std", reader=None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if norm_type not in ("mean_std", "min_max"):
            raise ValueError("norm_type must be mean_std or min_max")
        self.reader = load_reader_config(reader)
        self.prepared = prepare_lerobot_v21(data_dir, cache_dir, image_key=image_key,
                                           split_seed=split_seed, train_ratio=train_ratio, reader=self.reader)
        root, info, episodes, task_ids, image_key, _, _ = _source(data_dir, image_key, self.reader)
        self.episode_indices = self.prepared["split"]["train_episode_indices" if train else "val_episode_indices"]
        self.cache_dir = Path(self.prepared["cache_dir"])
        self.transform = transform
        self.chunksize = chunk_size
        self.labels = {}
        self._ends = []
        total = 0
        episode_map = {episode["episode_index"]: episode for episode in episodes}
        for index in self.episode_indices:
            self.labels[index] = read_episode(root, info, episode_map[index], image_key, task_ids, self.reader)[:3]
            total += len(self.labels[index][0])
            self._ends.append(total)
        self._ends = np.asarray(self._ends)
        stats = self.prepared["stats"]
        epsilon = self.reader["std_epsilon"]
        self.obs_mean = torch.tensor(stats["observation_mean"], dtype=torch.float32)
        self.obs_std = torch.tensor(stats["observation_std"], dtype=torch.float32).clamp_min(epsilon)
        self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["action_std"], dtype=torch.float32).clamp_min(epsilon)
        self.norm_type = norm_type
        if norm_type == "min_max":
            self.obs_mean = torch.tensor(stats["observation_min"], dtype=torch.float32)
            self.obs_std = (torch.tensor(stats["observation_max"], dtype=torch.float32) - self.obs_mean).clamp_min(epsilon)
            self.action_mean = torch.tensor(stats["action_min"], dtype=torch.float32)
            self.action_std = (torch.tensor(stats["action_max"], dtype=torch.float32) - self.action_mean).clamp_min(epsilon)
        self._frames = OrderedDict()
        self._pid = os.getpid()

    def __len__(self):
        return int(self._ends[-1])

    def locate(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self._ends, index, side="right"))
        start = int(self._ends[position - 1]) if position else 0
        return self.episode_indices[position], index - start

    def _video(self, episode_id):
        if self._pid != os.getpid():
            self._frames = OrderedDict()
            self._pid = os.getpid()
        if episode_id not in self._frames:
            self._frames[episode_id] = np.load(self.cache_dir / "frames" / f"episode_{episode_id:06d}.npy",
                                              mmap_mode="r", allow_pickle=False)
            if len(self._frames) > self.reader["max_cached_episodes"]:
                self._frames.popitem(last=False)
        self._frames.move_to_end(episode_id)
        return self._frames[episode_id]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_frames"] = OrderedDict()
        return state

    def __getitem__(self, index):
        episode_id, frame = self.locate(index)
        rgb = np.array(self._video(episode_id)[frame], copy=True)
        if self.transform is not None:
            image = self.transform(Image.fromarray(rgb))
        else:
            image = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(np.iinfo(rgb.dtype).max)
        states, actions, tasks = self.labels[episode_id]
        state = torch.from_numpy(states[frame].copy())
        valid = min(self.chunksize, len(actions) - frame)
        positions = np.minimum(np.arange(frame, frame + self.chunksize), len(actions) - 1)
        action = torch.from_numpy(actions[positions].copy())
        state = (state - self.obs_mean) / self.obs_std
        action = (action - self.action_mean) / self.action_std
        mask = torch.arange(self.chunksize) < valid
        empty = torch.empty(0, dtype=torch.float32)
        return image, empty, empty.clone(), empty.clone(), state, action, mask, torch.tensor(tasks[frame], dtype=torch.long)
