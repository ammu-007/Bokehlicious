import random
import cv2
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from pathlib import Path
from json import load
from typing import Union, Optional
import torch

from dataset.util import Mode, calculate_aperture_embedding, generate_maps, build_input_dict, crop_to_divisible


class RealBokeh(Dataset):
    """
    Dataset class for the RealBokeh dataset.
        :param data_path: Path to the dataset directory with train/val/test subdirectories. Only test is needed for testing!
        :param mode: one of the modes defined in the Mode enum.
        :param binary_bokeh: whether to use binary bokeh strength of F22.0 and F2.0, RealBokeh_bin in the paper.
        :param defocus_deblur_mode: If true, reverses input and outputs, RealDefocus in the paper
    """
    def __init__(self, data_path: Union[str, Path], mode: Mode,
                 binary_bokeh: bool = False,
                 defocus_deblur_mode: bool = False,
                 device: str = 'cuda',
                 challenge: bool = False
                 ):
        self._data_path: Path = Path(data_path) if isinstance(data_path, str) else data_path
        assert self._data_path.exists(), f"Data directory {self._data_path.absolute()} does not exist!"

        self.defocus_deblur_mode = defocus_deblur_mode
        if self.defocus_deblur_mode:
            print("Dataset is in Defocus Deblur mode!")

        self._mode = mode
        self.challenge = challenge # Activate if used in the context of a challenge.
        # Iterate over individual samples of each scene for validation and test, over scenes where a random sample of a
        self._iteration_mode = 'sample'

        self._binary_bokeh = binary_bokeh
        if self._binary_bokeh:
            print("Using binary bokeh strength of F22.0 and F2.0!")
            self._iteration_mode = 'scene'

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # load metadata list
        self._scene_list = sorted([load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")],
                                  key=lambda x: x['id'])
        if len(self._scene_list) == 0:
            raise FileNotFoundError(f"No metadata files found in {self._mode_dir.joinpath('metadata')}!")

        self._sample_list = []
        for scene_id, metadata in enumerate(self._scene_list):
            for sample_id in range(len(metadata['target_images'])):
                self._sample_list.append((scene_id, sample_id))

        self._device = device

        print(f"RealBokeh Dataloader initialized in {mode} mode with {len(self._scene_list)} scenes and {len(self._sample_list)} samples!")

    def __len__(self):
        # Since for Validation and Test we are iterating over every sample of every scene, we need to return the
        # length of the sample list, otherwise we return the length of the scene list.
        if self._iteration_mode == 'sample':
            return len(self._sample_list)
        else:
            return len(self._scene_list)

    def __getitem__(self, index: int):
        if self._iteration_mode == 'sample':
            metadata: dict = self._scene_list[self._sample_list[index][0]]

            target = Image.open(self._mode_dir.joinpath(metadata['target_images'][self._sample_list[index][1]])) if not self.challenge else None
            tgt_av = metadata['target_avs'][self._sample_list[index][1]]

            target_name = Path(metadata['target_images'][self._sample_list[index][1]]).stem
        else:
            metadata: dict = self._scene_list[index]

            target_idx = 0
            target = Image.open(self._mode_dir.joinpath(metadata['target_images'][target_idx])) if not self.challenge else None
            tgt_av = metadata['target_avs'][target_idx]

            target_name = Path(metadata['target_images'][target_idx]).stem

        source = Image.open(self._mode_dir.joinpath(metadata['source_image']))

        if self.defocus_deblur_mode:
            tmp = source
            source = target
            target = tmp

        # Embed metadata and generate auxiliary maps

        aperture_embedding = calculate_aperture_embedding(tgt_av)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(maps, aperture_embedding, target_name, self._device)

        return return_dict

class EBB(Dataset):

    def __init__(self, data_path: Union[str, Path], mode: Mode, device: str = 'cuda'):
        self._data_path: Path = Path(data_path) if isinstance(data_path, str) else data_path
        self._mode = mode
        self._device = device

        # initialize dir
        self._mode_dir = self._data_path.joinpath(self._mode.value)
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Mode directory {self._mode_dir} does not exist!")

        # Initialize input and gt lists
        self._image_list = sorted([Path(f) for f in self._mode_dir.joinpath("in").glob("*.jpg")], key=lambda x: x.stem)
        self._gt_list = sorted([Path(f) for f in self._mode_dir.joinpath("gt").glob("*.jpg")], key=lambda x: x.stem)

        if len(self._image_list) == 0:
            raise FileNotFoundError(f"No images found in {self._mode_dir.joinpath('in')}!")
        if len(self._gt_list) == 0:
            raise FileNotFoundError(f"No images found in {self._mode_dir.joinpath('gt')}!")
        if len(self._image_list) != len(self._gt_list):
            raise ValueError(f"Number of images and gt images do not match in {self._mode_dir}, {len(self._image_list)} inputs != {len(self._gt_list)} gts!")

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, index: int):
        source = crop_to_divisible(Image.open(self._image_list[index]), divisor=4)
        target = crop_to_divisible(Image.open(self._gt_list[index]), divisor=4)

        aperture_embedding = calculate_aperture_embedding(2.0)

        maps = generate_maps(source, aperture_embedding, target=target)

        return_dict = build_input_dict(maps, aperture_embedding, self._image_list[index].name, self._device)

        return return_dict


class RealBokehTrain(Dataset):
    """
    Training dataset class for RealBokeh.
    Returns random crops and applies horizontal flip augmentation.
    Compatible with torch.utils.data.DataLoader (no per-item .cuda() call).

    Args:
        data_path: Path to the RealBokeh_3MP directory (must contain a 'train' subdirectory).
        patch_size: Size of the random square crop (paper: 512).
        binary_bokeh: If True, only use f/2.0 target images (RealBokeh_bin variant).
    """

    def __init__(self, data_path: Union[str, Path], patch_size: int = 512,
                 binary_bokeh: bool = False):
        self._data_path = Path(data_path) if isinstance(data_path, str) else data_path
        assert self._data_path.exists(), f"Data directory {self._data_path.absolute()} does not exist!"

        self._patch_size = patch_size
        self._binary_bokeh = binary_bokeh

        self._mode_dir = self._data_path / Mode.TRAIN.value
        if not self._mode_dir.exists():
            raise FileNotFoundError(f"Train directory {self._mode_dir} does not exist!")

        # Load all scene metadata
        self._scene_list = sorted(
            [load(open(f)) for f in self._mode_dir.joinpath("metadata").glob("*.json")],
            key=lambda x: x['id']
        )
        if len(self._scene_list) == 0:
            raise FileNotFoundError(f"No metadata files found in {self._mode_dir / 'metadata'}!")

        # Build flat sample list: (scene_metadata, target_index)
        self._sample_list = []
        for metadata in self._scene_list:
            for sample_id, tgt_av in enumerate(metadata['target_avs']):
                if self._binary_bokeh and abs(float(tgt_av) - 2.0) > 0.1:
                    continue  # Only keep f/2.0 targets for binary variant
                self._sample_list.append((metadata, sample_id))

        print(f"RealBokehTrain: {len(self._scene_list)} scenes, "
              f"{len(self._sample_list)} training samples, "
              f"patch_size={patch_size}")

    def __len__(self):
        return len(self._sample_list)

    def __getitem__(self, index: int):
        metadata, sample_id = self._sample_list[index]

        source = Image.open(self._mode_dir / metadata['source_image']).convert('RGB')
        target = Image.open(self._mode_dir / metadata['target_images'][sample_id]).convert('RGB')
        tgt_av = metadata['target_avs'][sample_id]

        # --- Random crop ---
        # Ensure both source and target are cropped at the same location
        w, h = source.size
        p = self._patch_size
        if w < p or h < p:
            # Upscale if image is smaller than patch (rare edge case)
            scale = max(p / w, p / h) + 0.01
            new_w, new_h = int(w * scale), int(h * scale)
            source = source.resize((new_w, new_h), Image.Resampling.BICUBIC)
            target = target.resize((new_w, new_h), Image.Resampling.BICUBIC)
            w, h = source.size

        left = random.randint(0, w - p)
        top = random.randint(0, h - p)
        # PIL.Image.crop takes (left, upper, right, lower)
        box = (left, top, left + p, top + p)
        source = source.crop(box)
        target = target.crop(box)

        # --- Random horizontal flip ---
        if random.random() > 0.5:
            source = ImageOps.mirror(source)
            target = ImageOps.mirror(target)

        # --- Aperture encoding and auxiliary maps ---
        aperture_embedding = calculate_aperture_embedding(tgt_av)
        maps = generate_maps(source, aperture_embedding, target=target)

        # Build dict — device='cpu' so DataLoader workers don't fight over CUDA
        sample = build_input_dict(maps, aperture_embedding,
                                  Path(metadata['target_images'][sample_id]).stem,
                                  device='cpu')
        return sample


class RealBokehParquet(Dataset):
    """
    Training/validation dataset that reads from HuggingFace parquet files.

    Actual parquet schema (one row = one image):
        image: struct<bytes: binary, path: string>
        path format: "{scene_id}_f{aperture}.JPG"
        source image: path ends with "f20.JPG"  (f/22 -- wide aperture, sharp)
        target images: all other f-stops (f/2.0, f/4.0, etc.)

    The class groups rows by scene_id and pairs each source with its targets.

    Args:
        parquet_dir: Directory containing .parquet files for one split.
        patch_size:  Random crop size. None = full image (for validation).
        augment:     Whether to apply random horizontal flip.
    """

    def __init__(self, parquet_dir: Union[str, Path],
                 patch_size: Optional[int] = 512,
                 augment: bool = True):
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
        except ImportError:
            raise ImportError("pyarrow is required. Install with: pip install pyarrow")

        self._parquet_dir = Path(parquet_dir)
        self._patch_size = patch_size
        self._augment = augment

        parquet_files = sorted(self._parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {self._parquet_dir.absolute()}")

        print(f"RealBokehParquet: loading {len(parquet_files)} parquet file(s) from {self._parquet_dir}")
        tables = [pq.read_table(str(f)) for f in parquet_files]
        self._table = pa.concat_tables(tables)
        print(f"  Total rows (individual images): {len(self._table)}")

        # Build path -> row_index lookup for fast image retrieval
        paths = [self._table['image'][i].as_py()['path'] for i in range(len(self._table))]
        self._path_to_idx = {p: i for i, p in enumerate(paths)}

        # Group by scene_id, separate source (f20=f/22) from targets
        # path format: "{scene_id}_f{aperture}.JPG"
        from collections import defaultdict
        scenes = defaultdict(dict)
        for path in paths:
            stem = path.replace('.JPG', '').replace('.jpg', '')
            parts = stem.split('_')
            scene_id = parts[0]
            f_tag = '_'.join(parts[1:])  # e.g. "f2.0", "f20"
            scenes[scene_id][f_tag] = path

        # Build flat sample list: (source_path, target_path, target_av)
        self._samples = []
        for scene_id, images in scenes.items():
            source_path = images.get('f20')
            if source_path is None:
                continue
            for f_tag, target_path in images.items():
                if f_tag == 'f20':
                    continue
                try:
                    av = float(f_tag.lstrip('f'))
                except ValueError:
                    continue
                self._samples.append((source_path, target_path, av))

        print(f"  Scenes: {len(scenes)}, Training pairs: {len(self._samples)}")

    @staticmethod
    def _find_col(names: list, candidates: list) -> str:
        for c in candidates:
            if c in names:
                return c
        raise KeyError(f"Could not find any of {candidates} in parquet columns: {names}")

    def __len__(self):
        return len(self._samples)

    def _load_image(self, path: str) -> Image.Image:
        import io
        idx = self._path_to_idx[path]
        cell = self._table['image'][idx].as_py()
        return Image.open(io.BytesIO(cell['bytes'])).convert('RGB')

    def __getitem__(self, index: int):
        source_path, target_path, tgt_av = self._samples[index]

        source = self._load_image(source_path)
        target = self._load_image(target_path)

        # Random crop
        if self._patch_size is not None:
            w, h = source.size
            p = self._patch_size
            if w < p or h < p:
                scale = max(p / w, p / h) + 0.01
                new_w, new_h = int(w * scale), int(h * scale)
                source = source.resize((new_w, new_h), Image.Resampling.BICUBIC)
                target = target.resize((new_w, new_h), Image.Resampling.BICUBIC)
                w, h = source.size
            left = random.randint(0, w - p)
            top  = random.randint(0, h - p)
            box  = (left, top, left + p, top + p)
            source = source.crop(box)
            target = target.crop(box)

        # Random horizontal flip
        if self._augment and random.random() > 0.5:
            source = ImageOps.mirror(source)
            target = ImageOps.mirror(target)

        aperture_embedding = calculate_aperture_embedding(tgt_av)
        maps = generate_maps(source, aperture_embedding, target=target)
        stem = Path(target_path).stem
        return build_input_dict(maps, aperture_embedding, stem, device='cpu')
