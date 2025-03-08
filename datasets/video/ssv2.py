import json
from pathlib import Path
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from torchvision.datasets.video_utils import _VideoTimestampsDataset, _collate_fn
from datasets.video.base_video import BaseVideoDataset, SPLIT
from datasets.video.utils import read_video  # Assumes your read_video can handle webm videos
from datasets.video.base_video import BaseAdvancedVideoDataset

class SomethingSomethingV2Dataset(BaseVideoDataset):
    """
    Base dataset for Something-Something V2.
    This implementation assumes:
      - All videos (in .webm format, vp9 codec) are stored in one directory.
      - Annotations for train/val/test splits are provided as separate JSON files,
        with one JSON object per line.
      - Each JSON object has keys such as "id", "label", "template", and "placeholders".
    """
    def download_dataset(self) -> None:
        # We assume the dataset is already downloaded and stored locally.
        print("Using local Something-Something V2 dataset. Please verify that videos and annotation files exist.")

    def build_metadata(self, split: SPLIT) -> None:
        # Expecting self.cfg.annotations_paths to be a dict mapping splits to JSON file paths.
        ann_path = Path(self.cfg.annotation_paths[split])
        if not ann_path.exists():
            raise ValueError(f"Annotation file {ann_path} does not exist!")
        
        # Read the annotations (each line is a JSON object)
        with open(ann_path, "r") as f:
            annotations = json.load(f)
        
        # Build video file paths by appending .webm to each id
        video_paths = []
        for ann in annotations:
            video_file = Path(self.cfg.videos_dir) / f"{ann['id']}.webm"
            if not video_file.exists():
                print(f"Warning: video file {video_file} not found!")
            video_paths.append(str(video_file))
        
        # Use _VideoTimestampsDataset to extract frame timestamps and FPS
        dl = torch.utils.data.DataLoader(
            _VideoTimestampsDataset(video_paths),
            batch_size=16,
            num_workers=4,
            collate_fn=_collate_fn,
        )
        video_pts = []
        video_fps = []
        for batch in tqdm(dl, desc=f"Building metadata for {split}"):
            batch_pts, batch_fps = list(zip(*batch))
            video_pts.extend([torch.as_tensor(pts, dtype=torch.long) for pts in batch_pts])
            video_fps.extend(batch_fps)
        
        # Save all relevant metadata, including the annotations for conditioning.
        metadata = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
            "annotations": annotations,  # Contains keys: id, label, template, placeholders, etc.
        }
        self.metadata_dir.mkdir(exist_ok=True, parents=True)
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

# Simple version if needed
class SomethingSomethingV2SimpleDataset(SomethingSomethingV2Dataset):
    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        super().__init__(cfg, split)
        # Additional filtering or transforms could be added here.

# Advanced dataset with conditioning based on the label
class SomethingSomethingV2AdvancedDataset(SomethingSomethingV2Dataset, BaseAdvancedVideoDataset):
    """
    Advanced dataset that loads clips (with frame skipping, etc.) and also returns a conditioning tensor
    computed from the video’s label.
    """
    def __init__(self, cfg: DictConfig, split: SPLIT = "training", current_epoch: int = None):
        # For consistency with similar datasets, we map 'validation' to 'test'
        if split == "validation":
            split = "test"
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

    def load_cond(self, video_metadata: dict, start_frame: int, end_frame: int) -> torch.Tensor:
        # Get the annotation for this video
        ann = video_metadata["annotations"]
        label = ann.get("label", "")
        
        return label

