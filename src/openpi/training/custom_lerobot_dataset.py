from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_safe_version, get_episode_data_index, check_timestamps_sync, check_delta_timestamps, get_delta_indices
from lerobot.common.datasets.video_utils import VideoFrame
import torch
import random
import logging
import os
from pathlib import Path
from typing import Callable
import datasets

class CustomLeRobotDataset(LeRobotDataset):
    """
    A custom extension of LeRobotDataset for progress estimation and value function training.
    
    This class extends LeRobotDataset with additional features:
    - History frame sampling (n_history)
    - Episode start frame inclusion (with_episode_start)
    - Timestep difference mode for contrastive learning
    - Stage progress supervision mode
    - Skip sampling within episodes
    """
    
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = 'pyav',
        
        # * Custom parameters
        n_history: int = 0,
        with_episode_start: bool = False,
        skip_sample_ratio_within_episode: float = 0.,  # * 0 for no skipping, 0.5 for skipping first 50% samples in an episode
        timestep_difference_mode: bool = False,  # * Selecting samples from two different timesteps and do comparison as learning target.
        stage_process_mode: bool = False,  # * Using stage progress supervision.
    ):
        """
        Initialize CustomLeRobotDataset.
        
        Args:
            repo_id: Repository ID for the dataset
            root: Local directory for dataset files
            episodes: List of episode indices to load
            image_transforms: Image transformation functions
            delta_timestamps: Delta timestamps configuration
            tolerance_s: Timestamp tolerance in seconds
            revision: Git revision/version
            force_cache_sync: Force sync with remote cache
            download_videos: Whether to download video files
            video_backend: Video decoding backend
            
            # Custom args:
            n_history: Number of history frames to include
            with_episode_start: Include episode start frame
            skip_sample_ratio_within_episode: Ratio of samples to skip at episode start
            timestep_difference_mode: Enable timestep difference learning
            stage_process_mode: Enable stage progress supervision
        """
        # 初始化继承自LeRobotDataset的初始化参数
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend
        )
        # Store custom parameters before calling parent __init__
        self.n_history = n_history
        self.with_episode_start = with_episode_start
        self.skip_sample_ratio_within_episode = skip_sample_ratio_within_episode
        self.timestep_difference_mode = timestep_difference_mode
        self.stage_process_mode = stage_process_mode
        
        # Validation
        assert self.skip_sample_ratio_within_episode <= 0.5
        if self.timestep_difference_mode:
            assert not self.with_episode_start, "Cannot use episode start when using timestep difference mode."

        # * Custom: Episode index to array index mapping
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(episodes)} if episodes else {}
    
    def get_sample_with_imgs_from_idx(self, idx: int) -> dict:
        """
        Get a sample with video frames decoded from the given index.
        This is a helper method used by __getitem__ for fetching individual samples.
        """
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
            query_indices, padding = self._get_query_indices(idx, arr_idx)
        
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            
            try:
                video_frames = self._query_videos(query_timestamps, ep_idx)
            except Exception as e:
                print(f"Error decoding video frames for ep_idx {ep_idx} at idx {idx}: {e}")
                import pdb; pdb.set_trace()
            
            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        
        return item
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get item with custom features for progress estimation.
        
        Returns a dict containing:
        - Base sample data
        - History frames (if n_history > 0)
        - Episode start frame (if with_episode_start)
        - Random timestep comparison frame (if timestep_difference_mode)
        - Progress labels
        """
        episode_level_dict = {}
        item_sequence = []
        final_item = {}
        
        # Get main sample
        item = self.get_sample_with_imgs_from_idx(idx)
        
        ep_idx = item["episode_index"].item()
        cur_timestamp = item["timestamp"].item()
        
        _EP_IDX = ep_idx
        _CUR_TIMESTAMP = cur_timestamp
        
        episode_level_dict['episode_length'] = self.meta.episodes[ep_idx]["length"]
        
        # Handle delta indices for action sequences
        if self.delta_indices is not None:
            episode_level_dict = self.handle_delta_indices(idx, ep_idx, episode_level_dict)
        
        episode_level_dict['action_advantage'] = item.get('action_advantage', None)
        
        # Add task as a string
        task_idx = item["task_index"].item()
        episode_level_dict["task"] = self.meta.tasks[task_idx]
        
        item_sequence.append(item.copy())
        
        # Handle timestep difference mode
        if self.timestep_difference_mode:
            final_item = self.handle_timestep_difference_mode(idx, ep_idx, final_item,_EP_IDX, _CUR_TIMESTAMP)
        
        # Handle episode start frame
        if self.with_episode_start:
            final_item = self.handle_episode_start_frame(idx, ep_idx, item, final_item)
        
        # Handle history frames
        N_HISTORY, item_sequence = self.handle_history_frames(idx, ep_idx,  
                                                              cur_timestamp=cur_timestamp,
                                                              item_sequence=item_sequence
                                                              )
        # Unpack the item_sequence
        for his_idx in range(-N_HISTORY, 0):
            his_pos_in_sequence = his_idx - 1
            his_item = item_sequence[his_pos_in_sequence]
            
            _keys = list(his_item.keys())
            for key in _keys:
                new_key = f"his_{his_idx}_{key}"
                his_item[new_key] = his_item.pop(key)
                final_item = {**final_item, **his_item}
        
        final_item = {**final_item, **item_sequence[-1]}
        
        # Append episode-level data
        final_item = {**final_item, **episode_level_dict}
        
        # Compute progress labels
        if self.timestep_difference_mode:
            if self.stage_process_mode:
                stage_progress_gt_random = final_item[f"his_-100_stage_progress_gt"].item()
                stage_progress_gt = final_item[f"stage_progress_gt"].item()
                final_item['progress'] = stage_progress_gt - stage_progress_gt_random
            else:
                progress_gt_random = final_item[f"his_-100_progress_gt"].item()
                progress_gt = final_item[f"progress_gt"].item()
                final_item['progress'] = progress_gt - progress_gt_random
        else:
            if self.stage_process_mode:
                stage_progress_gt = final_item[f"stage_progress_gt"].item()
                final_item['progress'] = stage_progress_gt
            else:
                progress_gt = final_item[f"progress_gt"].item()
                final_item['progress'] = progress_gt
        
        return final_item

    def handle_delta_indices(self, idx, ep_idx, episode_level_dict) -> dict:
        query_indices = None
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        query_indices, padding = self._get_query_indices(idx, arr_idx)
        query_result = self._query_hf_dataset(query_indices)
        
        episode_level_dict = {**episode_level_dict, **padding}
        for key, val in query_result.items():
            episode_level_dict[key] = val
        return episode_level_dict
    
    def handle_timestep_difference_mode(self, idx, ep_idx, final_item,_EP_IDX, _CUR_TIMESTAMP) -> dict:
        """
        同一个视频片段里的不同两帧？
        """
        random_timestep_name = -100
        arr_idx = self.ep_idx_to_arr_idx.get(ep_idx, ep_idx) if self.episodes else ep_idx
        ep_start_idx = self.episode_data_index["from"][arr_idx].item()
        ep_end_idx = self.episode_data_index["to"][arr_idx].item()
        while True:
            random_idx = random.randint(ep_start_idx, ep_end_idx - 1)
            if random_idx == idx:
                continue
            
            random_item = self.get_sample_with_imgs_from_idx(random_idx)
            
            ep_idx_check = random_item["episode_index"].item()
            cur_timestamp_check = random_item["timestamp"].item()
            if ep_idx_check != _EP_IDX or cur_timestamp_check == _CUR_TIMESTAMP:
                print(f"Randomly selected invalid timestep, re-sampling. For global idx: {random_idx}, ep_idx: {ep_idx_check}, cur_timestamp: {cur_timestamp_check}")
                continue
            break
        
        _keys = list(random_item.keys())
        for key in _keys:
            new_key = f"his_{random_timestep_name}_{key}"
            random_item[new_key] = random_item.pop(key)

        final_item = {**final_item, **random_item}
        return final_item

    def handle_episode_start_frame(self, idx, ep_idx, item,final_item) -> dict:
        start_frame_name = -100
        
        cur_frame_index = item['frame_index'].item()
        start_index = idx - cur_frame_index
        cur_progress_gt = item['progress_gt'].item()
        if cur_progress_gt == 0:
            start_index = idx
        start_item = self.get_sample_with_imgs_from_idx(start_index)
        start_episode_ind = start_item['episode_index'].item()
        start_progress_gt = start_item['progress_gt'].item()
        
        while start_episode_ind != ep_idx or start_progress_gt != 0:
            start_index += 1
            start_item = self.get_sample_with_imgs_from_idx(start_index)
            start_episode_ind = start_item['episode_index'].item()
            start_progress_gt = start_item['progress_gt'].item()
        
        _keys = list(start_item.keys())
        for key in _keys:
            new_key = f"his_{start_frame_name}_{key}"
            start_item[new_key] = start_item.pop(key)
        
        final_item = {**final_item, **start_item}
        return final_item
    
    def handle_history_frames(self, idx, ep_idx, cur_timestamp, item_sequence) -> dict:
        N_HISTORY = self.n_history
        for his_idx in range(-N_HISTORY, 0):
            check_his_idx = his_idx + idx
            check_item = self.hf_dataset[check_his_idx]
            check_episode_ind = check_item['episode_index'].item()
            check_timestamp = check_item['timestamp'].item()
            
            if check_his_idx >= 0 and check_episode_ind == ep_idx and check_timestamp < cur_timestamp:
                his_item = check_item
            else:
                # idx is the start of a new episode, repeat it as its history
                his_item = self.get_sample_with_imgs_from_idx(idx)
            
            his_episode_ind = his_item['episode_index'].item()
            his_timestamp = his_item['timestamp'].item()
            assert his_episode_ind == ep_idx and his_timestamp <= cur_timestamp, \
                f"his_item['episode_index']: {his_episode_ind}, ep_idx: {ep_idx}. his_item['timestamp']: {his_timestamp}, cur_timestamp: {cur_timestamp}"
            
            item_sequence.insert(-1, his_item.copy())  # Note the inserting position
        return N_HISTORY, item_sequence

class CustomMultiLeRobotDataset(MultiLeRobotDataset,torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `CustomLeRobotDataset`s.

    The underlying `CustomLeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        
        **custom_kwargs,
    ):
        # super().__init__()
        torch.utils.data.Dataset.__init__(self)
        self.repo_ids = repo_ids
        
        # Use more relaxed tolerance
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.1)
        
        self._datasets = [
            CustomLeRobotDataset(
                repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=del_ts,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
                **custom_kwargs
            )
            for repo_id, del_ts in zip(repo_ids, delta_timestamps)
        ]
        
        # Disable any data keys that are not common across all of the datasets
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if len(extra_keys) > 0:
                logging.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)
        
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
