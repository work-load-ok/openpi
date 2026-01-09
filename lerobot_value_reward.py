"""
由 01_modify_task_index_based_on_progress.py 分析判断得到如下规则：
reward计算规则：
1. 如果advantage_source为"absolute_advantage"或"relative_advantage"，直接取*.parquet文件中的对应列作为奖励；
2. 如果advantage_source为"progress"，则按照以下规则计算奖励：
    - 如果i+chunk_size未超出范围，则直接计算progress[i+chunk_size] - progress[i]
    - 如果i+chunk_size超出范围，则用最后一个值计算，并进行数值调整，即rewards[i] = (progress[-1] - progress[i]) / (len(progress) - i) * chunk_size
    - progress为*.parquet文件中的progress列，chunk_size为预看帧数，默认为50，用于计算当前帧与未来chunk_size帧的progress差值，作为reward值

操作执行 Advantage 划分规则
1.包含Advantage: negative和Advantage: positive，分别对应0和1
2.negative与positive的划分依据为：
    - 划分阈值（threshold）计算：
        - 计算数据集中所有操作帧的reward
        - 计算上步中所有reward的N%分位数作为划分阈值，N为划分阈值百分比，默认为70
        - 如果任务具有多阶段，则需要计算每个阶段的划分阈值；每个阶段按照该阶段的阈值进行划分
    - Advantage: negative: reward < threshold
    - Advantage: positive: reward >= threshold
------------------------------------------------------------------------------------------------
!!!注意：目前reward的计算，均假设只有1个任务!!!
!!!注意：目前仅划分advantage: negative和advantage: positive，未考虑其他划分方式!!!
"""
from ctypes import Array
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import pyarrow.parquet as pq
import pyarrow as pa

def calculate_rewards(advantages: np.ndarray, chunk_size: int = 50, advantage_source: str = "progress") -> np.ndarray:
    """
    Calculate rewards based on progress differences.
    
    Args:
        data: DataFrame containing 'progress' column
        chunk_size: Number of frames to look ahead for progress calculation
        
    Returns:
        Array of rewards for each frame
    """
    assert advantage_source in ["absolute_advantage", "relative_advantage", "progress"], \
        f"Unknown advantage source, should be in [absolute_advantage, relative_advantage, progress], but got {advantage_source}"
    if advantage_source in ["absolute_advantage", "relative_advantage"]:
        return advantages
    elif advantage_source == "progress":
        n_frames = len(advantages)
        rewards = np.zeros(n_frames, dtype=np.float32)
        # 计算规则：progress[i+chunk_size] - progress[i]；如果i+chunk_size超出范围，则用最后一个值计算，并进行数值调整
        rewards[:-chunk_size] = advantages[chunk_size:] - advantages[:-chunk_size]
        rewards[-chunk_size:] = (advantages[-1] - advantages[-chunk_size:]) / np.linspace(chunk_size, 1, chunk_size)* chunk_size
    return rewards

# 更新episode_stats.jsonl，补充stats字段额外
def compute_reward_statistics(rewards: List[float], percentiles: List[int] = list(range(0, 101, 10))) -> dict:
    """
    Compute reward distribution statistics.
    
    Args:
        rewards: List of all rewards
        
    Returns:
        Dictionary containing percentile information
    """
    if len(rewards) == 0:  # 如果奖励列表为空，则返回0
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'percentiles': {p: 0.0 for p in percentiles}
        }
    
    rewards_array = np.array(rewards)
    
    # Compute percentiles points
    percentile_values = np.percentile(rewards_array, percentiles)
    
    stats = {
        'mean': [np.mean(rewards_array)],
        'std': [np.std(rewards_array)],
        'min': [np.min(rewards_array)],
        'max': [np.max(rewards_array)],
        'percentiles': dict(zip(percentiles, percentile_values)),
        'count': [len(rewards_array)]
    }
    return stats

def compute_threshold_points(rewards: Dict[int, List[float] | np.ndarray], positive_rate: float = 30) -> float | List[float]:
    """
    Compute threshold points based on positive rate.
    Args:
        rewards: List of all rewards
        positive_rate: Positive rate of the task, default is 30%
    Returns:
        Threshold points
    """
    if len(rewards) == 1:
        return np.percentile(rewards[0], 100 - positive_rate)
    else:
        return [np.percentile(rewards[i], 100 - positive_rate) for i in range(len(rewards))]

def collect_all_rewards(parquet_path: Path, chunk_size: int = 50, advantage_source: str = "progress",
                        stage_nums: int = 1) -> Tuple[Dict[int, List[float]], List[str]]:
    """
    Collect all rewards from all parquet files to compute statistics.
    
    Args:
        parquet_path: Base directory path containing chunk-*/*.parquet files
        chunk_size: Number of frames to look ahead for progress calculation
        advantage_source: Source of advantage values
        stage_nums: Number of stages to divide data into based on stage_progress_gt
    """

    assert advantage_source in ["absolute_advantage", "relative_advantage", "progress"], \
        f"Unknown advantage source, should be in [absolute_advantage, relative_advantage, progress], but got {advantage_source}"
    
    parquet_files = list(parquet_path.glob('chunk-*/*.parquet'))
    assert len(parquet_files) > 0, f"No parquet files found in {parquet_path}/chunk-*/"
    rewards_by_stage = {i: [] for i in range(stage_nums)} # 初始化奖励列表，用于存储每个阶段的奖励
    stage_cut_points = [i/stage_nums for i in range(stage_nums)] # 初始化阶段划分点
    for parquet_file in tqdm(parquet_files, desc="Collecting rewards from all parquet files"):
        advantages = pd.read_parquet(parquet_file)[advantage_source].values
        rewards = calculate_rewards(advantages, chunk_size, advantage_source)
        if stage_nums == 1:
            rewards_by_stage[0].extend(rewards.tolist())
        else:
            stage_progress_gt_values = pd.read_parquet(parquet_file)['stage_progress_gt'].values
            for frame_idx, spg in enumerate(stage_progress_gt_values):
                stage_idx = np.where(spg >= stage_cut_points)[0][-1] # 找到当前帧属于哪个阶段
                rewards_by_stage[stage_idx].append(rewards[frame_idx])
        
    return rewards_by_stage, parquet_files

def update_tasks_jsonl(tasks_path: Path, prompt:str|None=None) -> None:
    """
    Update tasks.jsonl file with reward statistics.
    
    Args:
        tasks_path: Base directory path containing tasks.jsonl file
    """
    tasks_jsonl_path = tasks_path / "tasks.jsonl"
    if prompt is None:
        assert tasks_jsonl_path.exists(), f"Tasks.jsonl file not found at {tasks_jsonl_path}"
        with open(tasks_jsonl_path, 'r') as f:
            tasks = [json.loads(i) for i in f]
        task_desc = tasks[0]['task'].strip()
    else:
        task_desc = prompt.strip()
    if not task_desc[-1].isalpha(): # 如果最后一个字符不是字母，则去掉最后一个字符
        task_desc = task_desc[:-1]
    with open(tasks_jsonl_path, 'w') as f:
        task_desc = tasks[0]['task']
        f.write(json.dumps({
            'task_index': 0,
            'task': task_desc+', Advantage: negative',
        }) + '\n')
        f.write(json.dumps({
            'task_index': 1,
            'task': task_desc+', Advantage: positive',
        }) + '\n')

def update_info_json(info_path: Path) -> None:
    """
    Update info.jsonl file, update total_tasks from 1 to 2.
    
    Args:
        info_path: Base directory path containing info.jsonl file
    """
    info_jsonl_path = info_path / "info.json"
    assert info_jsonl_path.exists(), f"Info.json file not found at {info_jsonl_path}"
    with open(info_jsonl_path, 'r') as f:
        info = json.load(f)
        info['total_tasks'] = 2
    with open(info_jsonl_path, 'w') as f:
        json.dump(info, f, indent=4)

def update_parquet_file(parquet_file: Path, threshold_points: float|List[float], chunk_size: int = 50, advantage_source: str = "progress") -> np.ndarray:
    """
    Update parquet file, add new column 'reward', update task_index based on threshold_points.
    
    Args:
        parquet_file: Path to the parquet file
        threshold_points: List of threshold points
    return:
        Array of reward values
    """
    df = pd.read_parquet(parquet_file)
    rewards = calculate_rewards(df[advantage_source].values, chunk_size, advantage_source)
    df['reward'] = rewards
    # 更新task_index based on threshold_points
    # 如果threshold_points是个单元素的列表，则使用该元素，视为单阶段任务
    if isinstance(threshold_points, list) and len(threshold_points) == 1:
        threshold_points = threshold_points[0]

    if isinstance(threshold_points, float):
        task_index = (rewards >= threshold_points).astype(np.int32)
    elif isinstance(threshold_points, List[float]):
        task_index = np.zeros(len(rewards), dtype=np.int32)
        stage_nums = len(threshold_points)
        stage_cut_points = [i/stage_nums for i in range(stage_nums)] # 初始化阶段划分点
        stage_progress_gt_values = df['stage_progress_gt'].values
        for frame_idx, spg in enumerate(stage_progress_gt_values):
            stage_idx = np.where(spg >= stage_cut_points)[0][-1] # 找到当前帧属于哪个阶段
            task_index[frame_idx] = rewards[frame_idx] >= threshold_points[stage_idx]
    else:
        raise ValueError(f"Unknown threshold_points type: {type(threshold_points)}")
    df['task_index'] = task_index

    df.to_parquet(parquet_file, index=False)
    return rewards

def update_episode_stats_jsonl(episode_stats_path: Path, rewards: Dict[int, List[float] | np.ndarray]) -> None:
    """
    Update episode_stats.jsonl file, add new column 'reward'.
    don't update the task_index as new task is assigned to each frame not episode.
    Args:
        episode_stats_path: Base directory path containing episode_stats.jsonl file
        rewards: Dictionary of rewards by episode_index
    """
    episode_stats_jsonl_path = episode_stats_path / "episodes_stats.jsonl"
    assert episode_stats_jsonl_path.exists(), f"Episode_stats.jsonl file not found at {episode_stats_jsonl_path}"
    with open(episode_stats_jsonl_path, 'r') as f:
        episode_stats = [json.loads(line) for line in f]
    for episode_stat in episode_stats:
        episode_index = episode_stat['episode_index']
        if rewards.get(episode_index) is not None:
            episode_stat['stats']['reward'] = compute_reward_statistics(rewards[episode_index])
    with open(episode_stats_jsonl_path, 'w') as f:
        for episode_stat in episode_stats:
            f.write(json.dumps(episode_stat) + '\n')

def update_all_advantage(repo_id:Path, parquet_path:str='data', prompt:str='fold the cloth', chunk_size: int = 50, advantage_source: str = "progress", stage_nums: int = 1, positive_rate: float = 30):
    """
    读取repo_id下的所有parquet文件，计算奖励，更新所有parquet文件，更新tasks.jsonl，info.jsonl，episode_stats.jsonl文件
    Args:
        repo_id: Path to the repository
        parquet_path: Path to the parquet files, default is 'data'
        chunk_size: Number of frames to look ahead for progress calculation
        advantage_source: Source of advantage values
        stage_nums: Number of stages to divide data into based on stage_progress_gt
        positive_rate: Positive rate of the task, default is 30%
    """
    rewards_by_stage, parquet_files = collect_all_rewards(repo_id/parquet_path, chunk_size=chunk_size, advantage_source=advantage_source, stage_nums=stage_nums)
    update_tasks_jsonl(repo_id/'meta', prompt)
    update_info_json(repo_id/'meta')
    # 计算阈值点,用于划分advantage: negative和advantage: positive
    threshold_points = compute_threshold_points(rewards_by_stage, positive_rate)
    all_rewards = {i: [] for i in range(stage_nums)}
    for parquet_file in parquet_files:
        rewards = update_parquet_file(parquet_file, threshold_points, chunk_size=chunk_size, advantage_source=advantage_source)
        episode_index = int(parquet_file.name.split('_')[-1].split('.')[0])
        all_rewards[episode_index] = rewards
    update_episode_stats_jsonl(repo_id/'meta', rewards_by_stage)
#--------------------------------End of update_all_advantage--------------------------------
############################################################################################
#--------------------------------Start of main--------------------------------
from value_realtime_evaluator_video_fast_thread import SimpleValueEvaluator
from lerobot_evaluation import edit_parquet_file
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def build_args():
    """
    Build arguments for the program.
    """
    parser = argparse.ArgumentParser(description='Update tasks.jsonl and info.jsonl files.')
    parser.add_argument('--chunk_size', type=int, default=50, help='Number of frames to look ahead for progress calculation')
    parser.add_argument('--advantage_source', type=str, default='absolute_advantage', help='Source of advantage values')
    parser.add_argument('--stage_nums', type=int, default=1, help='Number of stages to divide data into based on stage_progress_gt')
    parser.add_argument('--positive_rate', type=float, default=30, help='Positive rate of the task, default is 30%')
    parser.add_argument('--prompt', type=str|None, default=None, help='new task description')
    args = parser.parse_args()
    return args

def main():
    args = build_args()
    repo_id = Path("../test_data_40/")
    model_name = "baidx_Test"
    model_cfg = {
        "config_name": 'VALUE_TORCH_Pi05_KAI_FLATTEN_FOLD',
        "ckpt_dir": "../openpi/checkpoints/VALUE_TORCH_Pi05_KAI_FLATTEN_FOLD/0105/10000/",
        "ckpt_steps": 2000
    }
    evaluator = SimpleValueEvaluator(
        config_name=model_cfg['config_name'],
        ckpt_dir=model_cfg['ckpt_dir'],
        num_workers=8,  # 并行线程数，根据CPU核心数调整
    )

    dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=repo_id,)
    for i in tqdm([0,1], desc="Evaluating videos"):
        parquet_file = repo_id/dataset_metadata.data_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i)
        if not parquet_file.exists():
            print(f"Parquet file {parquet_file} not found")
            continue
        min_frame_index = pq.read_table(parquet_file)['frame_index'].to_pylist()[0]
        max_frame_index = pq.read_table(parquet_file)['frame_index'].to_pylist()[-1]
        top_video = repo_id/Path(dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.top_head'))
        left_video = repo_id/Path(dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.hand_left'))
        right_video = repo_id/Path(dataset_metadata.video_path.format(episode_chunk=i//dataset_metadata.chunks_size,episode_index=i,video_key='observation.images.hand_right'))
        results = evaluator.evaluate_video_1timestep_advantage(
            video_paths=(top_video, left_video, right_video),
            prompt="Flatten and fold the cloth.",
            batch_size=8,
            frame_interval=1,  # 1为全评估，2为隔一帧评估，3为每3帧评估一次
            relative_interval=50,
            min_frame_index=min_frame_index,
            max_frame_index=max_frame_index,
            prefetch=True,  # 启用数据预取
        )
        output_path=repo_id / f"data_{model_name}" / parquet_file.relative_to(repo_id / "data")
        edit_parquet_file(
            src_parquet=parquet_file,
            output_path=output_path,
            advantages_dict=results
        )

    update_all_advantage(repo_id, parquet_path=f"data_{model_name}", prompt=args.prompt, chunk_size=args.chunk_size, advantage_source=args.advantage_source, stage_nums=args.stage_nums, positive_rate=args.positive_rate)

if __name__ == "__main__":
    main()
    # uv run python lerobot_value_reward.py --prompt "Flatten and fold the cloth" --chunk_size 50 --advantage_source absolute_advantage --stage_nums 1 --positive_rate 30