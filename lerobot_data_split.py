from locale import D_T_FMT
from pathlib import Path
import json
import argparse
import shutil
import random
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

def split_lerobot_data(source_path: Path, dst_path: Path, episode_index:list[int], num:int):

    # 重新分配episode_index
    old_episode_index = episode_index
    old_episode_index.sort()
    new_episode_index = list(range(len(old_episode_index)))
    old2new_episode_index = dict(zip(old_episode_index, new_episode_index))
    new2old_episode_index = dict(zip(new_episode_index, old_episode_index))

    dst_path.mkdir(parents=True, exist_ok=True)
    # 读取episodes_stats.jsonl文件，并获得当前episode_index对应的episode_stat
    with open(source_path/'meta'/'episodes_stats.jsonl', 'r') as f:
        episodes_stats = [json.loads(line) for line in f if line.strip()]
    episodes_stats = [episode for episode in episodes_stats if episode['episode_index'] in episode_index]
    assert len(episodes_stats) == len(episode_index), f'episode_index: {episode_index}, episodes_stats: {episodes_stats}'
    episodes_stats.sort(key=lambda x: x['episode_index'])

    
    with open(source_path/'meta'/'info.json', 'r') as f:
        info = json.load(f)
    chunks_size = info['chunks_size']

    # 更新episodes_stats.jsonl
    new_episodes_stats = []
    new_frame_index = 0
    for idx, episode_stat in enumerate(tqdm(episodes_stats, desc=f'{num}, Processing episodes_stats.jsonl', total=len(episodes_stats))):
        index_count = episode_stat['stats']['index']['count'][0]
        # 更新episode序号和全局帧序号
        new_episode_index = old2new_episode_index[episode_stat['episode_index']]
        episode_stat['episode_index'] = new_episode_index
        episode_stat['stats']['index']['min'] = [new_frame_index]
        episode_stat['stats']['index']['max'] = [new_frame_index + index_count - 1]
        episode_stat['stats']['index']['mean'] = [(new_frame_index + new_frame_index + index_count - 1) / 2]
        episode_stat['stats']['index']['std'] = [np.std(range(new_frame_index, new_frame_index + index_count))]
        episode_stat['stats']['index']['count'] = [index_count]
        new_frame_index += index_count
        new_episodes_stats.append(episode_stat)

    (dst_path/'meta').mkdir(parents=True, exist_ok=True)
    with open(dst_path/'meta'/'episodes_stats.jsonl', 'w') as f:
        for episode_stat in new_episodes_stats:
            f.write(json.dumps(episode_stat) + '\n')
    # 更新info.json
    with open(dst_path/'meta'/'info.json', 'w') as f:
        info['total_episodes'] = len(old_episode_index)
        info['total_frames'] = new_frame_index
        info['total_videos'] = len(old_episode_index) * 3
        info['total_chunks'] = len(old_episode_index) // chunks_size + 1
        info['splits'] = {
            'train': f"0:{len(old_episode_index)}",
        }
        json.dump(info, f, indent=4)
    # parquet文件处理
    # (dst_path/'data').mkdir(parents=True, exist_ok=True)
    for new_stat in new_episodes_stats:
        new_index = new_stat['episode_index']
        old_index = new2old_episode_index[new_index]
        old_episode_path = source_path/'data'/f'chunk-{old_index//chunks_size:03d}'/f'episode_{old_index:06d}.parquet'
        new_episode_path = dst_path/'data'/f'chunk-{new_index//chunks_size:03d}'/f'episode_{new_index:06d}.parquet'
        if not new_episode_path.parent.exists():
            new_episode_path.parent.mkdir(parents=True, exist_ok=True)
        parquet = pd.read_parquet(old_episode_path)
        parquet['index'] = parquet['index'] - parquet['index'].min() + new_stat['stats']['index']['min'][0]
        parquet['episode_index'] = new_index
        parquet.to_parquet(new_episode_path, index=False)
    
    # 更新record.csv文件
    record_csv = os.listdir(source_path/'meta')
    record_csv = [item for item in record_csv if 'record.csv' in item and item.lower().startswith('v')]

    for record_file in record_csv:
        record_data = pd.read_csv(source_path/'meta'/record_file)
        record_data['episode_index'] = record_data['video'].str.split('_').str[-1].str.split('.').str[0].astype(int)
        record_data = record_data.loc[record_data['episode_index'].isin(episode_index)]
        record_data['episode_index'] = record_data['episode_index'].map(old2new_episode_index)
        record_data['video'] = record_data['video'].map(lambda x: f"episode_{old2new_episode_index[int(x.split('.')[0].split('_')[-1])]:06d}.mp4")
        record_data.drop(columns=['episode_index'], inplace=True)
        record_data.to_csv(dst_path/'meta'/record_file, index=False)
    # 更新episodes.jsonl
    with open(source_path/'meta'/'episodes.jsonl', 'r') as f:
        episodes = [json.loads(line) for line in f if line.strip()]
    episodes = [episode for episode in episodes if episode['episode_index'] in old_episode_index]
    assert len(episodes) == len(episode_index), f'episode_index: {episode_index}, episodes: {episodes}'
    episodes.sort(key=lambda x: x['episode_index'])
    for idx, episode in enumerate(episodes):
        episode['episode_index'] = old2new_episode_index[episode['episode_index']] 
    with open(dst_path/'meta'/'episodes.jsonl', 'w') as f:
        for episode in episodes:
            f.write(json.dumps(episode) + '\n')
    # 复制meta/tasks.jsonl
    shutil.copy(source_path/'meta'/'tasks.jsonl', dst_path/'meta'/'tasks.jsonl')

    # video文件处理
    # (dst_path/'videos').mkdir(parents=True, exist_ok=True)
    video_keys = ['observation.images.hand_left',  'observation.images.hand_right',  'observation.images.top_head']
    for new_stat in tqdm(new_episodes_stats, desc=f'{num}, Processing videos', total=len(new_episodes_stats)):
        new_index = new_stat['episode_index']
        old_index = new2old_episode_index[new_index]
        for video_key in video_keys:
            old_episode_path = source_path/'videos'/f'chunk-{old_index//chunks_size:03d}'/f'{video_key}'/f'episode_{old_index:06d}.mp4'
            new_episode_path = dst_path/'videos'/f'chunk-{new_index//chunks_size:03d}'/f'{video_key}'/f'episode_{new_index:06d}.mp4'
            if not new_episode_path.parent.exists():
                new_episode_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(old_episode_path, new_episode_path)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, required=True, help='原始lerobot数据集路径')
    parser.add_argument('--dst_path', type=str, required=True, help='拆分后的lerobot数据集路径')
    parser.add_argument('--split_num', type=int, default=4, help='拆分后的数据集数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子,用于打乱episode_index次序')
    args = parser.parse_args()
    source_path = Path(args.source_path)
    dst_path = Path(args.dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    with open(source_path/'meta'/'info.json', 'r') as f:
        info = json.load(f)
    total_episodes = info['total_episodes']
    episode_index = list(range(total_episodes))
    random.seed(args.seed)
    random.shuffle(episode_index)
    episode_index = np.array_split(episode_index, args.split_num)
    max_workers = min(args.split_num, os.cpu_count())
    with mp.Pool(processes=max_workers) as pool:
        pool.starmap(split_lerobot_data, [(source_path, dst_path/f'split_{i}', episode_index[i], i) for i in range(args.split_num)])
if __name__ == '__main__':
    main()