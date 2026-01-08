"""简化的高效Value评估类，直接读视频并输出结果列表"""

from __future__ import annotations

import dataclasses
import os
import cv2
import numpy as np
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
import torch
import safetensors.torch
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue

from openpi.training import config as _config
from openpi.shared import download
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch_Custom as PI0Pytorch
import openpi.models.tokenizer as _tokenizer
from types import SimpleNamespace
from openpi.shared import image_tools

class SimpleValueEvaluator:
    """简化的评估类，只进行推理并返回结果列表，不保存任何文件"""
    
    def __init__(self, config_name: str, ckpt_dir: str, num_workers: int = 4):
        """
        初始化评估器
        
        Args:
            config_name: 配置名称
            ckpt_dir: 检查点目录
            num_workers: 并行工作线程数，用于视频加载和图像预处理
        """
        self.config_name = config_name
        self.ckpt_dir = ckpt_dir
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置和模型
        self._load_model()
        
        # 初始化tokenizer
        self.tokenizer = _tokenizer.PaligemmaTokenizer(self.config.model.max_token_len)
        
        # 创建线程池用于并行处理
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        
        logging.info(f"评估器初始化完成，使用设备: {self.device}, 并行线程数: {num_workers}")
    
    def __del__(self):
        """清理线程池"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def shutdown(self):
        """显式关闭线程池"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
            logging.info("线程池已关闭")
    
    def _load_model(self):
        """加载模型和配置"""
        # 加载配置
        self.config = _config.get_config(self.config_name)
        checkpoint_dir = download.maybe_download(self.ckpt_dir)
        
        # 创建模型
        new_model = self.config.model.__class__(**{**self.config.model.__dict__,
                                                })
        self.config = dataclasses.replace(self.config, model=new_model)
        
        # 加载模型权重
        self.model = PI0Pytorch(new_model).to(self.device)
        self.model.eval()
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        logging.info(f"加载模型权重: {model_path}")
        safetensors.torch.load_model(self.model, model_path, strict=True)
        logging.info("模型加载完成")
    
    def _load_video_frames(self, video_path: str, frame_interval: int = 1) -> List[np.ndarray]:
        """
        从视频文件加载帧，支持间隔采样
        
        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔，1为全评估，2为隔一帧评估，依此类推
            
        Returns:
            numpy数组列表 (RGB格式)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 根据间隔采样
                if frame_count % frame_interval == 0:
                    # OpenCV读取的是BGR格式，转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
        finally:
            cap.release()
        
        logging.info(f"从视频 {os.path.basename(video_path)} 加载了 {len(frames)} 帧 (总帧数: {frame_count}, 间隔: {frame_interval})")
        return frames
    
    def _load_videos_parallel(
        self, 
        video_paths: Tuple[str, str, str], 
        frame_interval: int = 1
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        并行加载三个视频文件
        
        Args:
            video_paths: 三个视频路径元组 (top_video, left_video, right_video)
            frame_interval: 帧间隔
            
        Returns:
            三个视频的帧列表元组
        """
        top_video_path, left_video_path, right_video_path = video_paths
        
        # 使用线程池并行加载三个视频
        futures = {
            self._executor.submit(self._load_video_frames, top_video_path, frame_interval): 'top',
            self._executor.submit(self._load_video_frames, left_video_path, frame_interval): 'left',
            self._executor.submit(self._load_video_frames, right_video_path, frame_interval): 'right'
        }
        
        results = {}
        for future in as_completed(futures):
            video_type = futures[future]
            results[video_type] = future.result()
        
        return results['top'], results['left'], results['right']
    
    def _process_single_image(self, rgb_img: np.ndarray) -> torch.Tensor:
        """
        处理单个图像，转换为tensor格式
        
        Args:
            rgb_img: RGB格式的numpy图像
            
        Returns:
            处理后的tensor (C, H, W)
        """
        # 转换为tensor并归一化到[0, 1]
        tensor = torch.from_numpy(rgb_img).float() / 255.0
        # 归一化到[-1, 1]
        tensor = tensor * 2.0 - 1.0
        
        # Resize和padding
        tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)
        
        # HWC to CHW
        tensor = tensor.permute(2, 0, 1)
        
        return tensor
    
    def _batch_numpy_to_tensor_parallel(self, np_images: List[np.ndarray]) -> torch.Tensor:
        """
        并行将numpy图像列表批量转换为模型所需的tensor格式
        
        Args:
            np_images: numpy图像列表 (RGB格式)
            
        Returns:
            torch.Tensor with shape (batch_size, C, H, W)
        """
        # 使用线程池并行处理每个图像
        futures = [self._executor.submit(self._process_single_image, img) for img in np_images]
        
        tensors = []
        for future in futures:
            tensors.append(future.result())
        
        # Stack到batch维度
        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor
    
    def _batch_numpy_to_tensor(self, np_images: List[np.ndarray]) -> torch.Tensor:
        """
        将numpy图像列表批量转换为模型所需的tensor格式
        
        Args:
            np_images: numpy图像列表 (RGB格式)
            
        Returns:
            torch.Tensor with shape (batch_size, C, H, W)
        """
        tensors = []
        for rgb_img in np_images:
            # 转换为tensor并归一化到[0, 1]
            tensor = torch.from_numpy(rgb_img).float() / 255.0
            # 归一化到[-1, 1]
            tensor = tensor * 2.0 - 1.0
            
            # Resize和padding
            tensor = image_tools.resize_with_pad_torch(tensor, 224, 224)
            
            # HWC to CHW
            tensor = tensor.permute(2, 0, 1)
            
            tensors.append(tensor)
        
        # Stack到batch维度
        batch_tensor = torch.stack(tensors, dim=0)
        return batch_tensor
    
    def _prepare_batch_tensors(
        self,
        top_frames: List[np.ndarray],
        left_frames: List[np.ndarray],
        right_frames: List[np.ndarray],
        batch_indices: List[int],
        future_indices: List[int],
        initial_tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        并行准备一个批次的tensor数据
        
        Args:
            top_frames, left_frames, right_frames: 三个视角的帧列表
            batch_indices: 当前批次的帧索引列表
            future_indices: 对应的未来帧索引列表
            initial_tensors: 可选的初始帧tensor元组
            
        Returns:
            包含所有准备好的tensor的字典
        """
        # 收集当前帧和未来帧
        base_top_list = [top_frames[j] for j in batch_indices]
        base_left_list = [left_frames[j] for j in batch_indices]
        base_right_list = [right_frames[j] for j in batch_indices]
        future_top_list = [top_frames[j] for j in future_indices]
        future_left_list = [left_frames[j] for j in future_indices]
        future_right_list = [right_frames[j] for j in future_indices]
        
        # 并行处理所有图像列表
        all_lists = [
            base_top_list, base_left_list, base_right_list,
            future_top_list, future_left_list, future_right_list
        ]
        
        # 使用线程池并行转换
        futures = [
            self._executor.submit(self._batch_numpy_to_tensor_parallel, img_list)
            for img_list in all_lists
        ]
        
        results = [f.result() for f in futures]
        
        return {
            'base_top': results[0],
            'base_left': results[1],
            'base_right': results[2],
            'future_top': results[3],
            'future_left': results[4],
            'future_right': results[5]
        }
    
    def evaluate_video_2timesteps_advantages(
        self,
        video_paths: Tuple[str, str, str],
        prompt: str,
        batch_size: int = 8,
        frame_interval: int = 1,
        relative_interval: int = 50,
        min_frame_index: int = None,
        max_frame_index: int = None,
        prefetch: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从三个视频文件评估价值变化，使用第一帧和前向对比方式进行绝对和相对评估
        
        Args:
            video_paths: 三个视频路径元组 (top_video, left_video, right_video)
            prompt: 任务提示文本
            batch_size: 批处理大小，用于并行推理
            frame_interval: 帧间隔，1为全评估，2为隔一帧评估
            relative_interval: 相对评估间隔，表示用多少帧之后的图像作为对比
                             例如relative_interval=50，评估第10帧时，会用第60帧作为未来参考
                             结果存储在第n帧上，对比的是n帧(base_-100)和n+50帧(base_0)
            prefetch: 是否启用数据预取，在GPU推理时预先准备下一批数据
            
        Returns:
            结果列表，每个元素包含 frame_idx, value, future_frame_idx(用作对比的未来帧索引)
        """
        if len(video_paths) != 3:
            raise ValueError("需要提供三个视频路径: (top, left, right)")
        
        # 并行加载视频帧（带间隔采样）
        logging.info(f"开始并行加载视频帧 (间隔: {frame_interval}, 线程数: {self.num_workers})...")
        top_frames, left_frames, right_frames = self._load_videos_parallel(video_paths, frame_interval)
        
        # 检查帧数是否一致
        if len(left_frames) != len(top_frames) or len(right_frames) != len(top_frames):
            raise ValueError(
                f"三个视频的帧数不一致: top={len(top_frames)}, "
                f"left={len(left_frames)}, right={len(right_frames)}"
            )
        
        top_frames = top_frames[min_frame_index:max_frame_index+1]
        left_frames = left_frames[min_frame_index:max_frame_index+1]
        right_frames = right_frames[min_frame_index:max_frame_index+1]
        num_frames = len(top_frames)
        if num_frames < 2:
            raise ValueError(f"视频帧数不足: {num_frames}，至少需要2帧")

        logging.info(f"采样后总帧数: {num_frames}, 相对间隔: {relative_interval}")

        # 第一帧作为初始图像（并行处理）
        initial_futures = [
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [top_frames[0]]),
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [left_frames[0]]),
            self._executor.submit(self._batch_numpy_to_tensor_parallel, [right_frames[0]])
        ]
        initial_top_tensor = initial_futures[0].result().to(self.device)
        initial_left_tensor = initial_futures[1].result().to(self.device)
        initial_right_tensor = initial_futures[2].result().to(self.device)
        
        # Tokenize prompt
        tokens, token_masks = self.tokenizer.tokenize(prompt, state=None)
        
        # 结果列表
        all_results = []
        
        # 批量处理所有帧（从第0帧开始，对比n和n+50帧）
        logging.info(f"开始批量评估（前向评估间隔：n vs n+{relative_interval}，预取: {prefetch}）...")
        if num_frames <= 0:
            raise ValueError(f"视频帧数{num_frames}不足，需要至少{relative_interval + 1}帧")
        
        max_frame_idx = num_frames - 1
        batch_starts = list(range(0, num_frames, batch_size))
        
        # 预取第一批数据
        def prepare_batch_data(batch_start: int) -> Tuple[Dict, List[int], int]:
            """准备单个批次的数据"""
            end_idx = min(batch_start + batch_size, num_frames)
            current_batch_size = end_idx - batch_start
            batch_indices = list(range(batch_start, end_idx))
            future_frame_indices = [min(j + relative_interval, max_frame_idx) for j in batch_indices]
            
            # 并行处理所有图像
            all_images = []
            for j in batch_indices:
                all_images.extend([top_frames[j], left_frames[j], right_frames[j]])
            for fidx in future_frame_indices:
                all_images.extend([top_frames[fidx], left_frames[fidx], right_frames[fidx]])
            
            # 并行转换为tensor
            tensor_futures = [self._executor.submit(self._process_single_image, img) for img in all_images]
            tensors = [f.result() for f in tensor_futures]
            
            # 分割tensors
            n = current_batch_size
            base_top = torch.stack(tensors[0:n*3:3], dim=0)
            base_left = torch.stack(tensors[1:n*3:3], dim=0)
            base_right = torch.stack(tensors[2:n*3:3], dim=0)
            future_top = torch.stack(tensors[n*3::3], dim=0)
            future_left = torch.stack(tensors[n*3+1::3], dim=0)
            future_right = torch.stack(tensors[n*3+2::3], dim=0)
            
            return {
                'base_top': base_top,
                'base_left': base_left,
                'base_right': base_right,
                'future_top': future_top,
                'future_left': future_left,
                'future_right': future_right
            }, future_frame_indices, current_batch_size
        
        # 预取机制：在GPU计算时预先准备下一批数据
        prefetch_future = None
        if prefetch and len(batch_starts) > 1:
            prefetch_future = self._executor.submit(prepare_batch_data, batch_starts[0])
        
        for batch_idx, i in enumerate(tqdm(batch_starts, desc="评估进度")):
            # 获取当前批次数据
            if prefetch and prefetch_future is not None:
                batch_tensors, future_frame_indices, current_batch_size = prefetch_future.result()
            else:
                batch_tensors, future_frame_indices, current_batch_size = prepare_batch_data(i)
            
            # 预取下一批数据（在GPU推理之前启动）
            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare_batch_data, batch_starts[batch_idx + 1])
            
            # 移动到GPU
            base_top_batch = batch_tensors['base_top'].to(self.device)
            base_left_batch = batch_tensors['base_left'].to(self.device)
            base_right_batch = batch_tensors['base_right'].to(self.device)
            future_top_batch = batch_tensors['future_top'].to(self.device)
            future_left_batch = batch_tensors['future_left'].to(self.device)
            future_right_batch = batch_tensors['future_right'].to(self.device)

            # 扩展初始图像到batch大小
            initial_top_batch = initial_top_tensor.expand(current_batch_size, -1, -1, -1)
            initial_left_batch = initial_left_tensor.expand(current_batch_size, -1, -1, -1)
            initial_right_batch = initial_right_tensor.expand(current_batch_size, -1, -1, -1)
            
            # 构建batch observation
            relative_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_-100_rgb": base_top_batch,
                    "left_wrist_-100_rgb": base_left_batch,
                    "right_wrist_-100_rgb": base_right_batch,
                    
                    "base_0_rgb": future_top_batch,
                    "left_wrist_0_rgb": future_left_batch,
                    "right_wrist_0_rgb": future_right_batch,
                },
                "image_masks": {}
            }

            absolute_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_-100_rgb": initial_top_batch,
                    "left_wrist_-100_rgb": initial_left_batch,
                    "right_wrist_-100_rgb": initial_right_batch,
                    
                    "base_0_rgb": base_top_batch,
                    "left_wrist_0_rgb": base_left_batch,
                    "right_wrist_0_rgb": base_right_batch,
                },
                "image_masks": {}
            }
            
            # 扩展tokens到batch
            tokens_batch = np.tile(tokens[np.newaxis, :], (current_batch_size, 1))
            token_masks_batch = np.tile(token_masks[np.newaxis, :], (current_batch_size, 1))
            
            relative_observation = {
                **relative_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }
            absolute_observation = {
                **absolute_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }
            
            relative_observation = SimpleNamespace(**relative_observation)
            absolute_observation = SimpleNamespace(**absolute_observation)
            
            # 批量推理
            with torch.no_grad():
                relative_val_arr = self.model.sample_values(self.device, relative_observation)  # Shape=(batch_size, 1)
                absolute_val_arr = self.model.sample_values(self.device, absolute_observation)  # Shape=(batch_size, 1)
            
            # 处理每个结果 - 结果存储在第n帧上
            for j in range(current_batch_size):
                frame_idx = i + j  # 第n帧
                
                # 对相对评估结果进行调整
                if future_frame_indices[j] - frame_idx == relative_interval:
                    relative_val = float(relative_val_arr[j, 0].item())
                elif future_frame_indices[j] == frame_idx:
                    relative_val = float(0)
                else:
                    relative_val = float(relative_val_arr[j, 0].item()) / (future_frame_indices[j] - frame_idx) * relative_interval
                
                # 对绝对评估结果进行调整
                if frame_idx == 0:
                    absolute_val = float(0)
                else:
                    absolute_val = float(absolute_val_arr[j, 0].item())
                
                result = {
                    "frame_idx": frame_idx,  # 结果存储在第n帧
                    "future_frame_idx": future_frame_indices[j],  # 对比的未来帧索引(n+50)
                    "relative_advantage": relative_val,
                    "absolute_value": absolute_val
                }
                all_results.append(result)
                


        all_results_dict = {result["frame_idx"]:result for result in all_results}
        # 计算absolute_advantage
        for result in all_results:
            frame_idx = result["frame_idx"]
            future_frame_idx = result["future_frame_idx"]
            future_result = all_results_dict.get(future_frame_idx)
            if future_frame_idx == frame_idx:
                result["absolute_advantage"] = 0.0
            elif future_frame_idx - frame_idx != relative_interval:
                result["absolute_advantage"] = (future_result["absolute_value"] - result["absolute_value"]) / (future_frame_idx - frame_idx) * relative_interval
            else:
                result["absolute_advantage"] = future_result["absolute_value"] - result["absolute_value"]
            
            result["absolute_advantage"] = max(-1.0, min(1.0, result["absolute_advantage"]))
            result["relative_advantage"] = max(-1.0, min(1.0, result["relative_advantage"]))

        logging.info(f"评估完成，共处理 {len(all_results)} 帧")
        return all_results

    def evaluate_video_1timestep_advantage(
        self,
        video_paths: Tuple[str, str, str],
        prompt: str,
        batch_size: int = 8,
        frame_interval: int = 1,
        relative_interval: int = 50,
        min_frame_index: int = None,
        max_frame_index: int = None,
        prefetch: bool = True
    ) -> List[Dict[str, Any]]:
        """
        从三个视频文件评估价值变化，使用第一帧和前向对比方式进行绝对和相对评估
        
        Args:
            video_paths: 三个视频路径元组 (top_video, left_video, right_video)
            prompt: 任务提示文本
            batch_size: 批处理大小，用于并行推理
            frame_interval: 帧间隔，1为全评估，2为隔一帧评估
            relative_interval: 相对评估间隔，表示用多少帧之后的图像作为对比
                             例如relative_interval=50，评估第10帧时，会用第60帧作为未来参考
                             结果存储在第n帧上，对比的是n帧(base_-100)和n+50帧(base_0)
            prefetch: 是否启用数据预取，在GPU推理时预先准备下一批数据
            
        Returns:
            结果列表，每个元素包含 frame_idx, value, future_frame_idx(用作对比的未来帧索引)
        """
        if len(video_paths) != 3:
            raise ValueError("需要提供三个视频路径: (top, left, right)")
        
        # 并行加载视频帧（带间隔采样）
        logging.info(f"开始并行加载视频帧 (间隔: {frame_interval}, 线程数: {self.num_workers})...")
        top_frames, left_frames, right_frames = self._load_videos_parallel(video_paths, frame_interval)
        
        # 检查帧数是否一致
        if len(left_frames) != len(top_frames) or len(right_frames) != len(top_frames):
            raise ValueError(
                f"三个视频的帧数不一致: top={len(top_frames)}, "
                f"left={len(left_frames)}, right={len(right_frames)}"
            )
        
        
        top_frames = top_frames[min_frame_index:max_frame_index+1]
        left_frames = left_frames[min_frame_index:max_frame_index+1]
        right_frames = right_frames[min_frame_index:max_frame_index+1]
        num_frames = len(top_frames)
        if num_frames < 2:
            raise ValueError(f"视频帧数不足: {num_frames}，至少需要2帧")
        logging.info(f"采样后总帧数: {num_frames}, 相对间隔: {relative_interval}")
        
        # Tokenize prompt
        tokens, token_masks = self.tokenizer.tokenize(prompt, state=None)
        
        # 结果列表
        all_results = []
        
        # 批量处理所有帧（从第0帧开始，对比n和n+50帧）
        logging.info(f"开始批量评估（前向评估间隔：n vs n+{relative_interval}，预取: {prefetch}）...")
        if num_frames <= 0:
            raise ValueError(f"视频帧数{num_frames}不足，需要至少{relative_interval + 1}帧")
        
        max_frame_idx = num_frames - 1
        batch_starts = list(range(0, num_frames, batch_size))
        
        # 预取数据准备函数
        def prepare_batch_data_1timestep(batch_start: int) -> Tuple[Dict, List[int], int]:
            """准备单个批次的数据（仅当前帧）"""
            end_idx = min(batch_start + batch_size, num_frames)
            current_batch_size = end_idx - batch_start
            batch_indices = list(range(batch_start, end_idx))
            future_frame_indices = [min(j + relative_interval, max_frame_idx) for j in batch_indices]
            
            # 并行处理所有图像
            all_images = []
            for j in batch_indices:
                all_images.extend([top_frames[j], left_frames[j], right_frames[j]])
            
            # 并行转换为tensor
            tensor_futures = [self._executor.submit(self._process_single_image, img) for img in all_images]
            tensors = [f.result() for f in tensor_futures]
            
            # 分割tensors
            n = current_batch_size
            base_top = torch.stack(tensors[0:n*3:3], dim=0)
            base_left = torch.stack(tensors[1:n*3:3], dim=0)
            base_right = torch.stack(tensors[2:n*3:3], dim=0)
            
            return {
                'base_top': base_top,
                'base_left': base_left,
                'base_right': base_right,
            }, future_frame_indices, current_batch_size
        
        # 预取机制
        prefetch_future = None
        if prefetch and len(batch_starts) > 1:
            prefetch_future = self._executor.submit(prepare_batch_data_1timestep, batch_starts[0])
        
        for batch_idx, i in enumerate(tqdm(batch_starts, desc="评估进度")):
            # 获取当前批次数据
            if prefetch and prefetch_future is not None:
                batch_tensors, future_frame_indices, current_batch_size = prefetch_future.result()
            else:
                batch_tensors, future_frame_indices, current_batch_size = prepare_batch_data_1timestep(i)
            
            # 预取下一批数据
            if prefetch and batch_idx + 1 < len(batch_starts):
                prefetch_future = self._executor.submit(prepare_batch_data_1timestep, batch_starts[batch_idx + 1])

            # 移动到GPU
            base_top_batch = batch_tensors['base_top'].to(self.device)
            base_left_batch = batch_tensors['base_left'].to(self.device)
            base_right_batch = batch_tensors['base_right'].to(self.device)

            absolute_observation = {
                "state": torch.zeros((current_batch_size, 32), dtype=torch.float32).to(self.device),
                "images": {
                    "base_0_rgb": base_top_batch,
                    "left_wrist_0_rgb": base_left_batch,
                    "right_wrist_0_rgb": base_right_batch,
                },
                "image_masks": {}
            }
            
            # 扩展tokens到batch
            tokens_batch = np.tile(tokens[np.newaxis, :], (current_batch_size, 1))
            token_masks_batch = np.tile(token_masks[np.newaxis, :], (current_batch_size, 1))

            absolute_observation = {
                **absolute_observation,
                "tokenized_prompt": torch.from_numpy(tokens_batch).to(self.device),
                "tokenized_prompt_mask": torch.from_numpy(token_masks_batch).to(self.device)
            }
            
            absolute_observation = SimpleNamespace(**absolute_observation)
            
            # 批量推理
            with torch.no_grad():
                absolute_val_arr = self.model.sample_values(self.device, absolute_observation)  # Shape=(batch_size, 1)
            
            # 处理每个结果 - 结果存储在第n帧上
            for j in range(current_batch_size):
                frame_idx = i + j  # 第n帧
                
                # 对绝对评估结果进行调整
                if frame_idx == 0:
                    absolute_val = float(0)
                else:
                    absolute_val = float(absolute_val_arr[j, 0].item())
                
                result = {
                    "frame_idx": frame_idx,  # 结果存储在第n帧
                    "future_frame_idx": future_frame_indices[j],  # 对比的未来帧索引(n+50)
                    "absolute_value": absolute_val
                }
                all_results.append(result)

        all_results_dict = {result["frame_idx"]:result for result in all_results}
        # 计算absolute_advantage
        for result in all_results:
            frame_idx = result["frame_idx"]
            future_frame_idx = result["future_frame_idx"]
            future_result = all_results_dict.get(future_frame_idx)
            if future_frame_idx == frame_idx:
                result["absolute_advantage"] = 0.0
            elif future_frame_idx - frame_idx != relative_interval:
                result["absolute_advantage"] = (future_result["absolute_value"] - result["absolute_value"]) / (future_frame_idx - frame_idx) * relative_interval
            else:
                result["absolute_advantage"] = future_result["absolute_value"] - result["absolute_value"]
            
            result["absolute_advantage"] = max(-1.0, min(1.0, result["absolute_advantage"]))

        logging.info(f"评估完成，共处理 {len(all_results)} 帧")
        return all_results


def main():
    # 模型配置
    config_name = "VALUE_TORCH_Pi05_KAI_cloth_11_15"
    ckpt_dir = "/home/lirui/apollo-ckpts/value_model/VALUE_TORCH_Pi05_KAI_cloth_11_15/100000"
    ckpt_dir = "/gpfs/yangjiazhi/workspace/RoboRL/ScalableRL/checkpoints/VALUE_TORCH_Pi05_KAI_cloth_11_15/VALUE_TORCH_Pi05_KAI_cloth_11_15/100000"
    
    # 视频路径（三个视角）
    video_root = "/home/lirui/vla-apollo/examples/test_video"
    video_root = "/gpfs/zhaolirui/RoboRL/examples/test_video"
    top_video = os.path.join(video_root, "top_head.mp4")
    left_video = os.path.join(video_root, "hand_left.mp4")
    right_video = os.path.join(video_root, "hand_right.mp4")
    
    # 创建评估器，设置并行线程数
    evaluator = SimpleValueEvaluator(
        config_name=config_name,
        ckpt_dir=ckpt_dir,
        num_workers=48,  # 并行线程数，根据CPU核心数调整
    )
    
    # 评估视频
    results = evaluator.evaluate_video_2timesteps_advantages(
        video_paths=(top_video, left_video, right_video),
        prompt="Flatten and fold the cloth.",
        batch_size=8,
        frame_interval=1,  # 1为全评估，2为隔一帧评估，3为每3帧评估一次
        prefetch=True,  # 启用数据预取
    )
    
    # 打印结果示例
    print("\n=== 评估完成 ===")
    print(f"总结果数: {len(results)}")
    for res in results:
        print(f"帧{res['frame_idx']}, 未来帧{res['future_frame_idx']}: relative_advantage={res['relative_advantage']}, absolute_advantage={res['absolute_advantage']}, absolute_value={res['absolute_value']}")
    
    # 清理资源
    evaluator.shutdown()


if __name__ == "__main__":
    main()