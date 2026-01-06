# 功能引进

## data_loader.py 
* `create_torch_dataset()`函数改造
    * 使用CustomLeRobotDataset替换原LeRobotDataset
    * 支持从TrainConfig传入自定义参数（n_history、with_episode_start、timestep_difference_mode等）
    * 支持数据集split划分（train/val/all）
* 新增`episodes_split_through_task()`函数
    * 按任务（task）进行episode划分
    * 支持train/val按比例划分（默认90%/10%）
    * 确保同一任务的episode在训练集和验证集中均有分布

## custom_lerobot_dataset.py
* CustomLeRobotDataset  
    * 继承lerobot.common.datasets.lerobot_dataset.LeRobotDataset
    * 增加自定义处理需求
        * 自定义参数
            - `n_history`: 历史帧数量，用于包含历史帧信息
            - `with_episode_start`: 是否包含episode起始帧
            - `skip_sample_ratio_within_episode`: episode内跳过采样比例（0~0.5）
            - `timestep_difference_mode`: 时间步差异模式，用于对比学习
            - `stage_process_mode`: 阶段进度监督模式
        * 核心方法
            - `__getitem__()`: 自定义数据获取逻辑，返回包含进度标签的样本
            - `get_sample_with_imgs_from_idx()`: 获取带视频帧的样本
            - `handle_delta_indices()`: 处理动作序列的delta索引
            - `handle_timestep_difference_mode()`: 同episode随机采样另一帧进行对比
            - `handle_episode_start_frame()`: 获取episode起始帧
            - `handle_history_frames()`: 获取历史帧序列
        * 进度标签计算
            - 支持普通进度模式（progress_gt）
            - 支持阶段进度模式（stage_progress_gt）
            - 时间步差异模式下计算两帧进度差值

## config.py
* `LerobotPikaDataConfig`（DataConfigFactory）
    * Pika机器人数据集配置
    * 多相机设置：top_head、hand_left、hand_right
    * 支持delta关节动作转换（use_delta_joint_actions）
    * 自定义repack_transforms映射数据键名
* `TrainConfig`新增参数
    * `is_train`: 训练过程使用部分数据
    * `split`: 数据集划分模式（train/val/all）
    * `n_history`: 历史帧数量
    * `with_episode_start`: 是否包含episode起始帧
    * `skip_sample_ratio_within_episode`: episode内跳过采样比例
    * `timestep_difference_mode`: 时间步差异模式
    * `stage_process_mode`: 阶段进度监督模式
    * `drop_last`: 丢弃最后数量不足的batch
    * `skip_norm_stats`: 是否跳过归一化统计
* 预设TrainConfig配置
    * `VALUE_TORCH_Pi05_KAI_FLATTEN_FOLD`: 时间步差异模式训练配置
    * `VALUE_TORCH_Pi05_KAI_FLATTEN_FOLD_SINGLE`: 单帧进度预测训练配置
    * 均使用Pi0Config_Custom（带value head）和LerobotPikaDataConfig

## transforms.py
* `TokenizePrompt`类改造
    * 新增action_advantage处理逻辑
    * 将action_advantage离散化为10个区间（范围[-1, 1]）
    * 将离散化后的action_advantage传入tokenizer
    * 返回字典中新增`action_advantage`和`action_advantage_original`字段

## models/model.py
* `Observation`类扩展
    * 新增字段
        - `episode_index`: episode索引
        - `frame_index`: 帧索引
        - `progress`: 进度值
        - `episode_length`: episode长度
        - `action_advantage`: 离散化后的动作优势
        - `action_advantage_original`: 原始动作优势值
        - `image_original`: 原始图像（未经数据增强）
* `Observation.from_dict()`方法改造
    * 支持image_original的uint8到float32转换
    * 映射所有新增自定义字段
* `preprocess_observation()`函数改造
    * 传递所有自定义字段到输出Observation

## models/pi0_config.py
* 新增`Pi0Config_Custom`类（继承Pi0Config）
    * 自定义参数
        - `timestep_difference_mode`: 时间步差异模式
        - `stage_process_mode`: 阶段进度模式
        - `with_value_head`: 是否启用value head
        - `loss_action_weight`: action损失权重
        - `loss_value_weight`: value损失权重
        - `loss_value_use_bce`: 是否使用BCE损失
        - `p_mask_ego_state`: 训练时mask ego state的概率
        - `cfg_scale`: CFG缩放因子
        - `download_path`: tokenizer.model下载地址，控制下载/读取地址

## models/tokenizers.py
* PaligemmaTokenizer
    * 新增参数 
        - `download_path`: 控制模型下载/读取地址

## models_pytorch/pi0_pytorch.py
* 新增辅助函数`get_1d_sincos_pos_embed_from_grid()`
    * 生成1D正弦余弦位置编码
    * 用于action_advantage的嵌入表示
* 新增`PI0Pytorch_Custom`类（继承PI0Pytorch）
    * 配置参数
        - `with_value_head`: 是否启用value head
        - `loss_value_weight`: value损失权重
        - `loss_value_use_bce`: 是否使用BCE损失
        - `loss_action_weight`: action损失权重
        - `p_mask_ego_state`: 训练时随机mask ego state的概率
        - `timestep_difference_mode`: 时间步差异模式
        - `cfg_scale`: Classifier-Free Guidance缩放因子
    * Value Head结构
        - 3层MLP：Linear→SiLU→Linear→SiLU→Linear
        - 时间步差异模式使用Tanh激活（输出[-1,1]）
        - 普通模式使用Sigmoid激活（输出[0,1]）
        - 使用suffix_out第一个token（state token）的表示作为输入
    * 改造方法
        - `_preprocess_observation`: 支持返回full_obs，apply_aug默认关闭
        - `embed_prefix()`: action_advantage通过sincos位置编码嵌入，拼接到lang_emb
        - `embed_suffix()`: 支持训练时按概率mask ego state（置0）
        - `forward()`: 支持return_loss_dict参数，返回loss_aux_dict用于日志记录
        - `sample_actions()`: CFG模式下batch拆分为cond/uncond两部分进行采样
    * 新增方法
        - `sample_values()`: 使用dummy action和随机time进行前向传播，仅预测进度值

## policies/pika_policy.py
* 新增策略
* 新增`PikaInputs`类（DataTransformFn）
    * 相机配置
        - 必需相机：top_head、hand_left、hand_right
        - 可选相机：his_-100_top_head、his_-100_hand_left、his_-100_hand_right（历史帧）
        - 相机重命名：top_head→base_0_rgb、hand_left→left_wrist_0_rgb等
    * 数据处理
        - state/actions填充至action_dim
        - 过滤异常值（超出±π范围置0）
        - 图像格式转换：float32[C,H,W]→uint8[H,W,C]
    * 传递自定义字段
        - frame_index、episode_length、episode_index
        - action_advantage、action_advantage_original
        - progress、image_original
* 新增`PikaOutputs`类（DataTransformFn）
    * 返回actions的前14维（13关节+1夹爪）

## models_pytorch/preprocessing_pytorch.py
* 新增`preprocess_observation_pytorch_custom()`函数
    * 新增参数
        - `return_full_obs`: 是否返回完整observation（包含所有自定义字段）
        - `apply_aug`: 是否应用数据增强
    * 动态image_keys
        - 从observation.images动态获取键名
        - 自定义排序：按时间步和相机位置排序（base→left_wrist→right_wrist）
    * 返回完整observation时包含字段
        - token_ar_mask、token_loss_mask、
        - action_advantage、action_advantage_original
        - frame_index、episode_length、episode_index
        - image_original、progress

# 修复
## models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
* GemmaRMSNorm
    * 修复`extra_repr`错误打印weight.shape的问题