if __name__ == "__main__": # 主程序入口
    import sys # 导入sys模块
    import os # 导入os模块
    import pathlib # 导入pathlib模块

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) # 获取项目根目录
    sys.path.append(ROOT_DIR) # 将项目根目录添加到sys.path
    os.chdir(ROOT_DIR) # 切换到项目根目录

import os # 导入os模块
import hydra # 导入hydra模块
import torch # 导入torch模块
from omegaconf import OmegaConf # 导入OmegaConf类
import pathlib # 导入pathlib模块
from torch.utils.data import DataLoader # 导入DataLoader类
import copy # 导入copy模块
import random # 导入random模块
import wandb # 导入wandb模块
import tqdm # 导入tqdm模块
import numpy as np # 导入numpy模块
import shutil # 导入shutil模块
from diffusion_policy.workspace.base_workspace import BaseWorkspace # 导入BaseWorkspace类
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy # 导入DiffusionUnetImagePolicy类
from diffusion_policy.dataset.base_dataset import BaseImageDataset # 导入BaseImageDataset类
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner # 导入BaseImageRunner类
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager # 导入TopKCheckpointManager类
from diffusion_policy.common.json_logger import JsonLogger # 导入JsonLogger类
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to # 导入dict_apply和optimizer_to函数
from diffusion_policy.model.diffusion.ema_model import EMAModel # 导入EMAModel类
from diffusion_policy.model.common.lr_scheduler import get_scheduler # 导入get_scheduler函数

OmegaConf.register_new_resolver("eval", eval, replace=True) # 注册新的解析器

class TrainDiffusionUnetImageWorkspace(BaseWorkspace): # 定义TrainDiffusionUnetImageWorkspace类，继承自BaseWorkspace
    include_keys = ['global_step', 'epoch'] # 包含的键
    # 设置随机种子，实例化模型和优化器，配置训练状态和EMA模型
    def __init__(self, cfg: OmegaConf, output_dir=None): # 初始化方法
        super().__init__(cfg, output_dir=output_dir) # 调用父类初始化方法

        # set seed # 设置随机种子
        seed = cfg.training.seed # 获取配置中的随机种子
        torch.manual_seed(seed) # 设置torch的随机种子
        np.random.seed(seed) # 设置numpy的随机种子
        random.seed(seed) # 设置random的随机种子

        # configure model # 配置模型
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy) # 实例化模型

        self.ema_model: DiffusionUnetImagePolicy = None # 初始化EMA模型为None
        if cfg.training.use_ema: # 如果使用EMA
            self.ema_model = copy.deepcopy(self.model) # 深拷贝模型

        # configure training state # 配置训练状态
        self.optimizer = hydra.utils.instantiate( # 实例化优化器
            cfg.optimizer, params=self.model.parameters())

        # configure training state # 配置训练状态
        self.global_step = 0 # 初始化全局步数
        self.epoch = 0 # 初始化epoch

    # 执行训练过程，包括数据集配置、模型训练、验证、采样、日志记录和检查点保存
    def run(self): # 运行方法
        cfg = copy.deepcopy(self.cfg) # 深拷贝配置

        # resume training # 恢复训练
        if cfg.training.resume: # 如果需要恢复训练
            lastest_ckpt_path = self.get_checkpoint_path() # 获取最新检查点路径
            if lastest_ckpt_path.is_file(): # 如果检查点文件存在
                print(f"Resuming from checkpoint {lastest_ckpt_path}") # 打印恢复信息
                self.load_checkpoint(path=lastest_ckpt_path) # 加载检查点

        # configure dataset # 配置数据集
        dataset: BaseImageDataset # 声明数据集类型
        dataset = hydra.utils.instantiate(cfg.task.dataset) # 实例化数据集
        assert isinstance(dataset, BaseImageDataset) # 确认数据集类型
        train_dataloader = DataLoader(dataset, **cfg.dataloader) # 创建训练数据加载器
        normalizer = dataset.get_normalizer() # 获取数据归一化器

        # configure validation dataset # 配置验证数据集
        val_dataset = dataset.get_validation_dataset() # 获取验证数据集
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader) # 创建验证数据加载器

        self.model.set_normalizer(normalizer) # 设置模型归一化器
        if cfg.training.use_ema: # 如果使用EMA
            self.ema_model.set_normalizer(normalizer) # 设置EMA模型归一化器

        # configure lr scheduler # 配置学习率调度器
        lr_scheduler = get_scheduler( # 获取学习率调度器
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        # configure ema # 配置EMA
        ema: EMAModel = None # 初始化EMA为None
        if cfg.training.use_ema: # 如果使用EMA
            ema = hydra.utils.instantiate( # 实例化EMA
                cfg.ema,
                model=self.ema_model)

        # configure env runner # 配置环境运行器
        env_runner: BaseImageRunner # 声明环境运行器类型
        env_runner = hydra.utils.instantiate( # 实例化环境运行器
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner) # 确认环境运行器类型

        # configure logging # 配置日志记录
        wandb_run = wandb.init( # 初始化wandb运行
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update( # 更新wandb配置
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint # 配置检查点
        topk_manager = TopKCheckpointManager( # 创建TopK检查点管理器
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer # 设备转移
        device = torch.device(cfg.training.device) # 获取训练设备
        self.model.to(device) # 模型转移到设备
        if self.ema_model is not None: # 如果EMA模型不为空
            self.ema_model.to(device) # EMA模型转移到设备
        optimizer_to(self.optimizer, device) # 优化器转移到设备

        # save batch for sampling # 保存用于采样的批次
        train_sampling_batch = None # 初始化训练采样批次为None

        if cfg.training.debug: # 如果处于调试模式
            cfg.training.num_epochs = 2 # 设置训练轮数为2
            cfg.training.max_train_steps = 3 # 设置最大训练步数为3
            cfg.training.max_val_steps = 3 # 设置最大验证步数为3
            cfg.training.rollout_every = 1 # 设置每隔1个epoch进行一次rollout
            cfg.training.checkpoint_every = 1 # 设置每隔1个epoch保存一次检查点
            cfg.training.val_every = 1 # 设置每隔1个epoch进行一次验证
            cfg.training.sample_every = 1 # 设置每隔1个epoch进行一次采样

        # training loop # 训练循环
        log_path = os.path.join(self.output_dir, 'logs.json.txt') # 日志文件路径
        with JsonLogger(log_path) as json_logger: # 使用JsonLogger记录日志
            for local_epoch_idx in range(cfg.training.num_epochs): # 遍历训练轮数
                step_log = dict() # 初始化步日志
                # ========= train for this epoch ========== # 训练当前epoch
                if cfg.training.freeze_encoder: # 如果冻结编码器
                    self.model.obs_encoder.eval() # 设置编码器为评估模式
                    self.model.obs_encoder.requires_grad_(False) # 不需要计算编码器的梯度

                train_losses = list() # 初始化训练损失列表
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch: # 创建tqdm进度条
                    for batch_idx, batch in enumerate(tepoch): # 遍历训练数据加载器
                        # device transfer # 数据转移到设备
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None: # 如果训练采样批次为空
                            train_sampling_batch = batch # 设置训练采样批次

                        # compute loss # 计算损失
                        raw_loss = self.model.compute_loss(batch) # 计算原始损失
                        loss = raw_loss / cfg.training.gradient_accumulate_every # 损失除以梯度累积步数
                        loss.backward() # 反向传播

                        # step optimizer # 更新优化器
                        if self.global_step % cfg.training.gradient_accumulate_every == 0: # 如果到达梯度累积步数
                            self.optimizer.step() # 更新优化器
                            self.optimizer.zero_grad() # 梯度归零
                            lr_scheduler.step() # 更新学习率调度器
                        
                        # update ema # 更新EMA
                        if cfg.training.use_ema: # 如果使用EMA
                            ema.step(self.model) # 更新EMA

                        # logging # 日志记录
                        raw_loss_cpu = raw_loss.item() # 获取原始损失值
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False) # 更新进度条显示损失
                        train_losses.append(raw_loss_cpu) # 添加损失到训练损失列表
                        step_log = { # 记录步日志
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1)) # 是否为最后一个批次
                        if not is_last_batch: # 如果不是最后一个批次
                            # log of last step is combined with validation and rollout # 最后一步的日志与验证和rollout结合
                            wandb_run.log(step_log, step=self.global_step) # 记录wandb日志
                            json_logger.log(step_log) # 记录json日志
                            self.global_step += 1 # 全局步数加1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1): # 如果达到最大训练步数
                            break # 跳出循环

                # at the end of each epoch # 每个epoch结束时
                # replace train_loss with epoch average # 用epoch平均值替换训练损失
                train_loss = np.mean(train_losses) # 计算训练损失平均值
                step_log['train_loss'] = train_loss # 更新步日志中的训练损失

                # ========= eval for this epoch ========== # 评估当前epoch
                policy = self.model # 获取模型策略
                if cfg.training.use_ema: # 如果使用EMA
                    policy = self.ema_model # 获取EMA模型策略
                policy.eval() # 设置模型为评估模式

                # run rollout # 运行rollout
                if (self.epoch % cfg.training.rollout_every) == 0: # 如果到达rollout间隔
                    runner_log = env_runner.run(policy) # 运行环境并获取日志
                    # log all # 记录所有日志
                    step_log.update(runner_log) # 更新步日志

                # run validation # 运行验证
                if (self.epoch % cfg.training.val_every) == 0: # 如果到达验证间隔
                    with torch.no_grad(): # 不计算梯度
                        val_losses = list() # 初始化验证损失列表
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch: # 创建tqdm进度条
                            for batch_idx, batch in enumerate(tepoch): # 遍历验证数据加载器
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True)) # 数据转移到设备
                                loss = self.model.compute_loss(batch) # 计算损失
                                val_losses.append(loss) # 添加损失到验证损失列表
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1): # 如果达到最大验证步数
                                    break # 跳出循环
                        if len(val_losses) > 0: # 如果有验证损失
                            val_loss = torch.mean(torch.tensor(val_losses)).item() # 计算验证损失平均值
                            # log epoch average validation loss # 记录epoch平均验证损失
                            step_log['val_loss'] = val_loss # 更新步日志中的验证损失

                # run diffusion sampling on a training batch # 在训练批次上运行扩散采样
                if (self.epoch % cfg.training.sample_every) == 0: # 如果到达采样间隔
                    with torch.no_grad(): # 不计算梯度
                        # sample trajectory from training set, and evaluate difference # 从训练集中采样轨迹，并评估差异
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True)) # 获取训练采样批次
                        obs_dict = batch['obs'] # 构建观测字典
                        gt_action = batch['action'] # 获取真实动作
                        
                        result = policy.predict_action(obs_dict) # 预测动作
                        pred_action = result['action_pred'] # 获取预测动作
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action) # 计算均方误差
                        step_log['train_action_mse_error'] = mse.item() # 更新步日志中的均方误差
                        del batch # 释放内存
                        del obs_dict # 释放内存
                        del gt_action # 释放内存
                        del result # 释放内存
                        del pred_action # 释放内存
                        del mse # 释放内存
                
                # checkpoint # 检查点
                if (self.epoch % cfg.training.checkpoint_every) == 0: # 如果到达检查点间隔
                    # checkpointing # 保存检查点
                    if cfg.checkpoint.save_last_ckpt: # 如果保存最后一个检查点
                        self.save_checkpoint() # 保存检查点
                    if cfg.checkpoint.save_last_snapshot: # 如果保存最后一个快照
                        self.save_snapshot() # 保存快照

                    # sanitize metric names # 清理度量名称
                    metric_dict = dict() # 初始化度量字典
                    for key, value in step_log.items(): # 遍历步日志
                        new_key = key.replace('/', '_') # 替换斜杠为下划线
                        metric_dict[new_key] = value # 更新度量字典
                    
                    # We can't copy the last checkpoint here # 这里不能复制最后一个检查点
                    # since save_checkpoint uses threads. # 因为save_checkpoint使用线程
                    # therefore at this point the file might have been empty! # 因此此时文件可能是空的！
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict) # 获取TopK检查点路径

                    if topk_ckpt_path is not None: # 如果TopK检查点路径不为空
                        self.save_checkpoint(path=topk_ckpt_path) # 保存检查点
                # ========= eval end for this epoch ========== # 评估当前epoch结束
                policy.train() # 设置模型为训练模式

                # end of epoch # epoch结束
                # log of last step is combined with validation and rollout # 最后一步的日志与验证和rollout结合
                wandb_run.log(step_log, step=self.global_step) # 记录wandb日志
                json_logger.log(step_log) # 记录json日志
                self.global_step += 1 # 全局步数加1
                self.epoch += 1 # epoch加1

@hydra.main( # 定义hydra主函数
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
# 实例化 TrainDiffusionUnetImageWorkspace 类并运行
def main(cfg): # 主函数
    workspace = TrainDiffusionUnetImageWorkspace(cfg) # 创建TrainDiffusionUnetImageWorkspace实例
    workspace.run() # 运行工作区

if __name__ == "__main__": # 主程序入口
    main() # 调用主函数
