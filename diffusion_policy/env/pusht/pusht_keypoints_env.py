from typing import Dict, Sequence, Union, Optional # 导入类型提示
from gym import spaces # 导入spaces模块
from diffusion_policy.env.pusht.pusht_env import PushTEnv # 导入PushTEnv类
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager # 导入PymunkKeypointManager类
import numpy as np # 导入numpy模块

class PushTKeypointsEnv(PushTEnv): # 定义PushTKeypointsEnv类，继承自PushTEnv
    def __init__(self, # 初始化方法
            legacy=False, # 传统模式
            block_cog=None, # 块中心
            damping=None, # 阻尼
            render_size=96, # 渲染尺寸
            keypoint_visible_rate=1.0, # 关键点可见率
            agent_keypoints=False, # 是否使用代理关键点
            draw_keypoints=False, # 是否绘制关键点
            reset_to_state=None, # 重置状态
            render_action=True, # 渲染动作
            local_keypoint_map: Dict[str, np.ndarray]=None, # 局部关键点映射
            color_map: Optional[Dict[str, np.ndarray]]=None): # 颜色映射
        super().__init__( # 调用父类初始化方法
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size # 获取窗口大小

        if local_keypoint_map is None: # 如果局部关键点映射为空
            # create default keypoint definition # 创建默认关键点定义
            kp_kwargs = self.genenerate_keypoint_manager_params() # 生成关键点管理器参数
            local_keypoint_map = kp_kwargs['local_keypoint_map'] # 获取局部关键点映射
            color_map = kp_kwargs['color_map'] # 获取颜色映射

        # create observation spaces # 创建观测空间
        Dblockkps = np.prod(local_keypoint_map['block'].shape) # 计算块关键点维度
        Dagentkps = np.prod(local_keypoint_map['agent'].shape) # 计算代理关键点维度
        Dagentpos = 2 # 代理位置维度

        Do = Dblockkps # 初始观测维度为块关键点维度
        if agent_keypoints: # 如果使用代理关键点
            # blockkp + agnet_pos # 块关键点 + 代理位置
            Do += Dagentkps # 添加代理关键点维度
        else: # 如果不使用代理关键点
            # blockkp + agnet_kp # 块关键点 + 代理关键点
            Do += Dagentpos # 添加代理位置维度
        # obs + obs_mask # 观测 + 观测掩码
        Dobs = Do * 2 # 总观测维度为两倍的观测维度

        low = np.zeros((Dobs,), dtype=np.float64) # 创建观测空间下界
        high = np.full_like(low, ws) # 创建观测空间上界，值为窗口大小
        # mask range 0-1 # 掩码范围0-1
        high[Do:] = 1. # 设置掩码上界为1

        # (block_kps+agent_kps, xy+confidence) # (块关键点+代理关键点, 坐标+置信度)
        self.observation_space = spaces.Box( # 定义观测空间
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate # 设置关键点可见率
        self.agent_keypoints = agent_keypoints # 设置是否使用代理关键点
        self.draw_keypoints = draw_keypoints # 设置是否绘制关键点
        self.kp_manager = PymunkKeypointManager( # 创建关键点管理器
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None # 初始化绘制关键点映射

    @classmethod
    def genenerate_keypoint_manager_params(cls): # 生成关键点管理器参数的方法
        env = PushTEnv() # 创建PushTEnv实例
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env) # 从环境中创建关键点管理器
        kp_kwargs = kp_manager.kwargs # 获取关键点管理器参数
        return kp_kwargs # 返回参数

    def _get_obs(self): # 获取观测的方法
        # get keypoints # 获取关键点
        obj_map = {
            'block': self.block # 块对象
        }
        if self.agent_keypoints: # 如果使用代理关键点
            obj_map['agent'] = self.agent # 添加代理对象

        kp_map = self.kp_manager.get_keypoints_global( # 获取全局关键点
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values # Python字典保证键值顺序
        kps = np.concatenate(list(kp_map.values()), axis=0) # 拼接关键点

        # select keypoints to drop # 选择要删除的关键点
        n_kps = kps.shape[0] # 关键点数量
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate # 确定可见关键点
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1) # 创建关键点掩码

        # save keypoints for rendering # 保存用于渲染的关键点
        vis_kps = kps.copy() # 复制关键点
        vis_kps[~visible_kps] = 0 # 隐藏不可见的关键点
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])] # 块关键点映射
        }
        if self.agent_keypoints: # 如果使用代理关键点
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):] # 代理关键点映射
        self.draw_kp_map = draw_kp_map # 更新绘制关键点映射
        
        # construct obs # 构建观测
        obs = kps.flatten() # 展平关键点
        obs_mask = kps_mask.flatten() # 展平关键点掩码
        if not self.agent_keypoints: # 如果不使用代理关键点
            # passing agent position when keypoints are not available # 当关键点不可用时传递代理位置
            agent_pos = np.array(self.agent.position) # 获取代理位置
            obs = np.concatenate([
                obs, agent_pos # 拼接观测和代理位置
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool) # 拼接观测掩码和代理位置掩码
            ])

        # obs, obs_mask # 观测，观测掩码
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype) # 拼接观测和观测掩码
        ], axis=0)
        return obs # 返回观测
    
    def _render_frame(self, mode): # 渲染帧的方法
        img = super()._render_frame(mode) # 调用父类渲染帧方法
        if self.draw_keypoints: # 如果需要绘制关键点
            self.kp_manager.draw_keypoints( # 绘制关键点
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img # 返回图像
