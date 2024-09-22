from gym import spaces # 导入spaces模块
from diffusion_policy.env.pusht.pusht_env import PushTEnv # 导入PushTEnv类
import numpy as np # 导入numpy模块
import cv2 # 导入cv2模块

class PushTImageEnv(PushTEnv): # 定义PushTImageEnv类，继承自PushTEnv
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10} # 设置元数据，包括渲染模式和视频帧率
    # 调用父类 PushTEnv 的初始化方法，设置窗口大小和观测空间，初始化渲染缓存
    def __init__(self, # 初始化方法
            legacy=False, # 传统模式
            block_cog=None, # 块中心
            damping=None, # 阻尼
            render_size=96): # 渲染尺寸
        super().__init__( # 调用父类初始化方法
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size # 获取窗口大小
        self.observation_space = spaces.Dict({ # 定义观测空间
            'image': spaces.Box( # 图像空间
                low=0, # 最小值
                high=1, # 最大值
                shape=(3,render_size,render_size), # 形状
                dtype=np.float32 # 数据类型
            ),
            'agent_pos': spaces.Box( # 代理位置空间
                low=0, # 最小值
                high=ws, # 最大值
                shape=(2,), # 形状
                dtype=np.float32 # 数据类型
            )
        })
        self.render_cache = None # 初始化渲染缓存
    
    # 获取当前帧的图像观测和代理位置，处理图像数据，并将最新动作绘制到图像上，更新渲染缓存。
    def _get_obs(self): # 获取观测的方法
        img = super()._render_frame(mode='rgb_array') # 获取渲染帧

        agent_pos = np.array(self.agent.position) # 获取代理位置
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0) # 调整图像轴顺序，并归一化到[0,1]范围
        obs = { # 构建观测字典
            'image': img_obs,
            'agent_pos': agent_pos
        }
        # 画出动作
        if self.latest_action is not None: # 如果有最新动作
            action = np.array(self.latest_action) # 获取最新动作
            coord = (action / 512 * 96).astype(np.int32) # 计算坐标
            marker_size = int(8/96*self.render_size) # 计算标记大小
            thickness = int(1/96*self.render_size) # 计算厚度
            cv2.drawMarker(img, coord, # 在图像上画标记
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img # 更新渲染缓存
        return obs # 返回观测

    # 确保渲染模式为 'rgb_array'，如果渲染缓存为空则获取观测，返回渲染缓存。
    def render(self, mode): # 渲染方法
        assert mode == 'rgb_array' # 确保渲染模式为'rgb_array'

        if self.render_cache is None: # 如果渲染缓存为空
            self._get_obs() # 获取观测
        
        return self.render_cache # 返回渲染缓存
