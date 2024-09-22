import time
import multiprocessing as mp
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
test_num =1

class MultiCameraVisualizer(mp.Process):
    def __init__(self,
        camera,
        row, col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True
        ):
        super().__init__()
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr=rgb_to_bgr
        self.camera = camera
        # shared variables
        self.stop_event = mp.Event()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):
        cv2.setNumThreads(1)    # 设置OpenCV使用一个线程，避免多线程操作影响图像显示
        threadpool_limits(1)    # 设置线程池限制为1，确保只有一个线程在运行
        channel_slice = slice(None)
        if self.rgb_to_bgr:     # 如果需要将RGB图像转换为BGR格式，设置channel_slice
            channel_slice = slice(None,None,-1)
        
        # 初始化vis_data和vis_img，vis_data用于存储从摄像头获取的数据，vis_img用于存储最终的图像
        vis_data = None
        vis_img = None

        # 获取OpenCV窗口的尺寸
        # 获取OpenCV窗口的尺寸
        # vis_img_height_win, vis_img_width_win = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FRAME_WIDTH), cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FRAME_HEIGHT)
        # rect = cv2.getWindowImageRect('Multi Cam Vis')
        # width = rect[2]
        # height = rect[3]

        # # 确保vis_img的尺寸与显示窗口的尺寸匹配
        # if height is not None and width is not None:
        #     vis_img = np.full((height, width, 3), fill_value=self.fill_value, dtype=np.uint8)
        # else:
        #     print("无法获取窗口尺寸，请检查OpenCV窗口是否已正确创建")

        
        while not self.stop_event.is_set():
            vis_data = self.camera.get_vis(out=vis_data)    # 从摄像头获取可视化数据，并将结果存储在vis_data中
            color = vis_data['color']
            N, H, W, C = color.shape
            global test_num 
            # test_num =1
            if test_num !=0:
                print("color.shape:", color.shape)
                test_num -=1
            assert C == 3                                   # 获取图像的形状，并检查是否为RGB格式（C == 3）
            oh = H * self.row
            ow = W * self.col
            if vis_img is None:                             # 初始化vis_img，并设置其大小和填充值
                vis_img = np.full((oh, ow, 3), 
                    fill_value=self.fill_value, dtype=np.uint8)
            for row in range(self.row):                     # 将摄像头数据按照行和列的布局放置在主图像vis_img中
                for col in range(self.col):
                    idx = col + row * self.col
                    h_start = H * row
                    h_end = h_start + H
                    w_start = W * col
                    w_end = w_start + W
                    if idx < N:
                        # opencv uses bgr
                        vis_img[h_start:h_end,w_start:w_end
                            ] = color[idx,:,:,channel_slice]
            cv2.imshow(self.window_name, vis_img)           # 使用cv2.imshow在窗口中显示图像
            cv2.pollKey()                                   # 检查是否有按键事件发生
            time.sleep(1 / self.vis_fps)                    # 等待一定的时间，以控制显示的帧率

