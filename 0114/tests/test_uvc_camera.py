import sys
import os
# 导入sys和os模块，这两个模块提供了一种使用操作系统依赖功能的接口。
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)
# 设置项目的根目录，并将该目录添加到Python的模块搜索路径中，以确保可以正确导入项目中的模块。
# 同时将当前工作目录切换到根目录。
import cv2
import json
import time
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera, VideoRecorder
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from polymetis import RobotInterface
# 导入必要的模块，包括OpenCV（用于图像处理），json（用于处理JSON数据），time（用于时间相关的功能），
# 以及其他一些自定义模块，如SharedMemoryManager（用于进程间共享内存管理），UvcCamera和VideoRecorder（用于视频录制），
# usb_util（用于USB设备操作），以及RobotInterface（可能是用于机器人控制的接口）。

# 定义一个名为test的函数。
def test():
    # 重置所有Elgato捕获卡以解决固件错误。获取所有V4L（Video for Linux）设备的路径，并选择第一个路径。
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()
    v4l_paths = get_sorted_v4l_paths()
    v4l_path = v4l_paths[0]
    # 使用上下文管理器创建一个SharedMemoryManager实例，用于管理共享内存。
    with SharedMemoryManager() as shm_manager:
        # video_recorder = VideoRecorder.create_h264(
        #     shm_manager=shm_manager,
        #     fps=30,
        #     codec='h264_nvenc',
        #     input_pix_fmt='bgr24',
        #     thread_type='FRAME',
        #     thread_count=4
        # )
        # 创建一个VideoRecorder实例，用于视频录制。设置帧率为30，编解码器为h264_nvenc，像素格式为bgr24，比特率为6000kbps。
        video_recorder = VideoRecorder(
            shm_manager=shm_manager,
            fps=30,
            codec='h264_nvenc',
            input_pix_fmt='bgr24',
            bit_rate=6000*1000
        )
        # 创建一个UvcCamera实例，设置共享内存管理器，视频设备路径，分辨率，捕获帧率，视频录制器，是否降采样和是否打印详细信息。
        
        with UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=v4l_path,
            resolution=(1920, 1080),
            capture_fps=30,
            video_recorder=video_recorder,
            put_downsample=False,
            verbose=True
        ) as camera:
            # 设置OpenCV使用单线程。
            cv2.setNumThreads(1) 
             # 设置视频文件路径，计算录制开始时间（当前时间加2秒），并开始录制视频
            video_path = 'data_local/test.mp4'
            rec_start_time = time.time() + 2
            camera.start_recording(video_path, start_time=rec_start_time)
            # 无限循环，从相机获取数据，并打印时间戳信息。如果按下'q'键，则退出循环；如果按下'r'键，则开始录制视频；如果按下's'键，则停止录制。
            # 循环将在录制开始5秒后结束。
            data = None
            while True:
                data = camera.get(out=data)
                t = time.time()
                # print('capture_latency', data['receive_timestamp']-data['capture_timestamp'], 'receive_latency', t - data['receive_timestamp'])
                # print('receive', t - data['receive_timestamp'])

                dt = time.time() - data['timestamp']
                # print(dt)
                print(data['camera_capture_timestamp'] - data['camera_receive_timestamp'])

                bgr = data['color']
                # print(bgr.shape)
                # cv2.imshow('default', bgr)
                # key = cv2.pollKey()
                # if key == ord('q'):
                #     break
                # elif key == ord('r'):
                #     video_path = 'data_local/test.mp4'
                #     realsense.start_recording(video_path)
                # elif key == ord('s'):
                #     realsense.stop_recording()
                
                time.sleep(1/60)
                if time.time() > (rec_start_time + 5.0):
                    break


if __name__ == "__main__":
    test()
