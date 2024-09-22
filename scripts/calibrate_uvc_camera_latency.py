# %%
"""
这段Python脚本使用OpenCV和二维码来测量视频捕获系统的延迟。以下是脚本的主要功能的中文概述：
1. 它定义了一个`main`函数,该函数接受几个命令行选项,包括摄像头索引、二维码大小、每秒帧数（FPS）以及要捕获的帧数。
2. 它使用OpenCV初始化一个二维码检测器,并设置共享内存管理器以处理摄像头数据。
3. 它创建了一个`UvcCamera`对象来与USB视频类（UVC）摄像头接口,设置了分辨率和FPS,并指定了要捕获的最大帧数。
4. 它进入一个无限循环,从摄像头捕获帧,检测每帧中的二维码,并基于二维码中编码的时间戳和摄像头接收的时间戳计算延迟。
5. 它在屏幕上显示摄像头馈送和生成的带有当前时间戳的二维码。
6. 如果按下'c'键,它将退出循环并计算延迟的平均值和标准差。
7. 它绘制接收时间戳与二维码时间戳的关系图,并显示该图。
8. 如果按下'q'键,程序将退出。
脚本使用了以下库：
- `click`用于创建命令行界面。
- `cv2`（OpenCV）用于图像处理和二维码检测。
- `qrcode`用于生成二维码。
- `time`用于测量时间间隔。
- `numpy`用于数值运算。
- `collections`用于创建队列。
- `tqdm`用于显示进度条。
- `multiprocessing.managers`用于共享内存管理。
- `matplotlib`用于绘图。
要运行此脚本,您需要安装必要的库,并且系统上连接了兼容UVC的摄像头。脚本将捕获摄像头的帧,检测二维码,并显示摄像头馈送以及延迟测量结果。

"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import cv2
import qrcode
import time
import numpy as np
from collections import deque
from tqdm import tqdm
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.uvc_camera import UvcCamera
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from matplotlib import pyplot as plt

# %%
@click.command()
@click.option('-ci', '--camera_idx', type=int, default=0)
@click.option('-qs', '--qr_size', type=int, default=1080)# 720
@click.option('-f', '--fps', type=int, default=60)
@click.option('-n', '--n_frames', type=int, default=120)
def main(camera_idx, qr_size, fps, n_frames):
    # Find and reset all Elgato capture cards.
    # Required to workaround a firmware bug.
    reset_all_elgato_devices()
    time.sleep(0.1)
    v4l_paths = get_sorted_v4l_paths()
    print("len(v4l_paths) == ", len(v4l_paths))
    # if camera_idx >= len(v4l_paths):
    #     print("len(v4l_paths) == ", len(v4l_paths))
    #     print(f"Invalid camera index {camera_idx}. Available cameras: {v4l_paths}")
    #     return

    v4l_path = v4l_paths[camera_idx]
    get_max_k = n_frames
    detector = cv2.QRCodeDetector()
    with SharedMemoryManager() as shm_manager:
        with UvcCamera(
            shm_manager=shm_manager,
            dev_video_path=v4l_path,
            resolution=(1920, 1080),# (1280, 720),(3840, 2160),(1920, 1080),
            capture_fps=fps,
            get_max_k=get_max_k
        ) as camera:
            cv2.setNumThreads(1)
            qr_latency_deque = deque(maxlen=get_max_k)
            qr_det_queue = deque(maxlen=get_max_k)
            data = None
            while True:
                t_start = time.time()
                data = camera.get(out=data)
                cam_img = data['color']
                code, corners, _ = detector.detectAndDecodeCurved(cam_img)
                color = (0,0,255)
                if len(code) > 0:
                    color = (0,255,0)
                    ts_qr = float(code)
                    ts_recv = data['camera_receive_timestamp']
                    latency = ts_recv - ts_qr
                    qr_det_queue.append(latency)
                else:
                    qr_det_queue.append(float('nan'))
                if corners is not None:
                    cv2.fillPoly(cam_img, corners.astype(np.int32), color)
                
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                )
                t_sample = time.time()
                qr.add_data(str(t_sample))
                qr.make(fit=True)
                pil_img = qr.make_image()
                img = np.array(pil_img).astype(np.uint8) * 255
                img = np.repeat(img[:,:,None], 3, axis=-1)
                img = cv2.resize(img, (qr_size, qr_size), cv2.INTER_NEAREST)
                cv2.imshow('Timestamp QRCode', img)
                t_show = time.time()
                qr_latency_deque.append(t_show - t_sample)
                cv2.imshow('Camera', cam_img)
                keycode = cv2.pollKey()
                t_end = time.time()
                avg_latency = np.nanmean(qr_det_queue) - np.mean(qr_latency_deque)
                det_rate = 1-np.mean(np.isnan(qr_det_queue))
                print("Running at {:.1f} FPS. Recv Latency: {:.3f}. Detection Rate: {:.2f}".format(
                    1/(t_end-t_start),
                    avg_latency,
                    det_rate
                ))

                if keycode == ord('c'):
                    break
                elif keycode == ord('q'):
                    exit(0)
            data = camera.get(k=get_max_k)

        qr_recv_map = dict()
        for i in tqdm(range(len(data['camera_receive_timestamp']))):
            ts_recv = data['camera_receive_timestamp'][i]
            img = data['color'][i]
            code, corners, _ = detector.detectAndDecodeCurved(img)
            if len(code) > 0:
                ts_qr = float(code)
                if ts_qr not in qr_recv_map:
                    qr_recv_map[ts_qr] = ts_recv

        avg_qr_latency = np.mean(qr_latency_deque)
        t_offsets = [v-k-avg_qr_latency for k,v in qr_recv_map.items()]
        avg_latency = np.mean(t_offsets)
        std_latency = np.std(t_offsets)
        print(f'Capture to receive latency: AVG={avg_latency} STD={std_latency}')

        x = np.array(list(qr_recv_map.values()))
        y = np.array(list(qr_recv_map.keys()))
        y -= x[0]
        x -= x[0]
        plt.plot(x, x)
        plt.scatter(x, y)
        plt.xlabel('Receive Timestamp (sec)')
        plt.ylabel('QR Timestamp (sec)')
        plt.show()
        

# %%
if __name__ == "__main__":
    main()
