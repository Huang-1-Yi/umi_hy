import cv2
import numpy as np
import pyrealsense2 as rs
import os

images = []  # 创建一个列表来存储图像

try:
    # 创建 pipeline
    pipeline = rs.pipeline()

    # 创建配置，并请求彩色流
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)  # 请求彩色流
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)  # 请求深度流

    # 启动 pipeline
    # profile = pipeline.start(config)
    pipeline.start(config)

    frame_count = 0  # 用于计数读取了多少帧图像
    while True:
        # 等待新帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 将帧转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())

        # 将图像添加到列表
        images.append(color_image)

        frame_count += 1
        print(f"Captured frame {frame_count}")

        # 使用 OpenCV 进行处理 (这里只是简单地显示图像)
        cv2.imshow('RealSense Color Stream', color_image)


        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 停止 pipeline
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"Total captured images: {len(images)}")

# 创建图像存储文件夹（如果不存在）
image_folder = os.path.join('tests', 'image')
os.makedirs(image_folder, exist_ok=True)

# 保存前5张图片到 "image" 文件夹
for i, img in enumerate(images[:5]):
    filename = os.path.join(image_folder, f"captured_image_{i}.jpg") # 将文件名拼接到 image 文件夹下
    cv2.imwrite(filename, img)
    print(f"Saved image: {filename}")