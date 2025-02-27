"""
Usage:
python test_exposure_hy.py
显示默认曝光光和白平衡
"""


# import pyrealsense2 as rs

# # 创建一个管道
# pipeline = rs.pipeline()

# # 启动相机流
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

# # 获取当前的曝光和白平衡设置
# try:
#     # 获取设备
#     device = pipeline.get_active_profile().get_device()

#     # 获取颜色传感器
#     color_sensor = device.first_color_sensor()

#     # 获取当前的曝光和白平衡
#     current_exposure = color_sensor.get_option(rs.option.exposure)
#     current_white_balance = color_sensor.get_option(rs.option.white_balance)

#     print(f"Current Exposure: {current_exposure}")
#     print(f"Current White Balance: {current_white_balance}")

#     # 根据需要重新设置曝光和白平衡
#     # color_sensor.set_option(rs.option.exposure, new_exposure_value)
#     # color_sensor.set_option(rs.option.white_balance, new_white_balance_value)

# finally:
#     # 停止相机流
#     pipeline.stop()

import pyrealsense2 as rs
import time

# 创建一个管道
pipeline = rs.pipeline()

# 启动相机流
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 等待一段时间，让相机自动调整曝光和白平衡
time.sleep(5)  # 延迟5秒，等待自动调整完成

# 获取当前的曝光和白平衡设置
try:
    # 获取设备
    device = pipeline.get_active_profile().get_device()

    # 获取颜色传感器
    color_sensor = device.first_color_sensor()

    # 获取当前的曝光和白平衡
    current_exposure = color_sensor.get_option(rs.option.exposure)
    current_white_balance = color_sensor.get_option(rs.option.white_balance)

    print(f"Current Exposure: {current_exposure}")
    print(f"Current White Balance: {current_white_balance}")

finally:
    # 停止相机流
    pipeline.stop()
