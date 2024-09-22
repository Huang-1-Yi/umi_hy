"""
Example:
用于机器人手眼标定的Python脚本
python scripts/calibrate_robot_world_hand_eye.py -i data/calibration/hand_eye_calib.pkl -o data/calibration/robot_world_hand_eye.json --intr_json data/calibration/gopro_intrinsics_1080p.json --aruco_yaml data/calibration/aruco_config.yaml --tag_id 12
"""

# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import pickle
import json
import yaml
import numpy as np
import cv2

# 处理图像和位姿转换的辅助函数
from umi.common.cv_util import (
    parse_fisheye_intrinsics, 
    convert_fisheye_intrinsics_resolution, 
    parse_aruco_config,
    detect_localize_aruco_tags,
    draw_predefined_mask
)
from umi.common.pose_util import (
    pose_to_mat, mat_to_pose
)

# %%
# 输入的pickle文件、输出的json文件、相机内参的json文件、Aruco标记配置的yaml文件和Aruco标记的ID
@click.command()
@click.option('-i', '--input', type=str, help="Load pickle file output of record_robo_world_hand_eye.py")
@click.option('-o', '--output', type=str, help="Output json file")
@click.option('-ij', '--intr_json', type=str, help="Fisheye intrinsics file")
@click.option('-ay', '--aruco_yaml', type=str, help="Aruco config file")
@click.option('-t', '--tag_id', type=int, help="ArUcO tag id")
def main(input, output, intr_json, aruco_yaml, tag_id):
    # load data
    # 加载了输入的pickle文件，该文件包含了图像和机器人末端执行器的数据
    img_eef_data = pickle.load(open(input, 'rb'))
    assert len(img_eef_data) >= 3

    # load aruco config
    # 解析了Aruco标记配置文件和相机内参文件
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load intrinsics
    # 转换了相机内参以适应实际的图像分辨率
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intr_json, 'r')))

    # convert intrinsics to actual image resolution
    # NOOP most of the time
    example_img = img_eef_data[0]['img']
    res = example_img.shape[:2][::-1]
    fisheye_intr = convert_fisheye_intrinsics_resolution(
        opencv_intr_dict=raw_fisheye_intr, target_resolution=res)
    
    # detect and localize tags
    # 在图像中检测和定位Aruco标记，并收集标记相对于相机的位姿和机器人末端执行器相对于机器人基座的位姿 
    R_world2cam = list()
    t_world2cam = list()
    R_base2gripper = list()
    t_base2gripper = list()
    for this_data in img_eef_data:
        img = this_data['img']
        tcp_pose = this_data['tcp_pose']
        draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)

        tag_dict = detect_localize_aruco_tags(
            img=img, 
            aruco_dict=aruco_dict, 
            marker_size_map=marker_size_map,
            fisheye_intr_dict=fisheye_intr)
        
        if tag_id in tag_dict:
            tag_data = tag_dict[tag_id]
            R_world2cam.append(tag_data['rvec'])
            t_world2cam.append(tag_data['tvec'])

            tx_base_grip = pose_to_mat(tcp_pose)
            tx_grip_base = np.linalg.inv(tx_base_grip)
            tv_grip_base = mat_to_pose(tx_grip_base)
            R_base2gripper.append(tv_grip_base[3:])
            t_base2gripper.append(tv_grip_base[:3])
    
    # calibrate
    # 使用OpenCV的calibrateRobotWorldHandEye函数来计算机器人基座与相机坐标系之间的转换关系
    r_b2w, t_b2w, r_g2c, t_g2c  = cv2.calibrateRobotWorldHandEye(
        R_world2cam=R_world2cam,
        t_world2cam=t_world2cam,
        R_base2gripper=R_base2gripper,
        t_base2gripper=t_base2gripper,
        # the LI algorithm is returning wrong results. Don't use it.
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    )
    tx_base2world = np.eye(4)
    tx_base2world[:3,:3] = r_b2w
    tx_base2world[:3,3] = t_b2w.squeeze()

    tx_gripper2camera = np.eye(4)
    tx_gripper2camera[:3,:3] = r_g2c
    tx_gripper2camera[:3,3] = t_g2c.squeeze()

    result = {
        'tx_base2world': tx_base2world.tolist(),
        'tx_gripper2camera': tx_gripper2camera.tolist()
    }
    # 将结果保存到输出的json文件中
    print(f'Writing calibration result to {output}')
    json.dump(result, open(output, 'w'), indent=2)

# %%
if __name__ == '__main__':
    main()

