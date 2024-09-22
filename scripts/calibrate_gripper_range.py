# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
"""
导入Click库用于命令行参数处理，collections模块用于创建默认字典，pickle模块用于读取.pkl文件，json模块用于写入JSON文件，numpy模块用于数学运算，以及从umi.common.cv_util模块导入get_gripper_width函数
"""
import click
import collections
import pickle
import json
import numpy as np
from umi.common.cv_util import get_gripper_width

# %%
"""
使用Click库定义命令行接口，并设置四个命令行选项：input（标签检测结果的.pkl文件路径），output（输出JSON文件的路径），tag_det_threshold（标签检测阈值），nominal_z（抓手指尖标签的标称Z值）。
"""
@click.command()
@click.option('-i', '--input', required=True, help='Tag detection pkl')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-t', '--tag_det_threshold', type=float, default=0.8)
@click.option('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag")
def main(input, output, tag_det_threshold, nominal_z):
    tag_detection_results = pickle.load(open(input, 'rb'))# 读取命令行参数指定的.pkl文件，这个文件包含了标签检测结果
    
    # identify gripper hardware id 
    # 遍历所有帧，统计每个标签出现的次数
    n_frames = len(tag_detection_results)
    tag_counts = collections.defaultdict(lambda: 0)
    for frame in tag_detection_results:
        for key in frame['tag_dict'].keys():
            tag_counts[key] += 1
    # 计算每个标签出现的频率
    tag_stats = collections.defaultdict(lambda: 0.0)
    for k, v in tag_counts.items():
        tag_stats[k] = v / n_frames
    # 根据标签ID的最大值和每个抓手上的标签数量，计算可能的抓手ID的最大值
    max_tag_id = np.max(list(tag_stats.keys()))
    tag_per_gripper = 6
    max_gripper_id = max_tag_id // tag_per_gripper
    # 遍历所有可能的抓手ID，计算每个抓手被检测到的概率，并将概率存储在字典中
    gripper_prob_map = dict()
    for gripper_id in range(max_gripper_id+1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        left_prob = tag_stats[left_id]
        right_prob = tag_stats[right_id]
        gripper_prob = min(left_prob, right_prob)
        if gripper_prob <= 0:
            continue
        gripper_prob_map[gripper_id] = gripper_prob
    # 如果没有检测到任何抓手，则打印错误消息并退出程序
    if len(gripper_prob_map) == 0:
        print("No grippers detected!")
        exit(1)
    # 根据概率对抓手进行排序，并选择概率最高的抓手ID
    gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[1])
    gripper_id = gripper_probs[-1][0]
    gripper_prob = gripper_probs[-1][1]
    print(f"Detected gripper id: {gripper_id} with probability {gripper_prob}")
    # 如果检测概率低于阈值，则打印错误消息并退出程序
    if gripper_prob < tag_det_threshold:
        print(f"Detection rate {gripper_prob} < {tag_det_threshold} threshold.")
        exit(1)
     
    # run calibration
    # 根据抓手ID计算左右指尖标签的ID
    left_id = gripper_id * tag_per_gripper
    right_id = left_id + 1
    # 遍历所有帧，计算抓手的宽度，并将结果存储在列表中
    gripper_widths = list()
    for i, dt in enumerate(tag_detection_results):
        tag_dict = dt['tag_dict']
        width = get_gripper_width(tag_dict, left_id, right_id, nominal_z=nominal_z)
        if width is None:
            width = float('Nan')
        gripper_widths.append(width)
    gripper_widths = np.array(gripper_widths)
    max_width = np.nanmax(gripper_widths)
    min_width = np.nanmin(gripper_widths)
    # 输出标定结果
    result = {
        'gripper_id': gripper_id,
        'left_finger_tag_id': left_id,
        'right_finger_tag_id': right_id,
        'max_width': max_width,
        'min_width': min_width
    }
    json.dump(result, open(output, 'w'), indent=2)

# %%
if __name__ == "__main__":
    main()
