"""
执行SLAM标记和夹持器范围的校准操作。它接受一个或多个会话目录作为输入，针对每个目录内的数据执行一系列的校准脚本
python scripts_slam_pipeline/05_run_calibrations.py data_workspace/cup_in_the_wild/20240105_zhenjia_packard_2nd_conference_room
"""
# 导入必要的模块
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))   # 设置根目录为当前文件的上级目录的上级目录
sys.path.append(ROOT_DIR)                               # 将根目录添加到系统路径中，以便可以导入其他模块
os.chdir(ROOT_DIR)                                      # 更改当前工作目录到根目录
# 导入其他需要的模块
import click            # 用于创建命令行接口
import pathlib          # 用于路径操作
import subprocess       # 用于调用子进程执行外部命令
@click.command()        # 使用click创建命令行命令
@click.argument('session_dir', nargs=-1)                # 定义命令行参数，nargs=-1允许接受多个会话目录
def main(session_dir):  # 定义主函数
    script_dir = pathlib.Path(__file__).parent.parent.joinpath('scripts')       # 获取scripts目录的路径
    
    for session in session_dir:                         # 遍历每个会话目录
        session = pathlib.Path(session)                 # 将会话目录字符串转换为Path对象
        demos_dir = session.joinpath('demos')           # 获取demos子目录的路径
        mapping_dir = demos_dir.joinpath('mapping')     # 获取mapping子目录的路径
        slam_tag_path = mapping_dir.joinpath('tx_slam_tag.json')                # 设置SLAM标记校准结果文件的路径
            
        # 运行SLAM标记校准脚本
        script_path = script_dir.joinpath('calibrate_slam_tag.py')              # 获取校准SLAM标记的脚本路径
        assert script_path.is_file()                                            # 确保脚本文件存在
        tag_path = mapping_dir.joinpath('tag_detection.pkl')                    # 获取标记检测结果文件的路径
        assert tag_path.is_file()                                               # 确保文件存在
        csv_path = mapping_dir.joinpath('camera_trajectory.csv')                # 获取相机轨迹文件的路径
        if not csv_path.is_file():                                              # 如果文件不存在
            csv_path = mapping_dir.joinpath('mapping_camera_trajectory.csv')    # 尝试获取另一个可能的轨迹文件路径
            print("camera_trajectory.csv not found! using mapping_camera_trajectory.csv")  # 打印提示信息
        assert csv_path.is_file()                                               # 确保文件存在
        # 构建并执行校准命令
        cmd = [
            'python', str(script_path),
            '--tag_detection', str(tag_path),
            '--csv_trajectory', str(csv_path),
            '--output', str(slam_tag_path),
            '--keyframe_only'
        ]
        subprocess.run(cmd)                             # 使用subprocess执行命令
        
        # 运行夹持器范围校准脚本
        script_path = script_dir.joinpath('calibrate_gripper_range.py')         # 获取校准夹持器范围的脚本路径
        assert script_path.is_file()                                            # 确保脚本文件存在
        
        for gripper_dir in demos_dir.glob("gripper_calibration*"):              # 遍历所有夹持器校准目录
            gripper_range_path = gripper_dir.joinpath('gripper_range.json')     # 设置夹持器范围校准结果文件的路径
            tag_path = gripper_dir.joinpath('tag_detection.pkl')                # 获取标记检测结果文件的路径
            assert tag_path.is_file()                   # 确保文件存在
            # 构建并执行校准命令
            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path)
            ]
            subprocess.run(cmd)                         # 使用subprocess执行命令

# %%
if __name__ == "__main__":
    main()
