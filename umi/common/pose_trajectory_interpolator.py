from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

# 两个旋转之间的距离(旋转的角度)————返回两个旋转之间的距离，计算方法是将 b 旋转应用到 a 旋转的逆旋转上，然后计算结果旋转的欧几里得范数(即旋转的角度)
def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    return (b * a.inv()).magnitude()

# 两个包含位置和旋转的完整姿态之间的距离
def pose_distance(start_pose, end_pose):
    # 包含位置和旋转的完整姿态————转换为 numpy 数组，分别计算位置和旋转变换
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    start_rot = st.Rotation.from_rotvec(start_pose[3:])
    end_rot = st.Rotation.from_rotvec(end_pose[3:])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist


# 为机器人提供一个连续的姿态轨迹，以便在执行任务时保持稳定的运动。
# 通过位置和旋转的插值器，可以在任何时间点获取插值后的姿态，从而实现平滑的机器人运动
# 初始化(init)、获取时间(times)、获取姿态(poses)、修剪(trim)、驱动到路点(drive_to_waypoint)、安排路点(schedule_waypoint)、插值(call)
class PoseTrajectoryInterpolator:
    # 初始化(init):接受一系列的时间点和姿态，并创建一个插值器，用于在给定的时间点上获取插值后的姿态。
    # 如果时间点数组只有一个时间点，则直接使用这个时间点和姿态
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        if len(times) == 1:         # 单步插值的特殊处理
            self.single_step = True
            self._times = times
            self._poses = poses
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])
            pos = poses[:,:3]
            rot = st.Rotation.from_rotvec(poses[:,3:])
            self.pos_interp = si.interp1d(times, 
                                          pos, 
                                          axis=0, 
                                          assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)

    # 获取时间(times):
    # 如果是一个单步插值，则返回原始的时间点。
    # 否则，返回插值器的位置插值的时间点。
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    # 获取姿态(poses):
    # 如果是一个单步插值，则返回原始的姿态。
    # 否则，根据时间插值位置，再使用旋转插值器插值旋转，得到完整姿态。
    @property
    def poses(self) -> np.ndarray:
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, 6))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:] = self.rot_interp(self.times).as_rotvec()
            return poses

    # 修剪(trim):
    # 根据给定的开始时间和结束时间，修剪插值器的时间点和姿态，生成一个新的插值器
    # 快速地生成一个在特定时间段内执行任务所需的轨迹。这有助于优化轨迹，使其更加精确和高效
    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        assert start_t <= end_t                                     # 确保开始时间 start_t <= 结束时间 end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)           # 使用 start_t 和 end_t 来确定需要保留的时间点
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])# 创建新的时间序列 all_times，包括原始轨迹的开始时间、用户指定的时间点以及结束时间
        all_times = np.unique(all_times)                            # 删除重复的时间点，确保 Slerp（一种旋转插值器）可以正确地工作
        all_poses = self(all_times)                                 # 使用新的时间序列插值姿态，创建一个新的姿态序列
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses) # 返回修剪后的时间序列和姿态

    # 驱动到路点(drive_to_waypoint):
    # 计算从当前姿态到目标姿态的最小距离和最小时间，并创建一个新的插值器，包括这个路点
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # 插入新位姿
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp
    
    # 安排路点(schedule_waypoint):
    # 安排一个新的路点，根据给定的最大速度和时间，生成一个新的插值器。
    def schedule_waypoint(self,
            pose, time, # 机械臂当前的时间戳
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None # 机械臂上一个已知姿态点的时间戳。如果提供，这个参数将被用来调整机械臂的轨迹，以确保新的姿态点插入到正确的位置
        ) -> "PoseTrajectoryInterpolator":
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end_time 应该是所有时间点（除了指定的 time）中最新（即最大）的时间点 end time should be the latest of all times except time
        # 这样在执行后续的插值操作时，就可以假设这些时间点之间的顺序是正确的 after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    # 插值(call):
    # 如果输入是一个单一的时间点，则返回该时间点的插值姿态。
    # 否则，返回所有时间点的插值姿态
    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), 6))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose = np.zeros((len(t), 6))
            pose[:,:3] = self.pos_interp(t)
            pose[:,3:] = self.rot_interp(t).as_rotvec()

        if is_single:
            pose = pose[0]
        return pose
