import gym  # 导入 OpenAI Gym 库
from gym import spaces  # 从 Gym 库中导入 spaces 模块

import collections  # 导入 collections 模块
import numpy as np  # 导入 numpy 库并简写为 np
import pygame  # 导入 pygame 库
import pymunk  # 导入 pymunk 库
import pymunk.pygame_util  # 导入 pymunk 的 pygame_util 模块
from pymunk.vec2d import Vec2d  # 从 pymunk.vec2d 导入 Vec2d 类
import shapely.geometry as sg  # 导入 shapely.geometry 并简写为 sg
import cv2  # 导入 OpenCV 库
import skimage.transform as st  # 导入 skimage.transform 并简写为 st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions  # 从 diffusion_policy.env.pusht.pymunk_override 导入 DrawOptions

# 定义 pymunk_to_shapely 函数，将 pymunk 的形状转换为 shapely 的几何图形
def pymunk_to_shapely(body, shapes):  
    geoms = list()          # 创建一个空列表来存储几何图形
    for shape in shapes:    # 遍历 shapes 中的每个形状
        if isinstance(shape, pymunk.shapes.Poly):  # 如果形状是 pymunk.shapes.Poly 类型
            verts = [body.local_to_world(v) for v in shape.get_vertices()]  # 获取形状的顶点并转换为世界坐标系
            verts += [verts[0]]  # 将第一个顶点添加到顶点列表的末尾
            geoms.append(sg.Polygon(verts))  # 创建一个 shapely 的多边形并添加到 geoms 列表中
        else:  # 如果形状不是 pymunk.shapes.Poly 类型
            raise RuntimeError(f'Unsupported shape type {type(shape)}')  # 引发运行时错误，提示不支持的形状类型
    geom = sg.MultiPolygon(geoms)  # 创建一个 shapely 的多边形集合
    return geom  # 返回多边形集合

# 定义 PushTEnv 类，继承自 gym.Env
# 设置仿真参数、控制参数、观察空间和动作空间等。
class PushTEnv(gym.Env):  
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}  # 定义环境的元数据
    reward_range = (0., 1.)                 # 定义奖励范围

    def __init__(self,
            legacy=False,                   # 初始化参数 legacy，默认为 False
            block_cog=None, damping=None,   # 初始化参数 block_cog 和 damping，默认为 None
            render_action=True,             # 初始化参数 render_action，默认为 True
            render_size=96,                 # 初始化参数 render_size，默认为 96
            reset_to_state=None             # 初始化参数 reset_to_state，默认为 None
        ):
        self._seed = None                   # 初始化私有属性 _seed，默认为 None
        self.seed()                         # 调用 seed 方法
        self.window_size = ws = 512         # 初始化窗口大小为 512
        self.render_size = render_size      # 将 render_size 参数赋值给实例属性 render_size
        self.sim_hz = 100                   # 初始化仿真频率为 100 Hz
        # 本地控制参数Local controller params.
        self.k_p, self.k_v = 100, 20        # 初始化 PD 控制器的比例和微分增益
        self.control_hz = self.metadata['video.frames_per_second']      # 将元数据中的帧率赋值给控制频率
        # legcay set_state for data compatibility
        self.legacy = legacy                # 将 legacy 参数赋值给实例属性 legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),  # 定义观测空间的下界
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),     # 定义观测空间的上界
            shape=(5,),                     # 定义观测空间的形状
            dtype=np.float64                # 定义观测空间的数据类型
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),  # 定义动作空间的下界
            high=np.array([ws,ws], dtype=np.float64),  # 定义动作空间的上界
            shape=(2,),  # 定义动作空间的形状
            dtype=np.float64  # 定义动作空间的数据类型
        )

        self.block_cog = block_cog          # 将 block_cog 参数赋值给实例属性 block_cog
        self.damping = damping              # 将 damping 参数赋值给实例属性 damping
        self.render_action = render_action  # 将 render_action 参数赋值给实例属性 render_action


        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        如果您使用的是人类渲染模式，`self.window` 将是对我们绘制到的窗口的引用。
        `self.clock` 将是一个时钟，用于确保环境在人类模式下以正确的帧率渲染。
        在第一次使用人类模式之前，它们将保持为 `None`
        """
        self.window = None          # 初始化 window 属性为 None
        self.clock = None           # 初始化 clock 属性为 None
        self.screen = None          # 初始化 screen 属性为 None

        self.space = None           # 初始化 space 属性为 None
        self.teleop = None          # 初始化 teleop 属性为 None
        self.render_buffer = None   # 初始化 render_buffer 属性为 None
        self.latest_action = None   # 初始化 latest_action 属性为 None
        self.reset_to_state = reset_to_state  # 将 reset_to_state 参数赋值给实例属性 reset_to_state

    # 重置环境到初始状态，返回初始观察值。初始化仿真环境，包括设置初始状态、重心和阻尼等。
    def reset(self):
        seed = self._seed               # 获取当前种子
        self._setup()                   # 调用 _setup 方法初始化环境
        if self.block_cog is not None:          # 如果 block_cog 不为空
            self.block.center_of_gravity = self.block_cog  # 设置 block 的重心
        if self.damping is not None:            # 如果 damping 不为空
            self.space.damping = self.damping   # 设置 space 的阻尼

        state = self.reset_to_state     # 获取 reset_to_state 参数的值
        if state is None:               # 如果 state 为空
            rs = np.random.RandomState(seed=seed)  # 创建一个新的随机数生成器
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])                      # 生成一个随机状态
        self._set_state(state)          # 调用 _set_state 方法设置状态

        observation = self._get_obs()   # 获取当前观测
        return observation              # 返回观测

    # 执行一个动作，更新环境状态。包括执行物理仿真步骤、PD 控制和计算奖励等。返回新的观察值、奖励、是否完成和附加信息
    def step(self, action):  
        dt = 1.0 / self.sim_hz          # 计算时间步长
        self.n_contact_points = 0       # 重置接触点数量
        n_steps = self.sim_hz // self.control_hz  # 计算每个控制周期的仿真步数
        if action is not None:          # 如果动作不为空
            self.latest_action = action # 将动作赋值给 latest_action
            for i in range(n_steps):    # 遍历每个仿真步
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)  # 计算加速度
                self.agent.velocity += acceleration * dt  # 更新 agent 的速度
                # Step physics.
                self.space.step(dt)     # 更新物理仿真

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)            # 获取目标位置的身体对象
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)     # 将目标位置转换为 shapely 几何图形
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)   # 将 block 位置转换为 shapely 几何图形

        intersection_area = goal_geom.intersection(block_geom).area  # 计算目标位置和 block 位置的交集面积
        goal_area = goal_geom.area      # 获取目标位置的面积
        coverage = intersection_area / goal_area  # 计算覆盖率
        reward = np.clip(coverage / self.success_threshold, 0, 1)  # 计算奖励
        done = coverage > self.success_threshold  # 判断是否达到成功阈值

        observation = self._get_obs()   # 获取当前观测
        info = self._get_info()         # 获取当前信息

        return observation, reward, done, info  # 返回观测、奖励、是否完成和信息

    # 渲染当前环境状态。根据模式（例如 human 或 rgb_array）来选择渲染方法
    def render(self, mode):     # 定义 render 方法
        return self._render_frame(mode)  # 调用 _render_frame 方法进行渲染

    # 创建一个远程操作代理，用于手动控制 agent 的动作。主要用于调试或手动控制仿真中的 agent。
    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])  # 创建一个名为 TeleopAgent 的命名元组，包含一个 act 字段
        def act(obs):           # 定义 act 函数，接受一个 obs 参数
            act = None          # 初始化 act 变量为 None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)  # 获取鼠标位置并转换为 pymunk 坐标系
            if self.teleop or (mouse_position - self.agent.position).length < 30:  # 如果 teleop 为真或鼠标位置与 agent 位置的距离小于 30
                self.teleop = True  # 设置 teleop 为真
                act = mouse_position  # 将鼠标位置赋值给 act
            return act  # 返回 act
        return TeleopAgent(act)  # 返回 TeleopAgent 实例

    # 获取当前环境的观测值，返回一个包含 agent 位置、block 位置和 block 角度的数组。
    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))  # 创建一个包含 agent 位置、block 位置和 block 角度的数组
        return obs  # 返回观测数组

    # 根据给定的目标位姿，创建一个 pymunk 的身体对象，并设置其位置和角度
    def _get_goal_pose_body(self, pose):  # 定义 _get_goal_pose_body 方法
        mass = 1  # 定义质量
        inertia = pymunk.moment_for_box(mass, (50, 100))  # 计算惯量
        body = pymunk.Body(mass, inertia)  # 创建一个 pymunk 的身体对象
        body.position = pose[:2].tolist()  # 设置身体的位置
        body.angle = pose[2]  # 设置身体的角度
        return body  # 返回身体对象

    # 获取当前环境的附加信息，如 agent 的位置和速度、block 的位姿、目标位姿和接触点数量等
    def _get_info(self):  # 定义 _get_info 方法
        n_steps = self.sim_hz // self.control_hz  # 计算每个控制周期的仿真步数
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))  # 计算每步的接触点数量
        info = {
            'pos_agent': np.array(self.agent.position),  # agent 位置
            'vel_agent': np.array(self.agent.velocity),  # agent 速度
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),  # block 位姿
            'goal_pose': self.goal_pose,  # 目标位姿
            'n_contacts': n_contact_points_per_step}  # 每步的接触点数量
        return info  # 返回信息字典

    # 根据模式渲染当前帧。如果是 human 模式，会在窗口中显示环境；否则返回渲染的图像。
    def _render_frame(self, mode):  # 定义 _render_frame 方法

        if self.window is None and mode == "human":  # 如果 window 为空且模式为 human
            pygame.init()  # 初始化 pygame
            pygame.display.init()  # 初始化 pygame 显示
            self.window = pygame.display.set_mode((self.window_size, self.window_size))  # 创建显示窗口
        if self.clock is None and mode == "human":  # 如果 clock 为空且模式为 human
            self.clock = pygame.time.Clock()  # 创建一个时钟对象

        canvas = pygame.Surface((self.window_size, self.window_size))  # 创建一个绘图表面
        canvas.fill((255, 255, 255))  # 填充白色背景
        self.screen = canvas  # 将 canvas 赋值给 screen

        draw_options = DrawOptions(canvas)  # 创建绘图选项对象

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)  # 获取目标位置的身体对象
        for shape in self.block.shapes:  # 遍历 block 的形状
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]  # 获取目标位置的顶点并转换为 pygame 坐标系
            goal_points += [goal_points[0]]  # 将第一个顶点添加到顶点列表的末尾
            pygame.draw.polygon(canvas, self.goal_color, goal_points)  # 绘制多边形

        # Draw agent and block.
        self.space.debug_draw(draw_options)  # 绘制 space 中的对象

        if mode == "human":  # 如果模式为 human
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())  # 将 canvas 的内容复制到显示窗口
            pygame.event.pump()  # 处理 pygame 事件
            pygame.display.update()  # 更新显示窗口

            # the clock is already ticked during in step for "human"

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )  # 获取 canvas 的像素数组并转置
        img = cv2.resize(img, (self.render_size, self.render_size))  # 调整图像大小
        if self.render_action:  # 如果 render_action 为真
            if self.render_action and (self.latest_action is not None):  # 如果 render_action 为真且 latest_action 不为空
                action = np.array(self.latest_action)  # 将 latest_action 转换为数组
                coord = (action / 512 * 96).astype(np.int32)  # 计算动作的坐标
                marker_size = int(8/96*self.render_size)  # 计算标记大小
                thickness = int(1/96*self.render_size)  # 计算标记厚度
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)  # 在图像上绘制标记
        return img  # 返回图像

    # 关闭环境，释放资源。如果有显示窗口，会关闭显示窗口
    def close(self):  # 定义 close 方法
        if self.window is not None:  # 如果 window 不为空
            pygame.display.quit()  # 退出 pygame 显示
            pygame.quit()  # 退出 pygame
    
    # 设置环境的随机数种子，以确保结果的可复现性
    def seed(self, seed=None):  # 定义 seed 方法
        if seed is None:  # 如果 seed 为空
            seed = np.random.randint(0,25536)  # 生成一个随机种子
        self._seed = seed  # 将 seed 赋值给实例属性 _seed
        self.np_random = np.random.default_rng(seed)  # 创建一个新的随机数生成器

    # 处理碰撞事件，更新接触点数量。用于统计碰撞发生的次数或接触点数量
    def _handle_collision(self, arbiter, space, data):  # 定义 _handle_collision 方法
        self.n_contact_points += len(arbiter.contact_point_set.points)  # 更新接触点数量

    # 根据给定的状态，设置环境的状态。包括设置 agent 和 block 的位置和角度
    def _set_state(self, state):  # 定义 _set_state 方法
        if isinstance(state, np.ndarray):  # 如果 state 是 numpy 数组
            state = state.tolist()  # 将 state 转换为列表
        pos_agent = state[:2]  # 获取 agent 的位置
        pos_block = state[2:4]  # 获取 block 的位置
        rot_block = state[4]  # 获取 block 的角度
        self.agent.position = pos_agent  # 设置 agent 的位置
        if self.legacy:  # 如果 legacy 为真
            self.block.position = pos_block  # 设置 block 的位置
            self.block.angle = rot_block  # 设置 block 的角度
        else:  # 如果 legacy 为假
            self.block.angle = rot_block  # 设置 block 的角度
            self.block.position = pos_block  # 设置 block 的位置

        self.space.step(1.0 / self.sim_hz)  # 更新物理仿真

    # 根据局部状态设置环境状态，主要用于在仿射变换下转换状态
    def _set_state_local(self, state_local):  # 定义 _set_state_local 方法
        agent_pos_local = state_local[:2]  # 获取 agent 的局部位置
        block_pose_local = state_local[2:]  # 获取 block 的局部位姿
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])  # 创建目标位置的仿射变换
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )  # 创建新的仿射变换
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )  # 计算新的仿射变换
        agent_pos_new = tf_img_new(agent_pos_local)  # 计算新的 agent 位置
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])  # 创建新的状态数组
        self._set_state(new_state)  # 设置新的状态
        return new_state  # 返回新的状态

    # 初始化环境，包括创建 pymunk 空间、添加墙壁、agent、block 和目标区域等
    def _setup(self):  # 定义 _setup 方法
        self.space = pymunk.Space()  # 创建一个新的 pymunk 空间
        self.space.gravity = 0, 0  # 设置重力为零
        self.space.damping = 0  # 设置阻尼为零
        self.teleop = False  # 设置 teleop 为假
        self.render_buffer = list()  # 创建一个空的渲染缓冲区

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),  # 添加墙壁
            self._add_segment((5, 5), (506, 5), 2),  # 添加墙壁
            self._add_segment((506, 5), (506, 506), 2),  # 添加墙壁
            self._add_segment((5, 506), (506, 506), 2)  # 添加墙壁
        ]
        self.space.add(*walls)  # 将墙壁添加到空间中

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)  # 添加 agent
        self.block = self.add_tee((256, 300), 0)  # 添加 block
        self.goal_color = pygame.Color('LightGreen')  # 设置目标颜色
        self.goal_pose = np.array([256,256,np.pi/4])  # 设置目标位置

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)  # 添加碰撞处理程序
        self.collision_handeler.post_solve = self._handle_collision  # 设置碰撞后的处理函数
        self.n_contact_points = 0  # 重置接触点数量

        self.max_score = 50 * 100  # 设置最大得分
        self.success_threshold = 0.95  # 设置成功阈值为 95% 覆盖率

    # 添加一个静态的 pymunk 段，通常用于定义墙壁
    def _add_segment(self, a, b, radius):  # 定义 _add_segment 方法
        shape = pymunk.Segment(self.space.static_body, a, b, radius)  # 创建一个静态的 pymunk 段
        shape.color = pygame.Color('LightGray')  # 设置形状的颜色
        return shape  # 返回形状

    # 添加一个圆形的 pymunk 对象，通常用于定义 agent
    def add_circle(self, position, radius):  # 定义 add_circle 方法
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)  # 创建一个运动学类型的 pymunk 身体
        body.position = position  # 设置身体的位置
        body.friction = 1  # 设置摩擦力
        shape = pymunk.Circle(body, radius)  # 创建一个 pymunk 圆形
        shape.color = pygame.Color('RoyalBlue')  # 设置形状的颜色
        self.space.add(body, shape)  # 将身体和形状添加到空间中
        return body  # 返回身体

    # 添加一个矩形的 pymunk 对象，通常用于定义 block
    def add_box(self, position, height, width):  # 定义 add_box 方法
        mass = 1  # 定义质量
        inertia = pymunk.moment_for_box(mass, (height, width))  # 计算惯量
        body = pymunk.Body(mass, inertia)  # 创建一个 pymunk 的身体对象
        body.position = position  # 设置身体的位置
        shape = pymunk.Poly.create_box(body, (height, width))  # 创建一个 pymunk 的矩形
        shape.color = pygame.Color('LightSlateGray')  # 设置形状的颜色
        self.space.add(body, shape)  # 将身体和形状添加到空间中
        return body  # 返回身体

    # 添加一个 T 形的 pymunk 对象，定义一个复杂形状的 block
    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):  # 定义 add_tee 方法
        mass = 1  # 定义质量
        length = 4  # 定义长度
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]  # 定义第一个多边形的顶点
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)  # 计算第一个多边形的惯量
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]  # 定义第二个多边形的顶点
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)  # 计算第二个多边形的惯量
        body = pymunk.Body(mass, inertia1 + inertia2)  # 创建一个 pymunk 的身体对象
        shape1 = pymunk.Poly(body, vertices1)  # 创建第一个 pymunk 多边形
        shape2 = pymunk.Poly(body, vertices2)  # 创建第二个 pymunk 多边形
        shape1.color = pygame.Color(color)  # 设置第一个形状的颜色
        shape2.color = pygame.Color(color)  # 设置第二个形状的颜色
        shape1.filter = pymunk.ShapeFilter(mask=mask)  # 设置第一个形状的过滤器
        shape2.filter = pymunk.ShapeFilter(mask=mask)  # 设置第二个形状的过滤器
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2  # 设置身体的重心
        body.position = position  # 设置身体的位置
        body.angle = angle  # 设置身体的角度
        body.friction = 1  # 设置身体的摩擦力
        self.space.add(body, shape1, shape2)  # 将身体和形状添加到空间中
        return body  # 返回身体
