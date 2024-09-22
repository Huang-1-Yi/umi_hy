import time
# 精确的等待和睡眠操作
def precise_sleep(dt: float, slack_time: float=0.001, time_func=time.monotonic): # 定义精确睡眠函数，接收睡眠时间dt，松弛时间slack_time和时间函数time_func
    """
    使用时间.sleep和自旋的混合方式来最小化抖动。首先睡眠dt - slack_time秒，然后自旋剩余的时间
    Use hybrid of time.sleep and spinning to minimize jitter.
    Sleep dt - slack_time seconds first, then spin for the rest.
    """
    t_start = time_func() # 获取当前时间作为开始时间
    if dt > slack_time:     # 期望的睡眠时间 > 可接受的时间抖动
        time.sleep(dt - slack_time) # 休眠dt减去 时间抖动
    t_end = t_start + dt # 计算结束时间
    while time_func() < t_end: # 循环直到当前时间超过结束时间
        pass # 无操作，忙等待
    return # 返回

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic): # 定义精确等待函数，接收结束时间t_end，松弛时间slack_time和时间函数time_func
    t_start = time_func() # 获取当前时间作为开始时间
    t_wait = t_end - t_start # 计算需要等待的时间/期望的等待结束时间t_end
    if t_wait > 0: # 如果需要等待的时间大于0,使用time.sleep来睡眠一段时间
        t_sleep = t_wait - slack_time # 计算需要休眠的时间      (可接受的时间抖动slack_time)
        if t_sleep > 0: # 如果需要休眠的时间大于0
            time.sleep(t_sleep) # 休眠t_sleep的时间
        while time_func() < t_end: # 循环直到当前时间超过结束时间
            pass # 无操作，忙等待
    return # 返回
