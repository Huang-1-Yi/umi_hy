from typing import Tuple
import math
import cv2
import numpy as np
"""
导入必要的库和模块。typing用于类型注解，math提供数学计算，cv2是OpenCV库，用于图像处理，numpy用于数值计算
提供了基本的图像处理功能，如绘制标记、文本，以及调整图像大小和裁剪。最后一个函数则用于优化多个图像在特定分辨率下的排列方式。
"""
# 在图像上绘制一个十字准线（即瞄准标记）
def draw_reticle(img, u, v, label_color):
    """
    在图像上绘制准星（十字线）。
    @param img (输入/输出) uint8 3通道图像
    @param u X坐标（宽度）
    @param v Y坐标（高度）
    @param label_color 用于绘制的RGB颜色元组。
    """
    # 转换为整数
    u = int(u)
    v = int(v)

    white = (255, 255, 255)  # 定义白色
    cv2.circle(img, (u, v), 10, label_color, 1)  # 绘制第一个圆
    cv2.circle(img, (u, v), 11, white, 1)  # 绘制第二个圆
    cv2.circle(img, (u, v), 12, label_color, 1)  # 绘制第三个圆
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)  # 绘制垂直线
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)  # 绘制水平线
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)  # 绘制垂直线
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)  # 绘制水平线


# 在图像上绘制多行文本，并带有轮廓
def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    绘制带有轮廓的多行文本.
    """
    assert isinstance(text, str)  # 确保文本为字符串

    uv_top_left = np.array(uv_top_left, dtype=float)  # 转换为浮点数组
    assert uv_top_left.shape == (2,)  # 确保形状为(2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )  # 获取文本尺寸
        uv_bottom_left_i = uv_top_left + [0, h]  # 计算文本底部左侧位置
        org = tuple(uv_bottom_left_i.astype(int))  # 转换为整数元组

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )  # 绘制文本轮廓
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )  # 绘制文本

        uv_top_left += [0, h * line_spacing]  # 更新顶部左侧位置

# 获取图像的转换函数，用于调整图像大小和裁剪
def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False):

    iw, ih = input_res  # 输入分辨率宽高
    ow, oh = output_res  # 输出分辨率宽高
    rw, rh = None, None  # 调整后的宽高
    interp_method = cv2.INTER_AREA  # 插值方法

    if (iw/ih) >= (ow/oh):
        # 输入更宽
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    w_slice_start = (rw - ow) // 2  # 宽度裁剪起始位置
    w_slice = slice(w_slice_start, w_slice_start + ow)  # 宽度裁剪范围
    h_slice_start = (rh - oh) // 2  # 高度裁剪起始位置
    h_slice = slice(h_slice_start, h_slice_start + oh)  # 高度裁剪范围
    c_slice = slice(None)  # 颜色通道裁剪范围
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)  # 颜色通道翻转

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))  # 确保图像形状匹配
        # 调整大小
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # 裁剪
        img = img[h_slice, w_slice, c_slice]
        return img  # 返回调整后的图像

    return transform

# 计算最佳的行数和列数，以便将多个相机图像合理地排列在一个给定的最大分辨率区域内。
def optimal_row_cols(
        n_cameras,
        in_wh_ratio,
        max_resolution=(1920, 1080)
    ):
    out_w, out_h = max_resolution  # 输出分辨率宽高
    out_wh_ratio = out_w / out_h  # 输出宽高比
    
    n_rows = np.arange(n_cameras,dtype=np.int64) + 1  # 计算行数数组
    n_cols = np.ceil(n_cameras / n_rows).astype(np.int64)  # 计算列数数组
    cat_wh_ratio = in_wh_ratio * (n_cols / n_rows)  # 计算拼接宽高比
    ratio_diff = np.abs(out_wh_ratio - cat_wh_ratio)  # 计算宽高比差异
    best_idx = np.argmin(ratio_diff)  # 找到最佳索引
    best_n_row = n_rows[best_idx]  # 最佳行数
    best_n_col = n_cols[best_idx]  # 最佳列数
    best_cat_wh_ratio = cat_wh_ratio[best_idx]  # 最佳拼接宽高比

    rw, rh = None, None
    if best_cat_wh_ratio >= out_wh_ratio:
        # 拼接更宽
        rw = math.floor(out_w / best_n_col)
        rh = math.floor(rw / in_wh_ratio)
    else:
        rh = math.floor(out_h / best_n_row)
        rw = math.floor(rh * in_wh_ratio)
    
    # 返回最佳分辨率和行列数
    return rw, rh, best_n_col, best_n_row