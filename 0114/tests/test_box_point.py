
'''
1、读取一张图片
2、图片灰度化
3、Gaussion：高斯滤波：图片平滑，去噪声
4、dilate：边缘膨胀
5、canny边缘检测
6、findContours：轮廓检测
7、从轮廓中寻找最合适的顶点
8、以顶点为圆心，在原图上标记出来
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import os

# 1、读取一张图片
image_folder = os.path.join('tests', 'image') # 实际的绝对路径 ('/home/robot/umi/tests/', 'image')
# 你要读取的图像文件名
image_name = 'captured_image_0.jpg'  
# image_name = '1.jpg'
image_path = os.path.join(image_folder, image_name)# 构建完整的图片路径'/home/robot/umi/tests/image/captured_image_0.jpg'
print(f"Reading image from {image_path}")

def save_image(image, image_name, folder='tests/image'):
    filename = os.path.join(image_folder, image_name) # 将文件名拼接到 image 文件夹下
    cv2.imwrite(filename, image)
    print(f"Saved image: {filename}")

def save_txt(corners_nms):
  # 存储角点位置到文本文件
  with open(f"{image_path}{image_name}_corners.txt", "w") as file:
      for corner in corners_nms:
          file.write(f"{corner[0]} {corner[1]}\n")

def crop_center_quarter(img):
    """
    裁剪图像为正中心的 2/3 画面。

    Args:
      img: 输入图像，numpy array。

    Returns:
      裁剪后的图像，numpy array。
    """

    height, width = img.shape[:2] # 获取高度和宽度，通道数不需要

    # 计算裁剪区域的起始坐标
    x = width // 6  # 使用整除，避免浮点数
    y = height // 6

    # 计算裁剪区域的宽度和高度
    crop_width = width*2 // 3
    crop_height = height*2 // 3

    # 执行裁剪
    cropped_img = img[y:y+crop_height, x:x+crop_width]

    return cropped_img

def img_gray(img):
    """
    图像灰度化。

    Args:
      img: 输入图像，numpy array。

    Returns:
      灰度化后的图像，numpy array。
      
    """
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    save_image(img,f"{image_name}_1_gray.jpg",image_folder)
    return img

def otsu(img_otsu):
    h, w = img_otsu.shape[:2]
    pixel = h * w
    threshold_k = 0
    max_var = .0

    for k in range(255):
        c1 = img_otsu[img_otsu <= k]

        p1 = len(c1) / pixel
        if p1 == 0:
            continue
        elif p1 == 1:
            break
        MG = np.sum(img_otsu) / pixel
        m = np.sum(c1) / pixel
        d = (MG*p1 - m) ** 2 / (p1 * (1 - p1))
        if d > max_var:
            max_var = d
            threshold_k = k

    img_otsu[img_otsu <= threshold_k] = 0
    img_otsu[img_otsu > threshold_k] = 255

    print(f"{threshold_k}")
    plt.imshow(img_otsu, cmap='gray')
    plt.show()
    save_image(img_otsu,f"{image_name}_2_otsu_threshold.jpg",image_folder)
    return img_otsu

def img_threshold(img_threshold):
    """
    图像二值化。

    Args:
      img: 输入图像，numpy array。

    Returns:
      二值化后的图像，numpy array。
    """
    # 二值化
    ret, img = cv2.threshold(img_threshold, 92, 255, cv2.THRESH_BINARY)
    save_image(img,f"{image_name}_2_BINARY_threshold.jpg",image_folder)
    ret, img = cv2.threshold(img_threshold, 92, 255, cv2.ADAPTIVE_THRESH_MEAN_C)
    save_image(img,f"{image_name}_2_ADAPTIVE_MEAN_threshold.jpg",image_folder)
    ret, img = cv2.threshold(img_threshold, 92, 140, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    save_image(img,f"{image_name}_2_ADAPTIVE_GAUSSIAN_threshold.jpg",image_folder)

    return img

def img_Gauss(img):
    """
    图像高斯滤波。

    Args:
      img: 输入图像，numpy array。

    Returns:
      高斯滤波后的图像，numpy array。
    """
    # return cv2.GaussianBlur(img, (5, 5), 0)
    blurred = cv2.GaussianBlur(img,(3,3),0)            # 5,5
    save_image(blurred,f"{image_name}_3_Gaussian Blurred.jpg",image_folder)
    return blurred
    # # 绘图：调试过程中看图片处理成什么样的
    # plt.figure()
    # plt.imshow(blurred.astype(np.uint8), cmap='gray')
    # plt.axis('off')
    # plt.title("Gaussian Blurred")
    # plt.show()

def img_dilate(img):
    """
    图像膨胀。

    Args:
      img: 输入图像，numpy array。

    Returns:
      膨胀后的图像，numpy array。
    """
    # getStructuringElement:获取结构化元素
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # 十字结构
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    # 使用较大结构元素可以合并细小的边缘，但也会使边缘变得更加模糊
    img_dilate = cv2.dilate(img,cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))# 100 100
    save_image(img_dilate,f"{image_name}_4_Dilated.jpg",image_folder)
    return img_dilate

def img_clahe(img):
    """
    图像直方图均衡化。

    Args:
      img: 输入图像，numpy array。

    Returns:
      直方图均衡化后的图像，numpy array。
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    save_image(img_clahe,f"{image_name}_5_clahe.jpg",image_folder)
    return img_clahe

def img_canny(img):
    """
    图像边缘检测。

    Args:
      img: 输入图像，numpy array。

    Returns:
      边缘检测后的图像，numpy array。
    """
    # 降低下限阈值，例如 10 或 20
    # 提高上限阈值，例如 180 或 200，边缘线比较平滑，降低可减少边缘断裂
    edged = cv2.Canny(img,30,80,3)# 调整 Sobel 算子的大小可能也会影响检测结果，尝试使用 3 和 5
    save_image(edged,f"{image_name}_6_Canny_Edge.jpg",image_folder)
    return edged


def find_max_connected_component(binary_image):
    """
    查找最大连通域。

    Args:
      binary_image: 二值图像，numpy array。

    Returns:
      只包含最大连通域的新二值图像，numpy array。
    """
    # 查找所有连通域
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # 计算每个连通域的面积
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        area = np.sum(labels_im == i)
        if area > max_area:
            max_area = area
            max_label = i

    # 创建一个新的二值图像，只包含最大连通域
    largest_component = np.zeros_like(binary_image)
    largest_component[labels_im == max_label] = 255
    save_image(largest_component,f"{image_name}_largest_component.jpg",image_folder)
    return largest_component

def find_second_max_connected_component(binary_image):
    """
    查找第二大的连通域。

    Args:
      binary_image: 二值图像，numpy array。
      image_name: 保存图像的名称前缀。
      image_folder: 保存图像的文件夹路径。

    Returns:
      只包含第二大连通域的新二值图像，numpy array。
    """
    # 查找所有连通域
    num_labels, labels_im = cv2.connectedComponents(binary_image)

    # 初始化最大和第二大的连通域面积以及对应的标签
    max_area = 0
    second_max_area = 0
    max_label = 0
    second_max_label = 0

    # 计算每个连通域的面积
    for i in range(1, num_labels):
        area = np.sum(labels_im == i)
        if area > max_area:
            # 更新第二大和最大连通域
            second_max_area = max_area
            second_max_label = max_label
            max_area = area
            max_label = i
        elif area > second_max_area and area < max_area:
            # 更新第二大连通域
            second_max_area = area
            second_max_label = i

    # 创建一个新的二值图像，只包含第二大连通域
    second_largest_component = np.zeros_like(binary_image)
    second_largest_component[labels_im == second_max_label] = 255
    # 保存图像
    save_image(second_largest_component, f"{image_name}_second_largest_component.jpg", image_folder)
    
    return second_largest_component

def non_max_suppression(boxes, overlapThresh):
    # 此函数用于非极大值抑制
    if len(boxes) == 0:
        return []
    
    # 如果输入的坐标是整数，则转换为浮点数
    if boxes.dtype.kind == "i":
        print("boxes is int")
        boxes = boxes.astype("float")
    
    # 初始化最终的角点列表
    pick = []
    
    # 获取每个角点的坐标
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # 计算面积，并排序
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    # 保留重叠小于设定阈值的角点
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # 找到最大的 (x, y) 轴坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # 计算重叠区域的宽高
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # 计算重叠区域的面积
        overlap = (w * h) / area[idxs[:last]]
        
        # 删除所有重叠大于设定阈值的角点
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    
    # 返回最终保留的角点
    return boxes[pick].astype("int")

# def main():
#   # https://blog.csdn.net/SESESssss/article/details/106774854
#   # https://blog.csdn.net/my_kun/article/details/106918857
#   image = cv2.imread(image_path)
#   h, w, c = image.shape
#   print('image shape --> h:%d  w:%d  c:%d' % (h, w, c))
  
#   # cv2.imshow('image', image)
#   # cv2.waitKey(2000)
#   # cv2.destroyAllWindows()

#   # harris dst
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   gray = np.float32(gray)
#   dst = cv2.cornerHarris(gray, blockSize=3, ksize=5, k=0.05)
#   image_dst = image[:, :, :]
#   image_dst[dst > 0.01 * dst.max()] = [0, 0, 255]
#   save_image(image_dst, f"{image_name}_image-corners.jpg", image_folder)
#   # cv2.imwrite('./dst.jpg', image_dst)
#   cv2.imshow('dst', image_dst)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

# def main():
#   image = cv2.imread(image_path)
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   gray = np.float32(gray)#角点检测需要float32格式的图像
#   dst = cv2.cornerHarris(gray, blockSize=3, ksize=5, k=0.05)
  
#   # 阈值化角点响应图
#   image[dst > 0.01 * dst.max()] = [0, 0, 255]
  
#   # 找到角点的坐标
#   coordinates = np.argwhere(dst > 0.01 * dst.max())
  
#   # 转换为(x, y, x, y)格式用于NMS
#   corners = np.zeros((coordinates.shape[0], 4))
#   corners[:, :2] = coordinates
#   corners[:, 2:] = coordinates
  
#   # 应用非极大值抑制
#   corners_nms = non_max_suppression(corners, 0.1)
#   save_txt(corners_nms)

#   # 在图像上绘制圆圈
#   for corner in corners_nms:
#       # center_coordinates = (int(corner[0]), int(corner[1]))
#       center_coordinates = (corner[0], corner[1])
#       cv2.circle(image, center_coordinates, 10, (0, 255, 0))# 图像,位置,半径,颜色, -1 填充圆
#       # cv2.circle(img, tuple(peak), 10, (0, 0, 255))
  
#   # 保存和显示图像
#   save_image(image, f"{image_name}_corners.jpg", image_folder)
#   cv2.imshow('dst', image)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

def main():
  img = cv2.imread(image_path)
  if img is None:
      print(f"Error: Could not read image at {image_path}")
  # else:
  #     print("Image read successfully!")
  #     cv2.imshow('Image', img)  # 显示图像，方便你确认是否成功加载
  #     # cv2.waitKey(0)
  #     # cv2.destroyAllWindows()


  # img = crop_center_quarter(img)# 裁剪图像为正中心的 2/3 画面
  img = img_gray(img)       # 灰度化
  # img_new = otsu(img)
  img = img_threshold(img)  # 二值化

  find_max_connected_component(img)
  find_second_max_connected_component(img)
  # img =img_Gauss(img)     # 高斯边缘平滑：使用较大的卷积核可以进一步平滑图像，减少细小边缘，但也会使边缘变得更加模糊，
  # img = img_dilate(img)   # 边缘膨胀
  # img = img_clahe(img)    # 自适应直方图均衡化： 增强图像对比度cv2.createCLAHE 可以进行局部自适应的直方图均衡化

  img = img_canny(img)      # canny:边缘检测

  #6、findContours:轮廓检测
  # cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性
  # cnts[0]是图中所有的轮廓
  cnts = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
  # save_image(cnts,f"{image_name}_cnts.jpg",image_folder)

  docCnt = None

  # 如果轮廓的个数是大于0的就打印出所有的轮廓
  if len(cnts) > 0:
      # 根据轮廓面积从大到小排序：所求轮廓面积较大，提升运算速度
      cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
      print(f"Found {len(cnts)} contours")
      # 按照轮廓面积从大到小，打印出所有的轮廓
      for c in cnts:
          # cv.approxPolyDP() 的参数1是源图像的某个轮廓；参数2(epsilon)是一个距离值，
          # 表示多边形的轮廓接近实际轮廓的程度，值越小，越精确；参数3表示是否闭合
          approx = cv2.approxPolyDP(c, 10, True)# 123 # 减小 epsilon 值，以更精确地逼近实际轮廓
          # 轮廓为4个点表示找到纸盒
          if len(approx) == 4:
              docCnt = approx
              break

  #7、从轮廓中寻找最
  # 打印出docCnt
  if docCnt is None:
      print("No document found")
  else:
      print(f"Document found: {docCnt}")
  # 在原图上
  # 分别打
  save_image(img,f"{image_name}_over.jpg",image_folder)

  #打印出最终的图片
  cv2.imshow('img', img)
  cv2.waitKey(0)
  save_image(img,f"{image_name}_over.jpg",image_folder)

  #打印出最终的图片
  cv2.imshow('img', img)
  cv2.waitKey(0)

if __name__ == "__main__":
    main()



