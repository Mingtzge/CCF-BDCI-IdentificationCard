import cv2
import os
from scipy import stats
from math import *
import numpy as np
import pandas as pd


def CannyThreshold(lowThreshold, ori_img, gray):
    # 阈值自适应二值化
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(ori_img, ori_img, mask=detected_edges)  # just add some colours to edges from original image.
    return dst


def drop_columns(data):
    for c in data.columns:
        if data[c].sum() < threshold:
            data.drop(columns=[c], inplace=True)
    data.drop(columns=[e], inplace=True)  # 删除最后一列
    return data


def drop_rows(img):
    # 截出有效面积,用于找到顶点和计算斜率
    img_new = np.array(img, dtype='float32')
    e, g = img_new.shape[:2]
    arr2 = img_new.sum(axis=1)  # 每一行求和
    df = pd.DataFrame(img_new)  # 把像素点转化为dataframe
    df.insert(len(df.columns), len(df.columns), arr2)  # 最后一列插入每一行的和
    df1 = pd.concat([df, (pd.DataFrame(df.sum()).T)])  # 最后一行插入每一列的和
    img1_y1 = -1
    img1_y2 = -1
    img_1 = img_2 = df
    for index, value in enumerate(df1[e]):
        if value > 100000:
            continue
        if img1_y1 == -1:
            if value > threshold:
                img1_y1 = index
        elif value < threshold:
            img1_y2 = index
            img_1 = df1[img1_y1:img1_y2]
            break
    img2_y1 = -1
    for index, value in enumerate(df1[e]):
        if value > 100000 or index < img1_y2:
            continue
        if img2_y1 == -1:
            if value > threshold:
                img2_y1 = index
        elif value < threshold:
            img2_y2 = index
            img_2 = df1[img2_y1:img2_y2]
            break
    for c in img_1.columns:
        if img_1[c].sum() < threshold or img_1[c].sum() > 30000:
            img_1.drop(columns=[c], inplace=True)
    for c in img_2.columns:
        if img_2[c].sum() < threshold or img_2[c].sum() > 30000:
            img_2.drop(columns=[c], inplace=True)
    img_1_point = [[img_1.columns[0], img1_y1], [img_1.columns[-1], img1_y2]]
    img_2_point = [[img_2.columns[0], img2_y1], [img_2.columns[-1], img2_y2]]
    return img_1, img_2, img_1_point, img_2_point


# 旋转angle角度，缺失背景白色（255, 255, 255）填充
def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    h, w = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def cal_slope(img):
    rows, cols = img.shape
    thr_f = 100
    # b1 = 0
    points1 = []
    for i in range(int(cols / 8 * 3), int(cols / 8 * 5)):
        for y in range(rows):
            if img.iloc[y, i] > thr_f:
                points1.append([i, y])
                break
    slope = stats.linregress(points1)
    slope1 = slope.slope
    # flag = True
    # for i in range(rows):
    #     if not flag:
    #         break
    #     for y in range(cols):
    #         if img.iloc[i, y] > thr_f:
    #             b1 = int(i - slope1 * y)
    #             print(i, y)
    #             flag = False
    #             break
    degree = degrees(atan(slope1))
    return degree


def process(img_path):
    ori_img = cv2.imread(img_path)
    bak_img = ori_img
    gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    img = CannyThreshold(0, ori_img, gray)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    e_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img1, img2, point1, point2 = drop_rows(img)
    img1_degree = cal_slope(img1)
    img2_degree = cal_slope(img2)
    img1_rotated = rotate_bound_white_bg(bak_img, img1_degree)
    img1_gray_rotated = rotate_bound_white_bg(np.array(img), img1_degree)
    img1_gray_rotated = cv2.erode(img1_gray_rotated, e_kernel)
    img1_gray_rotated = cv2.dilate(img1_gray_rotated, d_kernel)
    img1_gray_rotated = cv2.erode(img1_gray_rotated, e_kernel)
    img1_gray_rotated = cv2.erode(img1_gray_rotated, e_kernel)
    ret, img1_gray_rotated = cv2.threshold(img1_gray_rotated, 20, 255, cv2.THRESH_BINARY)
    img1_gray_rotated = cv2.dilate(img1_gray_rotated, d_kernel)
    img1, img2, point1, point2 = drop_rows(img1_gray_rotated)
    img1_final = img1_rotated[point1[0][1]:point1[1][1], point1[0][0]:point1[1][0]]
    img2_rotated = rotate_bound_white_bg(bak_img, img2_degree)
    img2_gray_rotated = rotate_bound_white_bg(np.array(img), img2_degree)
    img2_gray_rotated = cv2.erode(img2_gray_rotated, e_kernel)
    img2_gray_rotated = cv2.dilate(img2_gray_rotated, d_kernel)
    img2_gray_rotated = cv2.erode(img2_gray_rotated, e_kernel)
    img2_gray_rotated = cv2.erode(img2_gray_rotated, e_kernel)
    ret, img2_gray_rotated = cv2.threshold(img2_gray_rotated, 20, 255, cv2.THRESH_BINARY)
    img2_gray_rotated = cv2.dilate(img2_gray_rotated, d_kernel)
    img2_gray_rotated = cv2.dilate(img2_gray_rotated, d_kernel)
    img1, img2, point1, point2 = drop_rows(img2_gray_rotated)
    img2_final = img2_rotated[point2[0][1]:point2[1][1], point2[0][0]:point2[1][0]]
    return img1_final, img2_final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the original image path")
    parser.add_argument("-o", "--output", help="the output directory of result images")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    lowThreshold = 0
    threshold = 2550
    max_lowThreshold = 100
    ratio = 3
    kernel_size = 3
    if not os.path.exists(input_path):
        print("the input directory not exists, exit!!")
        exit(0)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ori_files = os.listdir(input_path)
    for ori_file in ori_files:
        i_path = os.path.join(input_path, ori_file)
        try:
            img1, img2 = process(i_path)
            print("process %s successful" % i_path)
            o_img1_path = os.path.join(output_path, ori_file[:-4] + "_1" + ".png")
            o_img2_path = os.path.join(output_path, ori_file[:-4] + "_2" + ".png")
            cv2.imwrite(o_img1_path, img1)
            cv2.imwrite(o_img2_path, img2)
        except:
            print("process %s failed" % i_path)
