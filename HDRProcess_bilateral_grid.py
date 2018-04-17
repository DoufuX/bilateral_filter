# coding=utf-8
import numpy as np
import cv2
import sys
import math

import time
from matplotlib import pyplot as plt


# 对《High-QualityBrightnessEnhancementFunctionsforReal-TimeReverseToneMapping》轮文的复现
# 速度还比较慢（Mac上执行500*500图片约要一分半），还有点小bug（会有点色块现象）
# 且现在还只用于图片上，在普通电脑上进行实验

def get_max_intensity(input_img, channel):
    """
    :param input_img: 输入图片
    :param channel: 色彩通道
    :return: 该色彩通道最大值
    """
    max_I = 0
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            if input_img[i][j][channel] > max_I:
                max_I = input_img[i][j][channel]
    return max_I


def get_min_intensity(input_img, channel):
    """
    :param input_img: 输入图片
    :param channel: 色彩通道
    :return: 该色彩通道最小值
    """
    min_I = 255
    for i in range(input_img.shape[0]):
        for j in range(input_img.shape[1]):
            if input_img[i][j][channel] < min_I:
                min_I = input_img[i][j][channel]
    return min_I


def gaussian_kernel(dx, dy, dz, sigma_spacial, sigma_range):
    """
    :param dx: 距离中心点的x轴距离
    :param dy: 距离中心点的y轴距离
    :param dz: 距离中心点的z轴距离
    :param sigma_spacial: 空间sigma值
    :param sigma_range: 值域signma值
    :return:
    """
    Rsquared = ((dx * dx) + (dy * dy)) / (sigma_spacial * sigma_spacial) + (dz * dz) / (sigma_range * sigma_range)
    return math.e ** (-0.5 * Rsquared)


def create_gaussian_kernel_matrix(radius, sigma_spacial, sigma_range):
    """
    :param radius: 高斯半径
    :param sigma_spacial: 双边滤波的空间维度的sigma权重
    :param sigma_range: 双边滤波的值维度的sigma权重
    :return:
    """
    gaussian_kernel_matrix = np.zeros((2 * radius + 1, 2 * radius + 1, 2 * radius + 1))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                gaussian_kernel_matrix[i + radius][j + radius][k + radius] = gaussian_kernel(i, j, k, sigma_spacial,
                                                                                             sigma_range)
    return gaussian_kernel_matrix


def make_grid_channel(input_img, channel, space_sample, range_sample):
    """
    :param input_img: 输入图片数据
    :param channel: 处理的色彩通道
    :param space_sample: 空间采样率
    :param range_sample: 值采样率
    :return: 生成的初始化的对某个色彩通道的空间网格结构（用于双边网格）
    """
    tic = time.time()
    height, width = input_img.shape[0:2]
    grid_height = int(np.ceil(height / space_sample) + 1)
    grid_width = int(np.ceil(width / space_sample) + 1)
    E_range = int(np.ceil(1.0 / range_sample) + 1)

    # 运用齐次坐标形式，所以第四个维度为2,第一个纬度为Intensity，第二个纬度为count
    grid = np.zeros((grid_height, grid_width, E_range, 2))

    img_norm = input_img.copy().astype(float)

    for i in range(height):
        for j in range(width):
            img_norm[i][j][channel] /= 255.0

    # min_i = get_min_intensity(input_img, channel) / 255.0

    for i in range(height):
        row = int(round(i / space_sample))
        for j in range(width):
            col = int(round(j / space_sample))
            z = int(round(img_norm[i][j][channel] / range_sample))
            grid[row][col][z][0] += img_norm[i][j][channel]
            grid[row][col][z][1] += 1

    toc = time.time()
    print("make grid(channel) time:" + str(toc - tic))
    return grid


# TODO : change  input_img to E_value
def convol_grid(input_img, grid, space_sample, range_sample, radius, sigma_spacial, sigma_range):
    """
    :param input_img: 输入图像数据（这里只是为了提供宽、高信息）
    :param grid: 某一通道的双边网格数据结构
    :param space_sample: 空间取样率
    :param range_sample: 值域取样率
    :param radius: 进行滤波的半径
    :param sigma_spacial: 用于双边滤波的空间sigma
    :param sigma_range: 用于双边滤波的空值域sigma
    :return:
    """
    tic = time.time()
    height, width = np.shape(input_img)[0:2]
    grid_height = int(np.ceil(height / space_sample) + 1)
    grid_width = int(np.ceil(width / space_sample) + 1)

    E_range = int(np.ceil(1.0 / range_sample) + 1)

    grid_with_pad = np.pad(grid, ((radius, radius), (radius, radius), (radius, radius), (0, 0)), 'edge')

    grid_after_convol = np.zeros((grid_height, grid_width, E_range, 2))

    gaussian_kernel_matrix = create_gaussian_kernel_matrix(radius, sigma_spacial, sigma_range)
    # gaussian_kernel_weight = gaussian_kernel_matrix.sum()

    for i in range(grid_height):
        for j in range(grid_width):
            for k in range(E_range):
                temp_matrix_0 = grid_with_pad[i: i + 2 * radius + 1, j:j + 2 * radius + 1, k:k + 2 * radius + 1, 0]
                temp_matrix_1 = grid_with_pad[i: i + 2 * radius + 1, j:j + 2 * radius + 1, k:k + 2 * radius + 1, 1]
                gaussian_kernel_weight = (gaussian_kernel_matrix * temp_matrix_1).sum()
                # 比较困惑的一个点，如果该点周围有不存在的值时（gaussian_kernel_weight为0）
                if gaussian_kernel_weight == 0:
                    grid_after_convol[i][j][k][0] = 0
                else:
                    grid_after_convol[i][j][k][0] = (
                                                            gaussian_kernel_matrix * temp_matrix_0).sum() / gaussian_kernel_weight

    toc = time.time()
    print("convol grid time:" + str(toc - tic))
    return grid_after_convol


def triline_interp(slicing_img, new_image, grid_after_convol, channel, space_sample, range_sample):
    """
    :param slicing_img: 进行slicing操作的图片（slicing操作具体可看双边网格的论文）
    :param new_image: 经过slicing后得到的结果会存在该传入参数中
    :param grid_after_convol: 经过滤波卷积（convol_grid）后的双边网格数据结构
    :param channel: 双边网格数据结构对应的通道
    :param space_sample: 空间采样率
    :param range_sample: 值域采样率
    """
    tic = time.time()
    # max_intensity = get_max_intensity(raw_img, channel)
    height, width = np.shape(slicing_img)[0:2]
    for y in range(height):
        y_grid = y / space_sample
        grid_y_low = int(np.floor(y_grid))
        grid_y_high = int(np.ceil(y_grid))
        yd = 0 if grid_y_high == grid_y_low else (y_grid - grid_y_low) / (grid_y_high - grid_y_low)

        for x in range(width):
            x_grid = x / space_sample
            grid_x_low = int(np.floor(x_grid))
            grid_x_high = int(np.ceil(x_grid))
            xd = 0 if grid_x_low == grid_x_high else (x_grid - grid_x_low) / (grid_x_high - grid_x_low)

            z_val = slicing_img[y][x][channel] / 255.0 / range_sample
            grid_z_low = int(np.floor(z_val))
            grid_z_high = int(np.ceil(z_val))
            zd = 0 if grid_z_low == grid_z_high else (z_val - grid_z_low) / (grid_z_high - grid_z_low)

            c = grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * (1 - xd) * (1 - yd) * (1 - zd) + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * (1 - xd) * yd * (1 - zd) + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * xd * (1 - yd) * (1 - zd) + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * (1 - xd) * (1 - yd) * zd + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * (1 - xd) * yd * zd + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * xd * (1 - yd) * zd + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * xd * yd * (1 - zd) + \
                grid_after_convol[grid_y_low][grid_x_low][grid_z_low][0] * xd * yd * zd
            #             if np.isnan(c):
            #                 c = 0
            new_image[y][x][channel] = int(np.floor(c * 255))
    toc = time.time()
    print('triline_interp time:' + str(toc - tic))

    # def interp_image(img_raw, grid,):


# 运用双边网格实现的对三通道进行的双边滤波，在这个程序里面不会用到，只用来测试是否成功实现了双边滤波
def bilateral_grid_filter(input_img, space_sample, range_sample, gaussian_radius, sigma_spacial,
                          sigma_range):
    """
    :param input_img: 输入图像数据
    :param space_sample: 空间域取样率
    :param range_sample: 值域取样率
    :param gaussian_radius: 高斯半径
    :param sigma_spacial: 空间域sigma值
    :param sigma_range: 值域sigma值
    :return: 经过双边滤波处理后的图像
    """

    tic = time.time()

    grid_channel0 = make_grid_channel(input_img, 0, space_sample, range_sample)
    grid_channel1 = make_grid_channel(input_img, 1, space_sample, range_sample)
    grid_channel2 = make_grid_channel(input_img, 2, space_sample, range_sample)
    channel0_after_grid = convol_grid(input_img, grid_channel0, space_sample, range_sample, gaussian_radius,
                                      sigma_spacial, sigma_range)
    channel1_after_grid = convol_grid(input_img, grid_channel1, space_sample, range_sample, gaussian_radius,
                                      sigma_spacial, sigma_range)
    channel2_after_grid = convol_grid(input_img, grid_channel2, space_sample, range_sample, gaussian_radius,
                                      sigma_spacial, sigma_range)
    res_image = input_img.copy()
    triline_interp(input_img, res_image, channel0_after_grid, 0, space_sample, range_sample)
    triline_interp(input_img, res_image, channel1_after_grid, 1, space_sample, range_sample)
    triline_interp(input_img, res_image, channel2_after_grid, 2, space_sample, range_sample)

    toc = time.time()
    print('bilateral_grid_filter time:' + str(toc - tic))
    return res_image


def inverse_rgb_value(range_val):
    """
    :param range_val: 某点的值（如L值）
    :return: 归一化后的值
    """
    if range_val <= 16:
        range_val = 0
    elif range_val >= 235:
        range_val = 1
    else:
        range_val = (range_val - 16) / 219.0
    return range_val


def inverse_gamma(input_img):
    """
    :param input_img: 输入的图片数据
    :return: 经过反gamma矫正后的图片数据（公式为E' = E**2.2）
    """
    tic = time.time()
    inverse_hash_table = np.zeros(256)
    for range_val in range(256):
        range_val_norm = inverse_rgb_value(range_val)
        range_val_after_gamma = range_val_norm ** 2.2
        range_val_after_inverse = int(range_val_after_gamma * 219 + 16)
        inverse_hash_table[range_val] = range_val_after_inverse

    height, length = input_img.shape[0:2]
    output_img = input_img.copy()

    for i in range(height):
        for j in range(length):
            output_img[i][j][0] = inverse_hash_table[input_img[i][j][0]]
            output_img[i][j][1] = inverse_hash_table[input_img[i][j][1]]
            output_img[i][j][2] = inverse_hash_table[input_img[i][j][2]]

    toc = time.time()
    print("inverse_gamme_time:", str(toc - tic))
    return output_img


# 运用双边网格实现的对单通道进行的双边滤波

if __name__ == "__main__":
    img_path = r'00004.png'
    src = cv2.imread(img_path)
    img_lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    space_sample = 3
    range_sample = 0.01
    gaussian_radius = 10

    # grid_channel_l = make_grid_channel(img_lab, 0, space_sample, range_sample)
    # channel_l_after_grid = convol_grid(img_lab, grid_channel_l, space_sample, range_sample, gaussian_radius, 50.0, 5.0)
    # triline_interp(img_lab, img_lab, channel_l_after_grid, 0, space_sample, range_sample)

    I_rgb = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
    # Il = inverse_gamma(I_rgb)
    Il = I_rgb

    L1 = cv2.cvtColor(Il, cv2.COLOR_RGB2Lab)
    L1[:, :, 1:3] = 128

    L2 = Il.copy()
    tic = time.time()
    for i in range(L2.shape[0]):
        for j in range(L2.shape[1]):
            max_rgb_value = max(L2[i][j][0], L2[i][j][1], L2[i][j][2])
            if max_rgb_value > 230:
                max_rgb_value = 230
            L2[i][j][0] = max_rgb_value
            L2[i][j][1] = max_rgb_value
            L2[i][j][2] = max_rgb_value
    toc = time.time()
    print("get max_value time:" + str(toc - tic))

    L2 = cv2.cvtColor(L2, cv2.COLOR_RGB2Lab)
    L2[:, :, 1:3] = 128
    Fbe = L2.copy()  # TODO:change

    grid_channel_l_2 = make_grid_channel(L2, 0, space_sample, range_sample)
    channel_l_after_grid_2 = convol_grid(L2, grid_channel_l_2, space_sample, range_sample, 10, 100.0, 30.0)
    triline_interp(L1, Fbe, channel_l_after_grid_2, 0, space_sample, range_sample)

    Fbe_rgb = cv2.cvtColor(Fbe, cv2.COLOR_Lab2RGB)
    Fbe_rgb_float = Fbe_rgb.astype(float)

    Fbe_rgb_float = Fbe_rgb_float/255.0 * (187 / 56.0) + 1

    Il_res = (Il * Fbe_rgb_float).astype('uint8')

    # res = bilateral_grid_filter(src, 3, 0.01, 5, 10, 5)

    cv2.imwrite('res_' + img_path, cv2.cvtColor(Il_res, cv2.COLOR_RGB2BGR))
    cv2.imwrite('Fbe_' + img_path, cv2.cvtColor(Fbe_rgb, cv2.COLOR_RGB2BGR))

    cv2.imshow("res_" + img_path, cv2.cvtColor(Il_res, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
