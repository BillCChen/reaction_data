def label_connected_regions(grid):
    if not grid or not grid[0]:
        return
    
    rows, cols = len(grid), len(grid[0])
    current_label = 1  # 当前的标定序号，从1开始
    
    def dfs(x, y, label):
        # 检查边界以及当前格点是否为-1
        if (x < 0 or x >= rows or 
            y < 0 or y >= cols or 
            grid[x][y] != -1):
            return
        
        # 将当前格点的-1改为标定值
        grid[x][y] = label
        
        # 递归检查上下左右四个方向
        dfs(x-1, y, label)  # 上
        dfs(x+1, y, label)  # 下
        dfs(x, y-1, label)  # 左
        dfs(x, y+1, label)  # 右
    
    # 从左上到右下扫描
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == -1:
                # 遇到-1时进入标定状态，使用DFS标记整个连通区域
                dfs(i, j, current_label)
                current_label += 1  # 标定完成后序号加1
    
    return  # 不返回新数组，直接修改输入的grid

from PIL import Image
import numpy as np
from tqdm import tqdm,trange
# def label_connected_regions(grid):
#     if grid.size == 0 or grid[0].all() == 0:
#         return
#     print(f"grid shape: {grid.shape}")
#     print(f"rows: {len(grid)}, cols: {len(grid[0])}")
#     rows, cols = len(grid), len(grid[0])
#     current_label = 1  # 当前的标定序号，从1开始
    
#     def dfs(x, y, label):
#         if (x < 0 or x >= rows or 
#             y < 0 or y >= cols or 
#             grid[x][y] != -1):
#             return
        
#         grid[x][y] = label
#         dfs(x-1, y, label)  # 上
#         dfs(x+1, y, label)  # 下
#         dfs(x, y-1, label)  # 左
#         dfs(x, y+1, label)  # 右
    
#     for i in trange(rows, desc="Labeling regions",ncols=100):
#         for j in range(cols):
#             if grid[i][j] == -1:
#                 dfs(i, j, current_label)
#                 current_label += 1
from collections import deque
def label_connected_regions(grid):
    if grid.size == 0 or grid[0].all() == 0:
        return
    rows, cols = len(grid), len(grid[0])
    current_label = 1  # 当前的标定序号，从1开始
    
    def label_region(x, y, label):
        # 使用栈进行迭代DFS
        stack = deque([(x, y)])
        
        while stack:
            cx, cy = stack.pop()
            # 检查边界以及当前格点是否为-1
            if (cx < 0 or cx >= rows or 
                cy < 0 or cy >= cols or 
                grid[cx][cy] != -1):
                continue
            
            # 将当前格点的-1改为标定值
            grid[cx][cy] = label
            
            # 将上下左右四个方向加入栈
            stack.append((cx-1, cy))  # 上
            stack.append((cx+1, cy))  # 下
            stack.append((cx, cy-1))  # 左
            stack.append((cx, cy+1))  # 右
    
    # 从左上到右下扫描
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == -1:
                label_region(i, j, current_label)
                current_label += 1
def array_to_image(grid, output_file="labeled_regions.png"):
    rows, cols = len(grid), len(grid[0])
    # 创建一个RGB图像，初始为黑色 (0, 0, 0)
    img = Image.new("RGB", (cols, rows), "black")
    pixels = img.load()

    # 预定义100个颜色，用于标定值到颜色的映射
    color_map = {
        -1: (0, 0, 0),  # 黑色
        255: (255, 255, 255),  # 白色
    }
    for i in range(0, 100):
        color_map[i] = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256)
        )

    # 找到最大标定值，确保未定义的标定值也能映射到随机颜色
    max_label = max(max(row) for row in grid)
    if max_label > len(color_map) - 1:
        for label in range(len(color_map), max_label + 1):
            # 为超出预定义颜色的标定值生成随机颜色
            color_map[label] = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256)
            )

    # 将数组值映射到图像像素
    for i in range(rows):
        for j in range(cols):
            label = grid[i][j]
            pixels[j, i] = color_map[label]  # 注意：PIL 中 (x, y) 对应 (列, 行)

    # 保存图像为PNG文件
    img.save(output_file)
    print(f"Image saved as {output_file}")
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import skeletonize
from scipy.ndimage import label
import numpy as np
from scipy.ndimage import binary_erosion, label

from PIL import Image
import numpy as np
from collections import deque
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.morphology import skeletonize
from skimage.measure import find_contours
from skimage.feature import corner_harris, corner_peaks

def preprocess_grid(grid):
    sum_minus = np.sum(grid == -1)
    print(f"初始 -1 个数: {sum_minus}")
    binary_grid = (grid == -1).astype(np.uint8)
    binary_grid[binary_grid == 255] = 0

    skeleton = skeletonize(binary_grid).astype(np.uint8)
    print(f"骨架化后 -1 个数: {np.sum(skeleton)}")

    labeled_skeleton, num_features = label(skeleton)
    protected = np.zeros_like(binary_grid, dtype=np.uint8)
    for i in range(1, num_features + 1):
        region = (labeled_skeleton == i)
        region_pixels = np.sum(region)
        padded_region = np.pad(region, 1, mode='constant')
        neighbors = (padded_region[2:, 1:-1] + padded_region[:-2, 1:-1] + 
                     padded_region[1:-1, 2:] + padded_region[1:-1, :-2] +
                     padded_region[2:, 2:] + padded_region[2:, :-2] + 
                     padded_region[:-2, 2:] + padded_region[:-2, :-2])
        branch_points = np.sum(neighbors[1:-1, 1:-1] > 2)
        if branch_points < 5 and region_pixels > 10:
            protected |= region.astype(np.uint8)

    erode_mask = binary_grid & ~protected
    eroded = binary_erosion(erode_mask, structure=np.ones((3, 3)), iterations=1).astype(np.uint8)
    eroded = np.bitwise_or(eroded, protected)
    eroded = binary_dilation(eroded, structure=np.ones((2, 2)), iterations=1).astype(np.uint8)

    grid_preprocessed = np.zeros_like(grid)
    grid_preprocessed[eroded == 1] = -1
    grid_preprocessed[eroded == 0] = 255
    return grid_preprocessed

def erode_corners(grid_preprocessed):
    """
    对每个连通域的角点区域进行腐蚀
    输入：标记后的网格（值包括 0, 1, 2, 3 等）
    输出：腐蚀后的网格
    """
    grid_result = grid_preprocessed.copy()
    max_label = np.max(grid_preprocessed)
    
    for label_val in range(1, max_label + 1):
        # 提取当前连通域
        binary_region = (grid_preprocessed == label_val).astype(np.uint8)
        
        # 提取边界
        contours = find_contours(binary_region, level=0.5)
        if not contours:
            continue
        
        # 将边界转换为图像
        contour_img = np.zeros_like(binary_region)
        for contour in contours:
            for y, x in contour:
                contour_img[int(y), int(x)] = 1
        
        # Harris 角点检测
        harris_response = corner_harris(contour_img, sigma=1)
        corners = corner_peaks(harris_response, min_distance=5, threshold_rel=0.1)
        
        # 对每个角点区域进行局部腐蚀
        for y, x in corners:
            # 定义腐蚀区域（5x5 窗口）
            y_min, y_max = max(0, y-2), min(grid_preprocessed.shape[0], y+3)
            x_min, x_max = max(0, x-2), min(grid_preprocessed.shape[1], x+3)
            region_to_erode = binary_region[y_min:y_max, x_min:x_max]
            if np.sum(region_to_erode) > 0:  # 确保区域内有像素
                eroded_region = binary_erosion(region_to_erode, structure=np.ones((3, 3)), iterations=2)
                # 更新结果，只影响当前标签的值
                mask = (grid_result[y_min:y_max, x_min:x_max] == label_val)
                grid_result[y_min:y_max, x_min:x_max][mask] = 255  # 腐蚀后设为 0
        
    return grid_result
# 测试代码
if __name__ == "__main__":
    # 创建一个 500x500 的测试数组
    # grid = np.zeros((500, 500), dtype=int)
    # # 添加一些 -1 的连通区域
    # grid[50:100, 50:70] = -1    # 一个矩形区域
    # grid[200:220, 300:350] = -1 # 另一个矩形区域
    # grid[400:450, 400:420] = -1 # 第三个矩形区域
    import pickle
    file = "/root/reaction_data/chemical_tools/tmp_pic_array.pkl"
    with open(file, "rb") as f:
        grid = pickle.load(f)
    print(f"grid shape: {grid.shape}")
    # # 处理连通区域
    array_to_image(grid, "input_500x500_split.png")
    grid_preprocessed = preprocess_grid(grid)
    # label_connected_regions(grid_preprocessed)
    label_connected_regions(grid_preprocessed)
    grid_preprocessed = erode_corners(grid_preprocessed)
    # 将结果映射为图像并保存
    array_to_image(grid_preprocessed, "output_500x500_split.png")