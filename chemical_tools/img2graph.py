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
def preprocess_grid(grid):
    """
    预处理网格，分离粘连的化学键和原子区域。
    输入：500x500的二值图，0为背景，-1为原子或化学键区域
    输出：预处理后的网格
    """
    # 将-1转换为1，便于形态学处理
    binary_grid = (grid == -1).astype(np.uint8)

    # 1. 腐蚀操作：分离粘连的区域（化学键和较大原子）
    eroded = binary_erosion(binary_grid, structure=np.ones((3, 3)), iterations=1)

    # 2. 膨胀操作：恢复区域形状，但避免重新粘连
    preprocessed = binary_dilation(eroded, structure=np.ones((3, 3)), iterations=1)

    # 将处理后的结果转换回原始格式（0和-1）
    grid_preprocessed = np.zeros_like(grid)
    grid_preprocessed[preprocessed == 1] = -1

    return grid_preprocessed
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
    array_to_image(grid, "input_500x500.png")
    label_connected_regions(grid)

    # 将结果映射为图像并保存
    array_to_image(grid, "output_500x500.png")