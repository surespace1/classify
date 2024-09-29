import os

# 定义文件夹路径
folder_path = './dataset/train/nan'
i = 1

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构建原始文件的完整路径
    old_file = os.path.join(folder_path, filename)
    
    # 检查是否是文件
    if os.path.isfile(old_file):
        # 构建新的文件名
        new_filename = f"{i}.jpg"
        # 构建新文件的完整路径
        new_file = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        try:
            os.rename(old_file, new_file)
            i+=1
            print(f"文件 {filename} 已重命名为 {new_filename}")
        except OSError as e:
            print(f"无法重命名文件 {filename}：{e}")