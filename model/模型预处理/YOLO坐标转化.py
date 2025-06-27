import os
import shutil
from pathlib import Path

def generate_yolo_txt(image_path, class_id, output_dir, class_name):
    """
    为一张图片生成 YOLO 格式的 txt 文件，并复制图片和 txt 到目标文件夹的 images 和 labels 子文件夹
    image_path: 源图片路径
    class_id: 类别 ID（从文件夹名称最后两个数字提取）
    output_dir: 目标文件夹路径
    class_name: 类别的文件夹名称（用于文件名加前缀）
    """
    try:
        # 构造 txt 文件内容
        x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0
        txt_content = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        base_name = f"{class_name}_{image_path.name}"
        target_img_path = output_dir / "images" / base_name
        target_txt_path = output_dir / "labels" / Path(base_name).with_suffix(".txt")
        
        print(f"目标图片路径：{target_img_path}")
        print(f"目标 txt 路径：{target_txt_path}")
        
        # 创建目标目录
        target_img_path.parent.mkdir(parents=True, exist_ok=True)
        target_txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入 txt 文件
        with open(target_txt_path, "w") as f:
            f.write(txt_content)
        
        # 复制图片
        shutil.copy2(image_path, target_img_path)
        
        print(f"成功生成并复制：{target_img_path} 和 {target_txt_path}")
    except Exception as e:
        print(f"处理 {image_path} 时出错：{e}")

def process_folder(root_dir, output_dir):
    """
    遍历大文件夹，生成 YOLO txt 文件并复制到目标文件夹的 images 和 labels 子文件夹
    root_dir: 源大文件夹路径
    output_dir: 目标文件夹路径
    """
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # 检查源文件夹是否存在
    if not root_path.exists():
        print(f"错误：源文件夹 {root_path} 不存在！")
        return
    
    print(f"源文件夹绝对路径：{root_path.resolve()}")
    
    # 创建目标文件夹及其子文件夹
    try:
        # 注释清空逻辑以保留现有文件，方便调试
        # if output_path.exists():
        #     shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "images").mkdir(parents=True, exist_ok=True)
        (output_path / "labels").mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"错误：无法创建目标文件夹 {output_path} 或其子文件夹，请检查权限！")
        return
    
    # 遍历大文件夹中的小文件夹
    for folder in root_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        # 提取文件夹名称最后两个数字作为 class_id
        try:
            class_id = int(folder_name[-2:])
        except ValueError:
            print(f"错误：无法从文件夹 {folder_name} 提取最后两个数字作为 class_id，跳过！")
            continue
        
        print(f"开始处理文件夹：{folder_name} (class_id: {class_id})")
        images = list(folder.glob("*.[pP][nN][gG]"))
        print(f"文件夹 {folder_name} 中的图片：{images}")
        
        for image_path in images:
            print(f"处理图片：{image_path}")
            generate_yolo_txt(image_path, class_id, output_path, folder_name)
    
    # 打印处理的文件夹和对应的 class_id
    print("\n处理的文件夹和 class_id：")
    for folder in root_path.iterdir():
        if folder.is_dir():
            try:
                class_id = int(folder.name[-2:])
                print(f"{folder.name}: {class_id}")
            except ValueError:
                print(f"{folder.name}: 无法提取 class_id")

if __name__ == "__main__":
    dataset_dir = "./English/Img/BadImag/Bmp"  #目标文件夹
    output_dir = "output_dataset"  # 目标生成文件夹
    process_folder(dataset_dir, output_dir)