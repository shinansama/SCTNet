import os
import shutil
import random
from sklearn.model_selection import train_test_split
from PIL import Image


# 原始数据集路径
original_dataset_path = r"C:\Users\Administrator\Downloads\oxford3tpet"
# 转换后适用于mmsegmentation的数据集路径
mmseg_dataset_path = r"data/pets"

# 原数据集图像和标注文件夹相对路径
original_images_folder = os.path.join(original_dataset_path, "images")
original_annotations_folder = os.path.join(original_dataset_path, "annotations/trimaps")

# mmsegmentation数据集要求的目录结构及对应路径
mmseg_train_images_folder = os.path.join(mmseg_dataset_path, "train", "img_dir")
mmseg_train_annotations_folder = os.path.join(mmseg_dataset_path, "train", "ann_dir")
mmseg_val_images_folder = os.path.join(mmseg_dataset_path, "val", "img_dir")
mmseg_val_annotations_folder = os.path.join(mmseg_dataset_path, "val", "ann_dir")
mmseg_test_images_folder = os.path.join(mmseg_dataset_path, "test", "img_dir")
mmseg_test_annotations_folder = os.path.join(mmseg_dataset_path, "test", "ann_dir")

# 创建目标目录结构
os.makedirs(mmseg_train_images_folder, exist_ok=True)
os.makedirs(mmseg_train_annotations_folder, exist_ok=True)
os.makedirs(mmseg_val_images_folder, exist_ok=True)
os.makedirs(mmseg_val_annotations_folder, exist_ok=True)
os.makedirs(mmseg_test_images_folder, exist_ok=True)
os.makedirs(mmseg_test_annotations_folder, exist_ok=True)

# 获取所有图像文件名（假设图像和标注文件名一一对应）
image_filenames = [f for f in os.listdir(original_images_folder) if os.path.isfile(os.path.join(original_images_folder, f))]

# 划分训练集、验证集和测试集（这里使用8:1:1的比例作为示例，可根据需求调整）
train_val_filenames, test_filenames = train_test_split(image_filenames, test_size=0.1, random_state=42)
train_filenames, val_filenames = train_test_split(train_val_filenames, test_size=1 / 9, random_state=42)

# 拷贝文件到相应的文件夹
def copy_files(filenames, source_image_folder, source_annotation_folder, target_image_folder, target_annotation_folder):
    for filename in filenames:
        image_path = os.path.join(source_image_folder, filename)
        annotation_path = os.path.join(source_annotation_folder, os.path.splitext(filename)[0] + ".png")  # 假设标注文件后缀为.png
        shutil.copy(image_path, os.path.join(target_image_folder, filename))
        shutil.copy(annotation_path, os.path.join(target_annotation_folder, os.path.splitext(filename)[0] + ".png"))

copy_files(train_filenames, original_images_folder, original_annotations_folder, mmseg_train_images_folder, mmseg_train_annotations_folder)
copy_files(val_filenames, original_images_folder, original_annotations_folder, mmseg_val_images_folder, mmseg_val_annotations_folder)
copy_files(test_filenames, original_images_folder, original_annotations_folder, mmseg_test_images_folder, mmseg_test_annotations_folder)

print("数据集转换完成，已整理为适合mmsegmentation的格式，包含训练集、验证集和测试集。")
