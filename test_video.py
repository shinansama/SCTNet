import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
from mmseg.apis import init_segmentor, inference_segmentor

# config_file = r"configs/sctnet/pets/sctnet-s_8x4_160k_pets.py"
# checkpoint_file = r"work_dirs/sctnet-s_8x4_160k_pets/iter_136000.pth"

config_file = r"configs/sctnet/pets/sctnet-s_nt_8x4_160k_pets.py"
checkpoint_file = r"work_dirs/sctnet-s_nt_8x4_160k_pets/latest.pth"

# 初始化模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0' if torch.cuda.is_available() else 'cpu')

# 输入视频文件路径，替换为你实际的视频文件路径
input_video_path = 'demo/2cats.mp4'
# 输出视频文件路径，可自行指定输出的文件名及路径
output_video_path = 'output.mp4'

# 打开输入视频文件
cap = cv2.VideoCapture(input_video_path)

# 获取视频的一些基本属性
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度

# 创建VideoWriter对象，用于输出新视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器，这里使用mp4v，可根据需求调整
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 使用tqdm创建进度条，设置总迭代次数为视频总帧数
with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    start_time = time.time()  # 记录视频处理开始时间
    frame_count = 0  # 用于记录已处理的帧数
    while True:
        ret, frame = cap.read()
        if ret:
            with torch.no_grad():
                result = inference_segmentor(model, frame)
                # 将处理后的帧写入新视频文件
                out.write(model.show_result(frame, result))
            frame_count += 1
            # 更新进度条
            pbar.update(1)
        else:
            break
    end_time = time.time()  # 记录视频处理结束时间
    total_elapsed_time = end_time - start_time
    final_fps = frame_count / total_elapsed_time
    print(f"Final average FPS: {final_fps:.2f}")

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
