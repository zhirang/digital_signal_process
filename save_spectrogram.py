import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MusicGenreDataset(Dataset):
    def __init__(self, data_dir, target_sr=22050):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.file_paths = []
        self.labels = []

        # 1. 获取分类名称并映射为标签
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # 2. 遍历目录获取文件路径
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for filename in os.listdir(cls_dir):
                if filename.endswith('.wav'):
                    self.file_paths.append(os.path.join(cls_dir, filename))
                    self.labels.append(self.class_to_idx[cls_name])

        # 3. 定义音频到梅尔频谱图的转换流水线
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,         # 傅里叶变换的窗口大小
            hop_length=512,     # 每次滑动的步长
            n_mels=64           # 提取 64 个梅尔频率维度 (相当于图片的 Height)
        )
        # 将能量值转换为分贝(dB)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db, label


# ================= 测试与可视化代码 =================
if __name__ == "__main__":
    # 加载你的训练集
    train_dataset = MusicGenreDataset("datasets_process_final/train")
    all_classes = train_dataset.classes
    SAVE_FOLDER = "Mel_Spectrogram"

    # 准备一个集合，记录我们已经保存了哪些流派
    saved_genres = set()
    print(f"正在为 10 个流派生成代表性频谱图...")

    # 遍历数据集查找每个流派的第一个样本
    for i in range(len(train_dataset)):
        # 如果已经集齐了 10 个流派，就提前退出循环
        if len(saved_genres) == len(all_classes):
            break

        # 取出样本
        spectrogram, label = train_dataset[i]
        genre_name = all_classes[label]

        if genre_name not in saved_genres:
            print("=========================")
            print(f"  -> 正在绘制: {genre_name}")

            # ================= 绘制二维梅尔频谱图 =================
            # 1. 把张量从 GPU/CPU 转成普通的 numpy 数组，并去掉多余的 Channel 维度
            # 形状从 [1, 64, 216] 变成 [64, 216]
            spec_image = spectrogram.squeeze().numpy()
            print(f"矩阵形状 (Channels, Freqs, TimeFrames): {spectrogram.shape}")
            print(f"数字标签: {label} (对应类别: {train_dataset.classes[label]})")

            # 2. 设置画布大小
            plt.figure(figsize=(10, 4))

            # 3. 绘制图像
            # aspect='auto' 让它填满画布，origin='lower' 让低频在最下面，高频在上面
            plt.imshow(spec_image, aspect='auto', origin='lower', cmap='viridis')

            # 4. 添加标题、坐标轴标签和颜色条
            plt.title(f"Mel-Spectrogram - {train_dataset.classes[label]}", fontsize=14)
            plt.ylabel("Mel Frequency Bins")
            plt.xlabel("Time Frames")
            plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")

            save_path = os.path.join(SAVE_FOLDER, f"{genre_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            print(f"图像已保存为到{save_path}")
            plt.close()

            saved_genres.add(genre_name)

    print(f"\n✅ 10 张频谱图已全部保存至: {SAVE_FOLDER}")
