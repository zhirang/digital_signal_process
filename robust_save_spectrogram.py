import os
import torch
import torchaudio
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MusicDataset_robust(Dataset):
    def __init__(self, data_dir, target_sr=22050, snr_db=None, gain_factor=1.0):
        """
        初始化鲁棒性数据集类
        :param data_dir: 数据集目录
        :param target_sr: 目标采样率
        :param snr_db: 信噪比(dB)。若为 None，则不添加噪声；数值越小，噪声越大。
        :param gain_factor: 音量增益系数。1.0 为不变，0.5 为音量减半，2.0 为音量翻倍。
        """
        self.data_dir = data_dir
        self.target_sr = target_sr

        # 保存鲁棒性测试的控制参数
        self.snr_db = snr_db
        self.gain_factor = gain_factor

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
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    # ================= 核心子函数 1: 调节音量 =================
    def adjust_volume(self, waveform, gain):
        """直接按比例缩放音频波形的振幅"""
        return waveform * gain

    # ================= 核心子函数 2: 添加高斯白噪声 =================
    def add_white_noise(self, waveform, snr):
        """基于目标信噪比(SNR)向波形张量中混入高斯白噪声"""
        # 1. 计算原信号的平均功率 (均方值)
        signal_power = torch.mean(waveform ** 2)

        # 2. 将功率转换为分贝 (加上1e-10防止log(0)报错)
        signal_power_db = 10 * torch.log10(signal_power + 1e-10)

        # 3. 根据公式 SNR = Signal_dB - Noise_dB，反推目标噪声的分贝数
        noise_power_db = signal_power_db - snr

        # 4. 将噪声的分贝数转回真实的功率值 (方差)
        noise_power = 10 ** (noise_power_db / 10)

        # 5. 生成标准正态分布的随机噪声 (mean=0, std=1)，并乘以我们需要的目标标准差(功率的平方根)
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

        # 6. 将噪声叠加回原信号
        return waveform + noise

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. 读取原始波形
        waveform, sr = torchaudio.load(audio_path)

        # 统一转为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ================= 物理破坏阶段 =================
        # 根据初始化参数，选择性地对原始波形进行“破坏”

        # 调节音量 (如果不等于 1.0)
        if self.gain_factor != 1.0:
            waveform = self.adjust_volume(waveform, self.gain_factor)

        # 添加噪声 (如果传入了具体的 SNR 数值)
        if self.snr_db is not None:
            waveform = self.add_white_noise(waveform, self.snr_db)
        # ================================================

        # 2. 频谱转换阶段 (将可能被破坏过的波形转换为梅尔频谱图)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        return mel_spec_db, label


# ================= 测试与可视化代码 =================
if __name__ == "__main__":
    # 保存图片地址
    SAVE_FOLDER = "Mel_Spectrogram/robust"

    # 1.加载数据集
    train_dataset = MusicDataset_robust(
        data_dir="datasets_process_final/train",
        # 加入信噪比。一般取值None，20dB，10dB
        # snr_db=10,
        # 设置音量增益系数。一般取值1.0，0.5，0.25
        gain_factor=0.5
    )

    # 2.取出样本
    spectrogram, label = train_dataset[0]
    genre_name = train_dataset.classes[label]
    print(f"提取样本类别: {genre_name}")

    # 3.开始画图
    spec_image = spectrogram.squeeze().numpy()

    # 创建画布
    plt.figure(figsize=(10, 4))
    # aspect='auto' 让它填满画布，origin='lower' 让低频在最下面，高频在上面
    plt.imshow(spec_image, aspect='auto', origin='lower', cmap='viridis')

    # 4. 添加标题、坐标轴标签和颜色条
    # plt.title(f"Mel-Spectrogram - {train_dataset.classes[label]} - snr/10dB", fontsize=14)
    plt.title(f"Mel-Spectrogram - {train_dataset.classes[label]} - gain/0.5", fontsize=14)
    plt.ylabel("Mel Frequency Bins")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB", label="Amplitude (dB)")

    # save_path = os.path.join(SAVE_FOLDER, f"{genre_name}_snr.png")
    save_path = os.path.join(SAVE_FOLDER, f"{genre_name}_gain.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"图像已保存为到{save_path}")
    plt.close()
