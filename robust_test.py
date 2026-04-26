import os.path

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 导入你刚刚写好的数据集类
from robust_save_spectrogram import MusicDataset_robust


# ================= 模型定义 (必须和训练时保持一致) =================
def network(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    original_fc = model.fc
    model.fc = nn.Sequential(
        original_fc,
        nn.ReLU(),
        nn.Linear(1000, num_classes)
    )
    return model


# ================= 核心测试脚本 =================
def main():
    # --- 配置参数 ---
    TEST_DIR = "datasets_process_final/test"
    WEIGHTS_PATH = "save_models/digital_signal_process.pth"
    SAVE_FOLDER = "Mel_Spectrogram/robust"
    BATCH_SIZE = 32
    NUM_RUNS = 20  # 测试 5 轮取平均
    SNR_LEVELS = [None, 20, 10]  # 测试的三种信噪比条件

    # 颜色配置，对应三种 SNR
    COLORS = {None: 'green', 20: 'orange', 10: 'red'}
    LABELS = {None: 'Clean (Baseline)', 20: 'SNR = 20dB', 10: 'SNR = 10dB'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device.type.upper()}")

    # 1. 初始化模型并加载权重
    print("\n📦 正在加载预训练模型权重...")
    model = network(num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败，请检查路径: {e}")
        return

    model.eval()  # 锁定为评估模式

    # 用于保存所有 SNR 条件下的各流派准确率
    # 格式: {None: [acc0, acc1...], 20: [acc0, acc1...], 10: [acc0, acc1...]}
    all_snr_results = {}
    genre_classes = []  # 记录流派名称

    # 2. 遍历不同的 SNR 条件进行测试
    for snr in SNR_LEVELS:
        snr_name = "Clean" if snr is None else f"{snr}dB"
        print(f"\n=============================================")
        print(f"🚀 开始测试环境: [ {snr_name} ] | 将循环测试 {NUM_RUNS} 轮...")

        # 实例化对应的数据集
        dataset = MusicDataset_robust(data_dir=TEST_DIR, snr_db=snr, gain_factor=1.0)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        if not genre_classes:
            genre_classes = dataset.classes  # 记录十种流派名称用于画图

        # 记录 5 轮中，每个流派的总正确数和总样本数
        accumulated_correct = np.zeros(10)
        accumulated_total = np.zeros(10)

        # 进行 5 轮测试
        for run in range(NUM_RUNS):
            with torch.no_grad():
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)

                    # 统计每个流派的正确情况
                    for i in range(len(labels)):
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()

                        accumulated_total[true_label] += 1
                        if true_label == pred_label:
                            accumulated_correct[true_label] += 1

        # 计算该 SNR 下，5 轮综合的各个流派平均准确率
        avg_genre_acc = (accumulated_correct / accumulated_total) * 100
        all_snr_results[snr] = avg_genre_acc

        # 打印当前 SNR 的测试结果
        print(f"📊 [ {snr_name} ] 5 轮平均测试结果:")
        for i, genre in enumerate(genre_classes):
            print(f"   - {genre.ljust(10)}: {avg_genre_acc[i]:.2f}%")
        print(f"   🔥 整体平均准确率: {np.mean(avg_genre_acc):.2f}%")

    # ================= 3. 绘制折线图 =================
    print("\n📈 正在生成并保存鲁棒性折线图...")
    plt.figure(figsize=(12, 6))

    x_positions = np.arange(len(genre_classes))

    # 遍历每一种 SNR，画一条折线
    for snr in SNR_LEVELS:
        accuracies = all_snr_results[snr]
        plt.plot(
            x_positions,
            accuracies,
            marker='o',
            markersize=8,
            linewidth=2,
            color=COLORS[snr],
            label=LABELS[snr]
        )

    # 图表细节设置
    plt.title(f"RNN Robustness (Avg over {NUM_RUNS} Runs):Accuracy per Genre under Noise", fontsize=16, fontweight='bold')
    plt.xlabel("Music Genres", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)

    # 替换 X 轴的刻度数字为流派名称，并倾斜方便阅读
    plt.xticks(x_positions, genre_classes, rotation=45, fontsize=11)

    # Y 轴范围设置为 0-100
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(SAVE_FOLDER, "rnn_robust_snr.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 折线图已成功保存为: {save_path}")


'''
def main():
    # --- 配置参数 ---
    TEST_DIR = "datasets_process_final/test"
    WEIGHTS_PATH = "save_models/digital_signal_process.pth"
    SAVE_FOLDER = "Mel_Spectrogram/robust"
    BATCH_SIZE = 32
    GAIN_LEVEL = [1.0, 0.5, 0.25]  # 测试的三种增益条件

    # 颜色配置，对应三种 SNR
    COLORS = {1.0: 'green', 0.5: 'orange', 0.25: 'red'}
    LABELS = {1.0: 'Clean (Baseline)', 0.5: 'GAIN = 0.5', 0.25: 'GAIN = 0.25'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device.type.upper()}")

    # 1. 初始化模型并加载权重
    print("\n📦 正在加载预训练模型权重...")
    model = network(num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败，请检查路径: {e}")
        return

    model.eval()  # 锁定为评估模式

    # 用于保存所有 GAIN 条件下的各流派准确率
    # 格式: {None: [acc0, acc1...], 20: [acc0, acc1...], 10: [acc0, acc1...]}
    all_gain_results = {}
    genre_classes = []  # 记录流派名称

    # 2. 遍历不同的 SNR 条件进行测试
    for gain in GAIN_LEVEL:
        gain_name = "Clean" if gain == 1.0 else f"GAIN={gain}"
        print(f"\n=============================================")
        print(f"🚀 开始测试环境: [ {gain_name} ] | 将循环测试 1 轮...")

        # 实例化对应的数据集
        dataset = MusicDataset_robust(data_dir=TEST_DIR, snr_db=None, gain_factor=gain)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        if not genre_classes:
            genre_classes = dataset.classes  # 记录十种流派名称用于画图

        accumulated_correct = np.zeros(10)
        accumulated_total = np.zeros(10)

        # 进行 5 轮测试

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)

                # 统计每个流派的正确情况
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()

                    accumulated_total[true_label] += 1
                    if true_label == pred_label:
                        accumulated_correct[true_label] += 1

        avg_genre_acc = (accumulated_correct / accumulated_total) * 100
        all_gain_results[gain] = avg_genre_acc

        # 打印当前 GAIN 的测试结果
        print(f"📊 [ {gain_name} ] 测试结果:")
        for i, genre in enumerate(genre_classes):
            print(f"   - {genre.ljust(10)}: {avg_genre_acc[i]:.2f}%")
        print(f"   🔥 整体平均准确率: {np.mean(avg_genre_acc):.2f}%")

    # ================= 3. 绘制折线图 =================
    print("\n📈 正在生成并保存鲁棒性折线图...")
    plt.figure(figsize=(12, 6))

    x_positions = np.arange(len(genre_classes))

    # 遍历每一种 SNR，画一条折线
    for gain in GAIN_LEVEL:
        accuracies = all_gain_results[gain]
        plt.plot(
            x_positions,
            accuracies,
            marker='o',
            markersize=8,
            linewidth=2,
            color=COLORS[gain],
            label=LABELS[gain]
        )

    # 图表细节设置
    plt.title("RNN Robustness: Accuracy per Genre under GAIN", fontsize=16, fontweight='bold')
    plt.xlabel("Music Genres", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)

    # 替换 X 轴的刻度数字为流派名称，并倾斜方便阅读
    plt.xticks(x_positions, genre_classes, rotation=45, fontsize=11)

    # Y 轴范围设置为 0-100
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(SAVE_FOLDER, "rnn_robust_gain.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 折线图已成功保存为: {save_path}")
'''


if __name__ == "__main__":
    main()
