import os
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings

# 告诉系统忽略 sklearn 抛出的特征名不匹配警告
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ================= 1. Numpy版数据增强 =================
def add_white_noise_numpy(waveform, snr_db):
    """基于目标信噪比(SNR)向波形数组中混入高斯白噪声"""
    if snr_db is None:
        return waveform
    # 1. 计算原信号功率
    signal_power = np.mean(waveform ** 2)
    signal_power_db = 10 * np.log10(signal_power + 1e-10)
    # 2. 反推噪声功率
    noise_power_db = signal_power_db - snr_db
    noise_power = 10 ** (noise_power_db / 10)
    # 3. 生成并叠加噪声
    noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
    return waveform + noise


def adjust_volume_numpy(waveform, gain):
    return waveform * gain


# ================= 2. 核心特征提取 (纯内存计算) =================
def extract_features_from_array(y, sr):
    """从内存中的波形数组提取特征 (无需重复读写硬盘)"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 聚合特征为一维向量 (计算均值和方差)
    feature_vector = [
        np.mean(centroid), np.var(centroid),
        np.mean(bandwidth), np.var(bandwidth),
        np.mean(zcr), np.var(zcr),
        np.mean(chroma), np.var(chroma)
    ]
    for i in range(20):
        feature_vector.extend([np.mean(mfccs[i]), np.var(mfccs[i])])

    return np.array(feature_vector)


# ================= 3. 主程序 =================
def main():
    # --- 配置参数 ---
    TEST_DIR = "datasets_process_final/test"
    MODEL_PATH = "save_models/rf_model.pkl"
    SCALER_PATH = "save_models/rf_scaler.pkl"
    SAVE_FOLDER = "Mel_Spectrogram/robust"

    NUM_RUNS = 20  # 每种环境测试 20 轮

    SNR_LEVELS = [None, 20, 10]
    COLORS = {None: 'green', 20: 'orange', 10: 'red'}
    LABELS = {None: 'Clean', 20: 'SNR = 20dB', 10: 'SNR = 10dB'}

    print("📦 正在加载随机森林模型与 Scaler...")
    try:
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    classes = sorted(os.listdir(TEST_DIR))

    # 初始化统计字典 (每种 SNR 独立统计 20 轮的累加结果)
    correct_counts = {snr: {c: 0 for c in classes} for snr in SNR_LEVELS}
    total_counts = {snr: {c: 0 for c in classes} for snr in SNR_LEVELS}

    print(f"\n🚀 开始执行 SNR 鲁棒性测试 (每种噪声测试 {NUM_RUNS} 轮)...")
    print("⏳ 警告：传统特征提取全部依赖 CPU，此过程可能需要 30~60 分钟，请耐心等待或喝杯咖啡！\n")

    for label_idx, genre in enumerate(classes):
        genre_path = os.path.join(TEST_DIR, genre)
        if not os.path.isdir(genre_path): continue

        audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        for filename in audio_files:
            file_path = os.path.join(genre_path, filename)

            # 仅仅读一次硬盘
            y_original, sr = librosa.load(file_path, sr=22050)

            for snr in SNR_LEVELS:
                if snr is None:
                    # 【极致提速逻辑】
                    # Clean 数据没有随机性，测 1 次等于测 20 次，直接赋予 20 轮的权重
                    raw_features = extract_features_from_array(y_original, sr)
                    scaled_features = scaler.transform(raw_features.reshape(1, -1))
                    pred_label_idx = clf.predict(scaled_features)[0]

                    total_counts[snr][genre] += NUM_RUNS
                    if pred_label_idx == label_idx:
                        correct_counts[snr][genre] += NUM_RUNS

                else:
                    # 对于加噪声的环境，老老实实执行 20 轮真实的随机干扰测试
                    for run in range(NUM_RUNS):
                        y_test = np.copy(y_original)
                        y_test = add_white_noise_numpy(y_test, snr)

                        raw_features = extract_features_from_array(y_test, sr)
                        scaled_features = scaler.transform(raw_features.reshape(1, -1))
                        pred_label_idx = clf.predict(scaled_features)[0]

                        total_counts[snr][genre] += 1
                        if pred_label_idx == label_idx:
                            correct_counts[snr][genre] += 1

        print(f"   ✅ 流派 [{genre}] 的 20 轮测试已处理完毕.")

    # ================= 4. 计算并打印结果 =================
    all_snr_results = {}
    print("\n📊 传统方案 (Random Forest) - 20轮平均噪声鲁棒性结果:")

    for snr in SNR_LEVELS:
        snr_name = "Clean" if snr is None else f"{snr}dB"
        accs = [(correct_counts[snr][c] / total_counts[snr][c]) * 100 for c in classes]
        all_snr_results[snr] = accs

        overall_acc = np.mean(accs)
        print(f"   - [ {snr_name.ljust(6)} ] 整体平均准确率: {overall_acc:.2f}%")

    # ================= 5. 绘制折线图 =================
    print("\n📈 正在生成 20 轮平均 SNR 鲁棒性折线图...")
    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(classes))

    for snr in SNR_LEVELS:
        plt.plot(
            x_positions,
            all_snr_results[snr],
            marker='s',
            markersize=8,
            linewidth=2,
            color=COLORS[snr],
            label=LABELS[snr],
            linestyle='--'
        )

    plt.title(f"Random Forest Robustness (Avg over {NUM_RUNS} Runs): Accuracy per Genre under Noise", fontsize=15,
              fontweight='bold')
    plt.xlabel("Music Genres", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(x_positions, classes, rotation=45, fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(SAVE_FOLDER, "rf_robust_snr.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎉 折线图已成功保存为: {save_path}")


'''


def main():
    # --- 配置参数 ---
    TEST_DIR = "datasets_process_final/test"
    MODEL_PATH = "save_models/rf_model.pkl"
    SCALER_PATH = "save_models/rf_scaler.pkl"
    SAVE_FOLDER = "Mel_Spectrogram/robust"

    # 仅测试三种 Gain 条件 (音量调整是确定性操作，只需跑 1 轮)
    GAIN_LEVELS = [1.0, 0.5, 0.25]
    COLORS = {1.0: 'green', 0.5: 'orange', 0.25: 'red'}
    LABELS = {1.0: 'Gain=1.0', 0.5: 'Gain = 0.5', 0.25: 'Gain = 0.25'}

    print("📦 正在加载随机森林模型与 Scaler...")
    try:
        clf = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ 模型与标准化器加载成功！")
    except Exception as e:
        print(f"❌ 加载失败，请检查目录下是否有 .pkl 文件: {e}")
        return

    classes = sorted(os.listdir(TEST_DIR))

    # 初始化字典，记录每种 Gain 下每个流派的 正确数 和 总数
    # 格式: { 1.0: {'blues': 0, ...}, 0.5: {'blues': 0, ...}, 0.25: {'blues': 0, ...} }
    correct_counts = {gain: {c: 0 for c in classes} for gain in GAIN_LEVELS}
    total_counts = {c: 0 for c in classes}

    print("\n🚀 开始遍历测试集进行高并发特征提取与预测 (只需测试 1 轮，速度较快)...")

    for label_idx, genre in enumerate(classes):
        genre_path = os.path.join(TEST_DIR, genre)
        if not os.path.isdir(genre_path): continue

        audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        for filename in audio_files:
            file_path = os.path.join(genre_path, filename)

            # 💡 核心提速：每首歌只读一次硬盘！
            y_original, sr = librosa.load(file_path, sr=22050)
            total_counts[genre] += 1

            # 针对这首歌，在内存中瞬间跑完三种 Gain 测试
            for gain in GAIN_LEVELS:
                # 1. 复制干净的波形
                y_test = np.copy(y_original)

                # 2. 调整音量
                if gain != 1.0:
                    y_test = adjust_volume_numpy(y_test, gain)

                # 3. 提取特征 -> 标准化 -> 预测
                raw_features = extract_features_from_array(y_test, sr)
                scaled_features = scaler.transform(raw_features.reshape(1, -1))
                pred_label_idx = clf.predict(scaled_features)[0]

                # 4. 统计结果
                if pred_label_idx == label_idx:
                    correct_counts[gain][genre] += 1

        print(f"   ✅ 流派 [{genre}] 处理完毕.")

    # ================= 4. 计算并打印结果 =================
    all_gain_results = {}
    print("\n📊 传统方案 (Random Forest) - 音量(Gain)鲁棒性测试结果:")

    for gain in GAIN_LEVELS:
        gain_name = f"Gain={gain}"
        # 计算该 Gain 下，所有流派的准确率百分比
        accs = [(correct_counts[gain][c] / total_counts[c]) * 100 for c in classes]
        all_gain_results[gain] = accs

        overall_acc = np.mean(accs)
        print(f"   - [ {gain_name.ljust(9)} ] 整体平均准确率: {overall_acc:.2f}%")

    # ================= 5. 绘制折线图 =================
    print("\n📈 正在生成 Gain 鲁棒性折线图...")
    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(classes))

    for gain in GAIN_LEVELS:
        plt.plot(
            x_positions,
            all_gain_results[gain],
            marker='s',  # 使用正方形标记
            markersize=8,
            linewidth=2,
            color=COLORS[gain],
            label=LABELS[gain],
            linestyle='--'  # 使用虚线
        )

    plt.title("Random Forest Robustness: Accuracy per Genre under Gain", fontsize=16, fontweight='bold')
    plt.xlabel("Music Genres", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(x_positions, classes, rotation=45, fontsize=11)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(SAVE_FOLDER, "rf_robust_gain.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"🎉 折线图已成功保存为: {save_path}")

'''


if __name__ == "__main__":
    main()
