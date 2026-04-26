import os
import librosa
import numpy as np
import pandas as pd

# ================= 配置参数 =================
BASE_DIR = "datasets_process_final"  # 你的 5s 数据集目录
TARGET_SR = 22050


# ============================================

def extract_audio_features(input_dir, output_csv):
    features_list = []
    classes = sorted(os.listdir(input_dir))

    for label_idx, genre in enumerate(classes):
        genre_path = os.path.join(input_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"正在提取 [{genre}] 的特征...")

        for filename in os.listdir(genre_path):
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(genre_path, filename)

            try:
                # 1. 加载音频
                y, sr = librosa.load(file_path, sr=TARGET_SR)

                # 2. 提取核心人工特征
                # MFCC (梅尔频率倒谱系数): 最核心的“音色”特征，提取 20 维
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                # 频谱质心 (Spectral Centroid): 描述声音的“明亮度”（重金属会比古典乐高）
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                # 频谱带宽 (Spectral Bandwidth): 频率分布的范围
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                # 过零率 (Zero Crossing Rate): 描述声音的“打击感”或噪声程度
                zcr = librosa.feature.zero_crossing_rate(y)
                # 色度图 (Chroma): 捕捉和弦与音高信息
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)

                # 3. 数据压缩与聚合 (计算均值和方差，把矩阵变成一维数组)
                row_data = {
                    'filename': filename,
                    'label': label_idx,
                    'centroid_mean': np.mean(centroid),
                    'centroid_var': np.var(centroid),
                    'bandwidth_mean': np.mean(bandwidth),
                    'bandwidth_var': np.var(bandwidth),
                    'zcr_mean': np.mean(zcr),
                    'zcr_var': np.var(zcr),
                    'chroma_mean': np.mean(chroma),
                    'chroma_var': np.var(chroma)
                }

                # 将 20 个 MFCC 维度的均值和方差分别展开加入字典
                for i in range(1, 21):
                    row_data[f'mfcc{i}_mean'] = np.mean(mfccs[i - 1])
                    row_data[f'mfcc{i}_var'] = np.var(mfccs[i - 1])

                features_list.append(row_data)

            except Exception as e:
                print(f"提取 {filename} 时出错: {e}")

    # 保存为 CSV 表格
    df = pd.DataFrame(features_list)
    df.to_csv(output_csv, index=False)
    print(f"✅ 提取完成！已保存为: {output_csv}\n")


if __name__ == "__main__":
    # 分别为 train, val, test 提取特征表
    extract_audio_features(os.path.join(BASE_DIR, "train"), "features_extract/features_train.csv")
    extract_audio_features(os.path.join(BASE_DIR, "val"), "features_extract/features_val.csv")
    extract_audio_features(os.path.join(BASE_DIR, "test"), "features_extract/features_test.csv")
