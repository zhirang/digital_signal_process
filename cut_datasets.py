import os
import librosa
import soundfile as sf

# ================= 配置参数 =================
INPUT_DIR = "datasets_process"  # 上一步生成的、已经按 7:1.5:1.5 分好类的长音频目录
OUTPUT_DIR = "datasets_process_final"  # 最终输出的、装满 5s 切片的完美数据集目录
SEGMENT_LEN_SEC = 5  # 目标裁剪长度：5秒
TARGET_SR = 22050  # 统一目标采样率


# ============================================

def process_and_crop():
    # 统计数据，让你心里有数
    total_processed_files = 0
    total_generated_segments = 0

    # 1. 遍历 train, val, test 三个主文件夹
    for split_name in ['train', 'val', 'test']:
        split_in_path = os.path.join(INPUT_DIR, split_name)
        split_out_path = os.path.join(OUTPUT_DIR, split_name)

        # 确保输入文件夹存在（防止手误删除了）
        if not os.path.exists(split_in_path):
            continue

        print(f"\n🚀 开始处理数据集分支: [{split_name.upper()}]")

        # 2. 遍历该分支下的所有流派文件夹 (blues, classical 等)
        for genre in os.listdir(split_in_path):
            genre_in_path = os.path.join(split_in_path, genre)

            if not os.path.isdir(genre_in_path):
                continue

            # 在输出目录中创建对应的分类文件夹
            genre_out_path = os.path.join(split_out_path, genre)
            os.makedirs(genre_out_path, exist_ok=True)

            # 3. 读取并切分该流派下的每一首 .wav 歌曲
            audio_files = [f for f in os.listdir(genre_in_path) if f.endswith('.wav')]

            for filename in audio_files:
                file_path = os.path.join(genre_in_path, filename)

                try:
                    # 读取音频，强制统一采样率
                    y, sr = librosa.load(file_path, sr=TARGET_SR)

                    # 计算 5 秒对应的采样点总数
                    samples_per_segment = sr * SEGMENT_LEN_SEC

                    # 计算可以切出几个 5 秒片段
                    num_segments = len(y) // samples_per_segment

                    for i in range(num_segments):
                        start_sample = i * samples_per_segment
                        end_sample = start_sample + samples_per_segment
                        y_segment = y[start_sample:end_sample]

                        # 构造新文件名 (例: blues.00000.wav -> blues.00000_part0.wav)
                        base_name = filename.replace('.wav', '')
                        new_filename = f"{base_name}_part{i}.wav"
                        new_filepath = os.path.join(genre_out_path, new_filename)

                        # 保存 5s 音频文件
                        sf.write(new_filepath, y_segment, sr)
                        total_generated_segments += 1

                    total_processed_files += 1

                except Exception as e:
                    print(f"  ❌ 裁剪文件 {filename} 时出错: {e}")

        print(f"✅ 分支 [{split_name.upper()}] 裁剪完成！")

    # 打印最终报告
    print("\n" + "=" * 50)
    print("🎉 恭喜！全自动裁剪与数据集构建已大功告成！")
    print(f"原始处理文件: {total_processed_files} 首长音频")
    print(f"最终生成样本: {total_generated_segments} 个 5s 音频")
    print(f"你的完美数据集在: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    process_and_crop()