import os
import librosa
import numpy as np
import soundfile as sf  # 专门用来保存音频的库
from robust_rf_test import add_white_noise_numpy


# ================= 主程序 =================
def main():
    # 1. 挑选一首你想“破坏”的测试歌曲
    # 这里你需要替换成你本地真实存在的一首歌曲的路径
    input_audio_path = "datasets_process_final/train/blues/blues.00000_part0.wav"

    # 获取文件名（比如 "blues.00000"）
    base_name = os.path.splitext(os.path.basename(input_audio_path))[0]

    # 2. 创建保存音频的文件夹
    SAVE_FOLDER = "noise_music"
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print(f"🎧 正在读取原始音频: {input_audio_path}")
    # 读取原始音频
    y_original, sr = librosa.load(input_audio_path, sr=22050)

    # 3. 先把原始干净的音频也存一份，方便等下对比着听
    clean_path = os.path.join(SAVE_FOLDER, f"{base_name}_Clean.wav")
    sf.write(clean_path, y_original, sr)
    print(f"✅ 已保存纯净版: {clean_path}")

    # 4. 生成并保存 SNR = 20dB 的版本
    print("🌧️ 正在生成 20dB 噪声版本...")
    y_20db = add_white_noise_numpy(y_original, 20)
    path_20db = os.path.join(SAVE_FOLDER, f"{base_name}_SNR_20dB.wav")
    sf.write(path_20db, y_20db, sr)
    print(f"✅ 已保存 20dB 版: {path_20db}")

    # 5. 生成并保存 SNR = 10dB 的版本
    print("⛈️ 正在生成 10dB 噪声版本...")
    y_10db = add_white_noise_numpy(y_original, 10)
    path_10db = os.path.join(SAVE_FOLDER, f"{base_name}_SNR_10dB.wav")
    sf.write(path_10db, y_10db, sr)
    print(f"✅ 已保存 10dB 版: {path_10db}")

    print(f"\n🎉 大功告成！快去 [{SAVE_FOLDER}] 文件夹里戴上耳机听听看吧！")


if __name__ == "__main__":
    main()