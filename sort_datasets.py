import os
import shutil
import random

# ================= 配置参数 =================
INPUT_DIR = "datasets_download"  # 你的原始 30s 音频文件夹
OUTPUT_DIR = "datasets_process"  # 划分好类别的原始数据保存路径
RATIO_TRAIN = 0.70
RATIO_VAL = 0.15
RATIO_TEST = 0.15
RANDOM_SEED = 42  # 固定随机种子，保证每次运行划分结果一致


# ============================================

def split_raw_dataset():
    # 设置随机种子
    random.seed(RANDOM_SEED)

    # 获取所有的流派名称 (blues, classical, rock 等)
    genres = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]

    # 1. 提前创建好所有的目标文件夹结构
    for split in ['train', 'val', 'test']:
        for genre in genres:
            os.makedirs(os.path.join(OUTPUT_DIR, split, genre), exist_ok=True)

    total_copied = 0

    # 2. 逐个流派进行划分和拷贝
    for genre in genres:
        genre_path = os.path.join(INPUT_DIR, genre)

        # 获取该流派下的所有 .wav 文件
        audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        # 打乱文件顺序
        random.shuffle(audio_files)

        # 3. 按比例计算分割点 (GTZAN每类刚好100首，算出来就是70, 15, 15)
        total_files = len(audio_files)
        num_train = int(total_files * RATIO_TRAIN)
        num_val = int(total_files * RATIO_VAL)
        # 剩下的全给 test
        num_test = total_files - num_train - num_val

        # 划分文件列表
        train_files = audio_files[:num_train]
        val_files = audio_files[num_train: num_train + num_val]
        test_files = audio_files[num_train + num_val:]

        print(f"[{genre}] 原始文件划分: 训练 {len(train_files)} | 验证 {len(val_files)} | 测试 {len(test_files)}")

        # 4. 根据划分好的列表，将原文件拷贝到目标目录
        splits_mapping = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split_name, files_list in splits_mapping.items():
            for filename in files_list:
                src_file = os.path.join(genre_path, filename)
                dst_file = os.path.join(OUTPUT_DIR, split_name, genre, filename)

                # 拷贝文件
                shutil.copy2(src_file, dst_file)
                total_copied += 1

    print("\n" + "=" * 50)
    print("✅ 原始数据划分大功告成！")
    print(f"总计拷贝了 {total_copied} 个 30s 原始音频文件。")
    print(f"请前往 {OUTPUT_DIR} 文件夹查看结构完整的原始数据集。")
    print("=" * 50)


if __name__ == "__main__":
    split_raw_dataset()