import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ================= 1. 数据集定义 (MusicGenreDataset) =================
from save_spectrogram import MusicGenreDataset


# ================= 2. 模型定义 =================
def network(num_classes=10):
    """获取单通道 ResNet-18，保留原输出层并追加新全连接层"""
    # 1. 加载未经预训练的 ResNet-18
    model = models.resnet18(weights=None)

    # 2. 将原本接受 3 通道的第一层卷积改为接受单通道
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # 3. 保持原 resnet 的 fc 结构不变，在其后追加网络
    original_fc = model.fc  # 提取出原来的全连接层 (in: 512, out: 1000)

    # 使用 nn.Sequential 将原本的 fc、激活函数和新的 fc 拼装起来替换原来的 model.fc
    model.fc = nn.Sequential(
        original_fc,  # 原本的层，输出 1000 维
        nn.ReLU(),  # 必须加非线性激活函数，防止层级坍缩
        nn.Linear(1000, num_classes)  # 追加的新全连接层，输入 1000，输出 10
    )
    return model


# ================= 3. 核心训练与绘图流程 =================
def main():
    # --- 配置参数 ---
    BASE_DIR = "datasets_process_final"
    PLOT_SAVE_DIR = "Mel_Spectrogram"  # 新增：折线图保存目录
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MODEL_SAVE_PATH = "save_models/digital_signal_process.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 当前使用的计算设备: {device.type.upper()}")

    print("\n📦 正在加载数据集...")
    train_dataset = MusicGenreDataset(os.path.join(BASE_DIR, "train"))
    val_dataset = MusicGenreDataset(os.path.join(BASE_DIR, "val"))
    test_dataset = MusicGenreDataset(os.path.join(BASE_DIR, "test"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = network(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0

    # --- 用于记录每轮数据的列表 ---
    history_train_loss = []
    history_val_acc = []
    history_test_acc = []

    print("\n🚀 开始训练网络...")
    for epoch in range(EPOCHS):
        start_time = time.time()

        # --- 1. 训练阶段 ---
        model.train()
        train_loss, train_total = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total

        # --- 2. 验证集阶段 (Val) ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total

        # --- 3. 测试集阶段 (Test) ---
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100. * test_correct / test_total

        scheduler.step()

        # --- 记录数据到列表 ---
        history_train_loss.append(avg_train_loss)
        history_val_acc.append(val_acc)
        history_test_acc.append(test_acc)

        # --- 打印本轮总结 ---
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        # --- 保存最佳权重 ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   🌟 权重已更新！(基于 Val Acc)")

    # ================= 4. 训练结束，绘制并保存折线图 =================
    print("\n📊 训练结束！正在生成并保存数据折线图...")

    epochs_range = range(1, EPOCHS + 1)

    # 创建一个 1行2列 的宽屏画布
    plt.figure(figsize=(12, 5))

    # 绘制左图：Train Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='red', marker='o', markersize=4)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 绘制右图：Val Acc vs Test Acc
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_val_acc, label='Val Accuracy', color='blue', marker='s', markersize=4)
    plt.plot(epochs_range, history_test_acc, label='Test Accuracy', color='green', marker='^', markersize=4)
    plt.title('Validation & Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 调整布局防止重叠，并保存文件
    plt.tight_layout()
    plot_filepath = os.path.join(PLOT_SAVE_DIR, "training_metrics_curve.png")
    plt.savefig(plot_filepath, dpi=300)  # 保存为高清图片
    plt.close()  # 养成好习惯：关闭画布释放内存

    print(f"🎉 折线图已成功保存至: {plot_filepath}")
    print(f"🏆 最高 Val Acc: {max(history_val_acc):.2f}% | 对应轮次最高 Test Acc: {max(history_test_acc):.2f}%")


if __name__ == "__main__":
    main()
