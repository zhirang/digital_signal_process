# 数字信号处理小组作业

### 一、 数据集

- 原始数据集地址：[0xaryan/music-classifier · Datasets at Hugging Face](https://huggingface.co/datasets/0xaryan/music-classifier)

  原始数据集是音乐类型十分类数据集，每一种类型有100个30s的wav音频文件。

  音乐类型分别是：blues、classical、country、disco、hiphop、jazz、metal、pop、reggae

- **download_dataset.py脚本文件**：从国内镜像下载数据集到本地

- **sort_datasets.py脚本文件**：将下载的数据集按比例（0.7:0.15:0.15）分到train、val、test三个文件夹下，文件目录为

			- datasets_final
		- train（blues、classical、country、disco、hiphop、jazz、metal、pop、reggae）
		- val（blues、classical、country、disco、hiphop、jazz、metal、pop、reggae）
		- test（blues、classical、country、disco、hiphop、jazz、metal、pop、reggae）
	
- **cut_datasets.py脚本文件**：将datasets_final文件下的所有数据集由原来的**1个30s → 6个5s**的wav音频文件，目的是扩大数据量减小单个数据大小。

### 二、CNN方案：STFT 频谱图 + CNN 自适应提取

- **save_spectrogram.py脚本文件**：加载数据集并转换为梅尔频谱图，并将每一种类型的音乐生成一张梅尔频谱图png文件以供观看（**digital_signal_process\Mel_Spectrogram\normal**）。

  - 音频采样频率**target_sr=22050**

  - 音频转换成梅尔频谱图的有关参数

    ```
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,         # 傅里叶变换的窗口大小
        hop_length=512,     # 每次滑动的步长
        n_mels=64           # 提取 64 个梅尔频率维度 (相当于图片的 Height)
    )
    self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()			将能量值转换为分贝(dB)
    ```
    
  
- **train.py脚本文件**：模型训练代码

  - 神经网络：(ResNet-18) + Linear(1000, 10)。①无预训练权重；②ResNet-18输入原本的3通道改为1通道。
  - 参数配置
    - 损失函数CrossEntropyLoss
    
    - 优化器Adam
    
    - 小批量batch_size = 32
    
    - 训练回合epochs = 30
    
    - 学习率lr = 0.001
    
    - 训练设备device = GPU
  
  - 准确率提升做的修改
      - 未修改：**digital_signal_process\Mel_Spectrogram\training_metrics_curve.png** 在train_loss逼近0的情况下，val_acc和test_acc值始终低于80%且上下波动较大，有过拟合的嫌疑。
      - 修改后：**digital_signal_process\Mel_Spectrogram\training_metrics_curve_lr.png** 在train_loss逼近0的情况下，val_acc和test_acc值有突破80%且整体呈向上增长，相较于波动小了许多。
        - 引入“学习率衰减” (Learning Rate Scheduler) - 解决曲线震荡
        - 加入“权重衰减” (Weight Decay / L2正则化) - 惩罚死记硬背，过拟合

### 三、经典 STFT 方案：STFT + 人工特征 + 传统分类器
- **features_extract.py脚本文件**：提取各种类型音乐的特征，并将数据生成CSV表格文件（**digital_signal_process\features_extract**）。

  - 第一步：提取音乐类型特征
  
    ```
    # 2.提取核心人工特征 
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
    ```
  
  - 第二步：数据压缩与聚合 (计算均值和方差，把矩阵变成一维数组)
  
    ```
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
    ```
  
  
  - 第三步：将20个MFCC维度的均值和方差分别展开加入字典
  
- **random_forest.py脚本文件**：训练传统分类器 （随机森林）。

  - 运行结果（**群里发过截图**）

    ```
  正在使用 CPU 训练随机森林分类器...
     验证集 (Val) 准确率: 67.85%
     测试集 (Test) 准确率: 68.04%
    测试集详细分类报告:
                       precision    recall  f1-score   support
         0                0.71      0.61      0.66        90
         1                0.91      0.93      0.92        90
         2                0.49      0.57      0.53        89
         3                0.52      0.48      0.50        90
         4                0.64      0.67      0.65        90
         5                0.80      0.87      0.83        90
         6                0.81      0.84      0.83        90
         7                0.78      0.81      0.80        90
         8                0.59      0.60      0.60        90
         9                0.51      0.42      0.46        89
     accuracy             0.68                           898
     macro avg            0.68      0.68      0.68       898
    weighted avg          0.68      0.68      0.68       898
    
    参数说明：
        precison：准确率
        recall：查全率
        f1-score：precision和recall的调和平均数
        support：支持的样本数
    ```
    

### 四、鲁棒性测试（加入噪声 + 音量增益）
- 1、鲁棒性实验设计介绍

  - 加入噪声：设置信噪比（SNR）取值为 <u>None、20dB 和 10dB</u>。
- 音量增益：设置音量增益（GAIN）取值为<u>1.0 、0.5 和 0.25</u>。
  
  - **noise_music.py脚本文件**：制作了两份音乐结果信噪（SNR）比为<u>None、20dB 和 10dB</u>处理的音频（**digital_signal_process\noise_music**）。
  
- 2、CNN方案
  - **robust_save_spectrogram.py脚本文件**：与**save_spectrogram.py**脚本文件类似，并保存了两张音乐经过<u>加入噪声</u>和<u>音量增益</u>的梅尔二维时频图（"**digital_signal_process\Mel_Spectrogram\robust\blues_snr.png**" 和 "**digital_signal_process\Mel_Spectrogram\robust\blues_gain.png**"）。

  - **robust_test.py脚本文件**：使用test数据集测试上述鲁棒性实验，并画出<u>加入噪声</u>和<u>音量增益</u>的测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rnn_robust_snr.png**" 和 "**digital_signal_process\Mel_Spectrogram\robust\rnn_robust_gain.png**"）。

    ```
    SNR鲁棒性运行结果：
    [ Clean ] | 将循环测试 20 轮...
        整体平均准确率: 75.47%
    [ 20dB ] | 将循环测试 20 轮...
        整体平均准确率: 49.82%
    [ 10dB ] | 将循环测试 20 轮...
        整体平均准确率: 43.87%
    
    GAIN鲁棒性运行结果：
    [ Clean ] | 将循环测试 1 轮...
        整体平均准确率: 75.47%
    [ GAIN=0.5 ] | 将循环测试 1 轮...
        整体平均准确率: 67.23%
    [ GAIN=0.25 ] | 将循环测试 1 轮...
        整体平均准确率: 55.43%
    ```

- 3、STFT方案

  - **robust_rf_test.py脚本文件**：使用test数据集测试上述鲁棒性实验，并画出<u>加入噪声</u>和<u>音量增益</u>的测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rf_robust_snr.png**" 和 "**digital_signal_process\Mel_Spectrogram\robust\rf_robust_gain.png**"）。

       - ```
         SNR鲁棒性运行结果：
          传统方案 (Random Forest) - 20轮平均噪声鲁棒性结果:
            - [ Clean  ] 整体平均准确率: 68.00%
            - [ 20dB   ] 整体平均准确率: 39.18%
            - [ 10dB   ] 整体平均准确率: 25.65%
         
         GAIN鲁棒性运行结果：
          传统方案 (Random Forest) - 音量(Gain)鲁棒性测试结果:
            - [ Gain=1.0  ] 整体平均准确率: 68.00%
            - [ Gain=0.5  ] 整体平均准确率: 66.10%
            - [ Gain=0.25 ] 整体平均准确率: 64.33%
         ```

### 五、可用于展示的资料

- 1、CNN方案
  - 每一种音乐类型的梅尔二维时频图（“**digital_signal_process\Mel_Spectrogram\normal**”）
  - 模型训练折线图（"**digital_signal_process\Mel_Spectrogram\training_metrics_curve.png**"）和性能提升后的模型训练折线图（"**digital_signal_process\Mel_Spectrogram\training_metrics_curve_lr.png**"）
- 2、STFT方案
  - 模型运行结果**random_forest.py脚本文件**的运行结果。
- 3、鲁棒性
  - 加入噪声（None、20dB 和 10 dB）的音频（"**digital_signal_process\noise_music**"）
  - 加入噪声和音量增益的梅尔二维时频图（"**digital_signal_process\Mel_Spectrogram\robust\blues_snr.png**" 和 "**digital_signal_process\Mel_Spectrogram\robust\blues_gain.png**"）
  - CNN鲁棒性：加入噪声测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rnn_robust_snr.png**"）和音量增益测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rnn_robust_gain.png**"）
  - STFT鲁棒性：加入噪声测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rf_robust_snr.png**"）和音量增益测试结果折线图（"**digital_signal_process\Mel_Spectrogram\robust\rf_robust_gain.png**"）