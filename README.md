### 2025 COSN Hackathon Project: ChineseEEG2 Investigation

参考文献：[ChineseEEG2](https://arxiv.org/abs/2508.04240)

## 👏our pipeline
```

HackathonCOSN2025/
├── data  # 变量配置
├── whisper/
├── README.md
├── LICENSE
├── audio_split.ipynb: split the audio according to the char or word
├── build_dataset.py: module funtions for building dataset
├── compare_char_and_words.py: comparison between spliting the autio based on char-level and words-level
├── encoding_model.py
├── main_whisper.py: extract audio feature using whisper
├── split_audio.py: spliting the autio according the result of  **audio_split.ipynb**
├── transciptor.py: transcipt the caption of audio
```


我们主要是在 ChineseEEG2 数据集（N=8，但是为了方便，我们只使用前四个被试数据）上复现 Uri Hasson 使用 Whisper Model 进行 ECoG 数据的 encoding model [研究](https://www.nature.com/articles/s41562-025-02105-9)，我们计划的实现路径（Main Track）是：

1. 对 ChineseEEG2 刺激材料利用 Whisper 进行转写，得到每个 run 开始第一个字的时间，计算平均每个字的时间→约 200 ms，对应 50 个数据点。
  
2. 提取 Whisper Model 在给定字 onset 之后 200 ms 的音频，提取 encoder 第 0 层、最后一层分别为 acoustic embedding 和 speech embedding，提取 decoder 最后一层为 language embedding，统一 PCA 降维至 50 维。
  
3. 对 EEG 单通道数据与各层级 embedding 做 encoding model（Ridge Regression），使用 r-value 做解码准确性指标，并分别做跨被试（cross-subject）和被试间（within-subject）解码。
  
  1. 跨被试解码：将所有被试数据放在一起，区分训练集和测试集进行训练
    
  2. 被试间解码：对每个被试数据区分训练集和测试集进行训练
    
  
  比较两种解码稳定性是否存在。
  

一些附加的内容（Bonus Track）：

1. 比较同等参数的 Whisper Model 在经过汉语语料微调前后是否可以提高解码准确性
  
2. 比较单字解码和词语解码的准确性，引入完整语义信息是否可以增加 language embedding 对神经活动的预测准确性
  
3. 将三层 embedding 同时用于预测神经信号，做方差分解计算不同传感器上不同语言特征的贡献程度。
