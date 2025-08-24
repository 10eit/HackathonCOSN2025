import glob
import json
from pydub import AudioSegment
import os
from typing import List, Optional
from main_whisper import load_and_preprocess_audio


def split_audio_by_timestamps(
        input_file: str,
        timestamps: List[float],
        output_dir: str = "split_audio",
        prefix: str = "segment", 
        charList = None
) -> None:
    """
    按照指定时间戳切分WAV音频并保存

    参数:
        input_file: 输入WAV音频文件路径
        timestamps: 时间戳列表(秒)，按升序排列，例如[1.5, 3.0, 5.0]
        output_dir: 输出目录，默认为"split_audio"
        prefix: 输出文件前缀，默认为"segment"
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 检查时间戳是否有效
    if not timestamps:
        raise ValueError("时间戳列表不能为空")

    if sorted(timestamps) != timestamps:
        raise ValueError("时间戳必须按升序排列")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok = True)

    # 加载音频文件
    try:
        audio = AudioSegment.from_wav(input_file)
    except Exception as e:
        raise RuntimeError(f"无法加载音频文件: {str(e)}")

    # 获取音频总时长(秒)
    total_duration = len(audio) / 1000.0  # pydub以毫秒为单位

    # 检查时间戳是否在音频时长范围内
    if timestamps[-1] >= total_duration:
        raise ValueError(f"最后一个时间戳({timestamps[-1]}s)超过音频总时长({total_duration:.2f}s)")

    # 添加起始点(0秒)和终点(总时长)到时间戳列表
    split_points = timestamps + [total_duration]

    # 切分音频并保存
    for i in range(len(split_points) - 1):
        start_time = split_points[i] * 1000  # 转换为毫秒
        end_time = split_points[i + 1] * 1000

        # 提取音频片段
        segment = audio[start_time:end_time]

        # 生成输出文件名
        output_filename = f"{os.path.basename(input_file).replace('.wav', '')}_{charList[i]}_{start_time / 1000:.2f}s-{end_time / 1000:.2f}s.wav"
        output_path = os.path.join(output_dir, output_filename)

        # 保存音频片段
        try:
            segment.export(output_path, format = "wav")
            print(f"已保存: {output_path} (时长: {end_time / 1000 - start_time / 1000:.2f}s)")
        except Exception as e:
            print(f"保存失败 {output_path}: {str(e)}")


# 示例用法
if __name__ == "__main__":
    splitreader = json.load(open("audiodata/transcription_results_m1.json"))
    os.makedirs("audiodata/audio_split", exist_ok = True)
    for filename in glob.glob('audiodata/audio_2male_xiaowangzi/xiaowangzi_male_sbject_m1/*.wav'):
        character_level_transcription = splitreader[os.path.basename(filename)]['character_level_transcription']
        # input_wav = load_and_preprocess_audio(filename)
        # 时间戳列表(秒)，按升序排列
        timestamps = [info['start'] for info in character_level_transcription]  # 例如在2.5秒、5.0秒和8.3秒处切分
        charlist = [info['char'] for info in character_level_transcription]

        # 调用函数进行切分
        try:
            split_audio_by_timestamps(
                input_file = filename,
                timestamps = timestamps,
                output_dir = "audiodata/audio_split",
                prefix = "audio",
                charList = charlist
            )
            print("音频切分完成!")
        except Exception as e:
            print(f"发生错误: {str(e)}")
