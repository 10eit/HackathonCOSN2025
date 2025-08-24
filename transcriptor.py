"""
This function intends to get exact word onset time of each recording files.
whisper_approx_char offer faster way but coarse estimation

"""
from faster_whisper import WhisperModel
from funasr import AutoModel

### For Faster extraction
def whisper_approx_char(audio_path, model_size="tiny", device="cuda"):
    model = WhisperModel(model_size, device=device, compute_type="float16")
    segments, info = model.transcribe(audio_path, word_timestamps=True, language="zh")
    onsets = []
    for seg in segments:
        for word in seg.words:
            text = word.word.strip()
            if not text:
                continue
            n = len(text)
            duration = word.end - word.start
            step = duration / n
            for i in range(n):
                start = word.start + i * step
                onsets.append(start)
    return onsets

### For accurate timestamp
def align_chinese(audio_path: str, device: str = "cpu"):
    model = AutoModel(
        model="paraformer-zh", 
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device=device
    )
    res = model.generate(
        input=audio_path, 
        output_dir=None, 
        param_dict={"align": True}
    )
    onsets = []
    for item in res[0]["timestamp"]:
        onsets.append(item[0] / 1000)
    return onsets
