"""
和弦识别命令行工具

使用 ChordRecognizer API 对音频文件进行和弦识别，并生成 .lab 和 .midi 文件。
"""

import argparse
import os

import mir_eval
import pretty_midi as pm

import btc_ismir19.utils.warnings_config  # noqa: F401
from btc_ismir19.chord_recognizer import ChordRecognizer
from btc_ismir19.utils import logger
from btc_ismir19.utils.mir_eval_modules import get_audio_paths

logger.logging_verbosity(1)

# 命令行参数
parser = argparse.ArgumentParser(description="BTC 和弦识别工具")
parser.add_argument(
    "--voca",
    default=True,
    type=lambda x: str(x).lower() == "true",
    help="是否使用大词汇表模式 (default: True)",
)
parser.add_argument("--audio_dir", type=str, default="./test", help="音频文件目录")
parser.add_argument("--save_dir", type=str, default="./test", help="输出文件目录")
args = parser.parse_args()

# 选择模型类型
model_type = "large_voca" if args.voca else "majmin"
logger.info(f"模型类型: {model_type}")

# 加载模型
recognizer = ChordRecognizer.from_pretrained(model_type)
logger.info("模型加载成功")

# 获取音频文件
audio_paths = get_audio_paths(args.audio_dir)

# 和弦识别
for i, audio_path in enumerate(audio_paths):
    logger.info(f"======== {i + 1} of {len(audio_paths)} in progress ========")
    logger.info(f"处理文件: {audio_path}")

    # 识别和弦并保存 .lab 文件
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    save_path = os.path.join(args.save_dir, f"{audio_name}.lab")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    recognizer.recognize(audio_path, output_format="lab", save_path=save_path)
    logger.info(f"lab 文件已保存: {save_path}")

    # 将 lab 文件转换为 midi 文件
    starts, ends, pitchs = [], [], []

    intervals, chords = mir_eval.io.load_labeled_intervals(save_path)
    for p in range(12):
        for j, (interval, chord) in enumerate(zip(intervals, chords)):
            root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
            tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]

            if j == 0:
                start_time = interval[0]
                label = tmp_label
                continue

            if tmp_label != label:
                if label == 1.0:
                    starts.append(start_time)
                    ends.append(interval[0])
                    pitchs.append(p + 48)
                start_time = interval[0]
                label = tmp_label

            if j == len(intervals) - 1 and label == 1.0:
                starts.append(start_time)
                ends.append(interval[1])
                pitchs.append(p + 48)

    # 创建 MIDI 文件
    midi = pm.PrettyMIDI()
    instrument = pm.Instrument(program=0)

    for start, end, pitch in zip(starts, ends, pitchs):
        note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi_path = save_path.replace(".lab", ".midi")
    midi.write(midi_path)
    logger.info(f"midi 文件已保存: {midi_path}")
