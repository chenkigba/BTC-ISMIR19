# A Bi-Directional Transformer for Musical Chord Recognition

This repository has the source codes for the paper "A Bi-Directional Transformer for Musical Chord Recognition"(ISMIR19).

<img src="png/model.png">

## Installation

### 使用 uv (推荐)

本项目使用 [uv](https://github.com/astral-sh/uv) 进行现代化的 Python 依赖管理。

首先安装 uv：
```bash
# macOS 和 Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

然后安装项目依赖：
```bash
# 同步依赖并创建虚拟环境
uv sync

# 激活虚拟环境（如果需要）
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate  # Windows
```

### 传统安装方式

如果您不使用 uv，也可以使用 pip 安装依赖：
```bash
pip install -e .
```

## Requirements

项目依赖（已定义在 `pyproject.toml` 中）：
- pytorch >= 2.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- pyrubberband >= 0.3.0
- librosa >= 0.10.0
- pyyaml >= 6.0
- mir-eval >= 0.7
- pretty-midi >= 0.2.9
- tensorflow >= 2.13.0
- scipy >= 1.11.0
- sortedcontainers >= 2.4.0
- pillow >= 10.0.0

**注意**: 本项目需要 Python 3.11 或更高版本。

## File descriptions
  * `audio_dataset.py` : loads data and preprocesses label files to chord labels and mp3 files to constant-q transformation. 
  * `btc_model.py` : contains pytorch implementation of BTC.
  * `train.py` : for training. 
  * `crf_model.py` : contatins pytorch implementation of Conditional Random Fields (CRFs) .
  * `baseline_models.py` : contains the codes of baseline models.
  * `train_crf.py` : for training CRFs.  
  * `run_config.yaml` : includes hyper parameters and paths that are needed.
  * `test.py` : for recognizing chord from audio file. 

## Using BTC : Recognizing chords from files in audio directory

### Using BTC from Python API (推荐)

本项目提供了简单易用的 Python API 接口 `ChordRecognizer`，可以更方便地在代码中使用。

#### 基本使用

```python
from btc_ismir19 import ChordRecognizer

# 加载预训练模型（大词汇表模式，170种和弦）
recognizer = ChordRecognizer.from_pretrained('large_voca')

# 识别单个音频文件
chords = recognizer.recognize('audio.mp3', output_format='list')

# 打印结果
for start, end, chord in chords:
    print(f"{start:.3f}s - {end:.3f}s: {chord}")
```

#### 使用大调/小调模式（25种和弦）

```python
# 使用大调/小调模式
recognizer = ChordRecognizer.from_pretrained('majmin')
chords = recognizer.recognize('audio.mp3')
```

#### 保存结果到文件

```python
# 保存为 .lab 文件
recognizer.recognize(
    'audio.mp3',
    output_format='lab',
    save_path='output/result.lab'
)

# 保存为 JSON 文件
recognizer.recognize(
    'audio.mp3',
    output_format='json',
    save_path='output/result.json'
)
```

#### 批量处理

```python
# 批量处理多个音频文件
audio_files = ['audio1.mp3', 'audio2.mp3', 'audio3.wav']
results = recognizer.recognize_batch(
    audio_files,
    output_format='lab',
    save_dir='output/batch_results',
    continue_on_error=True  # 遇到错误时继续处理其他文件
)
```

#### 使用自定义模型

```python
# 使用自定义模型路径
recognizer = ChordRecognizer(
    model_path='./path/to/custom_model.pt',
    large_voca=True,  # 根据模型类型设置
    device='cuda'     # 可选：'cuda' 或 'cpu'
)
```

#### API 参考

- `ChordRecognizer.from_pretrained(model_type)`: 从预训练模型加载
  - `model_type`: `'majmin'` (25种和弦) 或 `'large_voca'` (170种和弦)
- `recognize(audio_path, output_format='list', save_path=None)`: 识别单个音频文件
  - `output_format`: `'list'`, `'lab'`, 或 `'json'`
- `recognize_batch(audio_paths, output_format='list', save_dir=None, continue_on_error=False)`: 批量识别

更多示例请参考 `example_usage.py` 文件。

### 输出文件格式

#### .lab 文件格式

`.lab` 文件是标准的和弦标注文件格式，每行表示一个和弦时间段：

```
开始时间 结束时间 和弦标签
```

示例：
```
0.000 2.500 C
2.500 5.000 C:min
5.000 7.500 F
7.500 10.000 G
```

- **时间单位**：秒（精确到小数点后3位）
- **和弦标签格式**：`根音:和弦类型` 或 `根音`（大调时省略 `:maj`）

#### JSON 格式

JSON 格式返回一个数组，每个元素包含 `start`、`end` 和 `chord` 字段：

```json
[
  {
    "start": 0.0,
    "end": 2.5,
    "chord": "C"
  },
  {
    "start": 2.5,
    "end": 5.0,
    "chord": "C:min"
  }
]
```

#### 列表格式

Python 列表格式，每个元素是一个元组 `(start_time, end_time, chord_label)`：

```python
[
    (0.0, 2.5, "C"),
    (2.5, 5.0, "C:min"),
    (5.0, 7.5, "F")
]
```

### 和弦类型说明

根据选择的模型类型，输出可能包含以下和弦：

#### 大调/小调模式（majmin，25种和弦）

- **12个大调**：`C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `A#`, `B`
- **12个小调**：`C:min`, `C#:min`, `D:min`, `D#:min`, `E:min`, `F:min`, `F#:min`, `G:min`, `G#:min`, `A:min`, `A#:min`, `B:min`
- **无和弦**：`N`（表示该时间段没有和弦）

#### 大词汇表模式（large_voca，170种和弦）

大词汇表模式支持更丰富的和弦类型，共170种：

**根音**（12种）：
- `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `A#`, `B`

**和弦类型**（14种）：
- `min` - 小三和弦（minor）
- `maj` - 大三和弦（major，通常省略，如 `C` 表示 `C:maj`）
- `dim` - 减三和弦（diminished）
- `aug` - 增三和弦（augmented）
- `min6` - 小六和弦（minor sixth）
- `maj6` - 大六和弦（major sixth）
- `min7` - 小七和弦（minor seventh）
- `minmaj7` - 小大七和弦（minor-major seventh）
- `maj7` - 大七和弦（major seventh）
- `7` - 属七和弦（dominant seventh）
- `dim7` - 减七和弦（diminished seventh）
- `hdim7` - 半减七和弦（half-diminished seventh）
- `sus2` - 挂二和弦（suspended second）
- `sus4` - 挂四和弦（suspended fourth）

**特殊标记**：
- `N` - 无和弦（no chord）
- `X` - 未知和弦（unknown chord）

**示例和弦**：
- `C` - C大调
- `C:min` - C小调
- `C:7` - C属七和弦
- `C:maj7` - C大七和弦
- `C:min7` - C小七和弦
- `C:dim` - C减三和弦
- `C:sus4` - C挂四和弦

### Using BTC from command line

```bash 
$ python test.py --audio_dir audio_folder --save_dir save_folder --voca False
```
  * audio_dir : a folder of audio files for chord recognition (default: './test')
  * save_dir : a forder for saving recognition results (default: './test')
  * voca : False means major and minor label type, and True means large vocabulary label type (default: False)
  
The resulting files are lab files of the form shown below and midi files.

  <img src="png/example.png">

## Attention Map
The figures represent the probability values of the attention of self-attention layers 1, 3, 5 and 8 respectively. The
layers that best represent the different characteristics of each layers were chosen. The input audio is the song "Just A Girl"
(0m30s ~ 0m40s) by No Doubt from UsPop2002, which was in evaluation data.
  <img src="png/attention.png">

## Data
We used Isophonics[1], Robbie Williams[2], UsPop2002[3] dataset which consists of chord label files. Due to copyright issue, these datasets do not include audio files. The audio files used in this work were collected from online music service providers.

[1] http://isophonics.net/datasets 

[2] B. Di Giorgi, M. Zanoni, A. Sarti, and S. Tubaro. Automatic
chord recognition based on the probabilistic
modeling of diatonic modal harmony. In Proc. of the
8th International Workshop on Multidimensional Systems,
Erlangen, Germany, 2013.

[3] https://github.com/tmc323/Chord-Annotations

## Reference
  * pytorch implementation of Transformer and Crf: https://github.com/kolloldas/torchnlp 

## Comments
  * Any comments for the codes are always welcome.

