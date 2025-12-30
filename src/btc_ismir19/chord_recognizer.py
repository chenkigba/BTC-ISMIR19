"""
和弦识别器接口封装

提供简单易用的 API 来使用 BTC 模型进行和弦识别。
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from btc_ismir19.btc_model import BTC_model
from btc_ismir19.utils.hparams import HParams
from btc_ismir19.utils.mir_eval_modules import (
    audio_file_to_features,
    idx2chord,
    idx2voca_chord,
)


class ChordRecognizer:
    """
    和弦识别器类

    封装了 BTC 模型的加载、特征提取、推理等逻辑，提供简单易用的 API。

    示例:
        >>> recognizer = ChordRecognizer.from_pretrained('large_voca')
        >>> chords = recognizer.recognize('audio.mp3')
        >>> for start, end, chord in chords:
        ...     print(f"{start:.3f} {end:.3f} {chord}")
    """

    # 预训练模型路径映射
    PRETRAINED_MODELS = {
        "majmin": {
            "model_path": "./test/btc_model.pt",
            "num_chords": 25,
            "large_voca": False,
        },
        "large_voca": {
            "model_path": "./test/btc_model_large_voca.pt",
            "num_chords": 170,
            "large_voca": True,
        },
    }

    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        large_voca: bool = False,
    ):
        """
        初始化和弦识别器

        Args:
            model_path: 模型文件路径（.pt 文件，需包含 'model', 'mean', 'std' 键）
            config_path: 配置文件路径（默认使用项目根目录的 run_config.yaml）
            device: 设备 ('cuda', 'cpu' 或 None，None 时自动检测)
            large_voca: 是否使用大词汇表模式（如果模型路径不在预训练列表中，需要指定）
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载配置
        if config_path is None:
            # 从项目根目录查找 run_config.yaml
            package_dir = Path(__file__).parent.parent.parent
            config_path = package_dir / "run_config.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {config_path}. 请确保 run_config.yaml 在项目根目录。"
            )

        self.config = HParams.load(str(config_path))
        self.large_voca = large_voca

        # 根据 large_voca 设置配置
        if large_voca:
            self.config.feature["large_voca"] = True
            self.config.model["num_chords"] = 170
            self.idx_to_chord = idx2voca_chord()
        else:
            self.config.feature["large_voca"] = False
            self.config.model["num_chords"] = 25
            self.idx_to_chord = idx2chord

        # 加载模型
        self.model = BTC_model(config=self.config.model).to(self.device)

        # 加载模型权重和归一化参数
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # 验证 checkpoint 格式
        required_keys = ["model", "mean", "std"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"模型文件格式不正确，缺少必需的键: {missing_keys}. "
                f"文件路径: {model_path}"
            )

        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_type: str = "large_voca",
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> "ChordRecognizer":
        """
        从预训练模型加载识别器（工厂方法）

        Args:
            model_type: 模型类型，'majmin' 或 'large_voca'
            config_path: 配置文件路径（可选）
            device: 设备（可选）

        Returns:
            ChordRecognizer 实例

        Raises:
            ValueError: 如果 model_type 不在预训练模型列表中
        """
        if model_type not in cls.PRETRAINED_MODELS:
            raise ValueError(
                f"未知的模型类型: {model_type}. "
                f"可用类型: {list(cls.PRETRAINED_MODELS.keys())}"
            )

        model_info = cls.PRETRAINED_MODELS[model_type]
        return cls(
            model_path=model_info["model_path"],
            config_path=config_path,
            device=device,
            large_voca=model_info["large_voca"],
        )

    def recognize(
        self,
        audio_path: Union[str, Path],
        output_format: str = "list",
        save_path: Optional[Union[str, Path]] = None,
    ) -> Union[List[Tuple[float, float, str]], str]:
        """
        识别音频文件的和弦

        Args:
            audio_path: 音频文件路径（支持 .wav 和 .mp3）
            output_format: 输出格式，'list'（返回列表）、'lab'（返回 .lab 格式字符串）
                          或 'json'（返回 JSON 字符串）
            save_path: 如果指定，将结果保存到文件（仅当 output_format 为 'lab' 时有效）

        Returns:
            根据 output_format 返回不同格式：
            - 'list': [(start_time, end_time, chord_label), ...]
            - 'lab': .lab 格式的字符串
            - 'json': JSON 格式的字符串

        Raises:
            FileNotFoundError: 如果音频文件不存在
            ValueError: 如果 output_format 不支持
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 验证音频文件格式
        if audio_path.suffix.lower() not in [".wav", ".mp3"]:
            raise ValueError(
                f"不支持的音频格式: {audio_path.suffix}. 支持格式: .wav, .mp3"
            )

        try:
            # 提取特征
            feature, feature_per_second, _ = audio_file_to_features(
                str(audio_path), self.config
            )
        except Exception as e:
            raise RuntimeError(
                f"无法从音频文件提取特征: {audio_path}. 错误: {str(e)}"
            ) from e

        # 检查特征是否为空
        if feature.size == 0 or feature.shape[0] == 0:
            raise ValueError(f"音频文件太短或无法提取有效特征: {audio_path}")

        # 预处理特征
        feature = feature.T
        feature = (feature - self.mean) / self.std
        n_timestep = self.config.model["timestep"]

        # 填充到 timestep 的倍数
        if feature.shape[0] == 0:
            num_pad = n_timestep
        else:
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            if num_pad == n_timestep:
                num_pad = 0

        feature = np.pad(
            feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0
        )
        num_instance = feature.shape[0] // n_timestep

        # 边界情况：音频太短，无法形成一个完整的 timestep
        if num_instance == 0:
            # 至少需要填充到一个 timestep
            if feature.shape[0] < n_timestep:
                num_pad = n_timestep - feature.shape[0]
                feature = np.pad(
                    feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0
                )
                num_instance = 1

        # 进行预测（与 test.py 逻辑一致）
        lines = []
        start_time = 0.0
        prev_chord = None

        with torch.no_grad():
            feature_tensor = (
                torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            for t in range(num_instance):
                # 获取当前时间段的特征
                segment = feature_tensor[:, n_timestep * t : n_timestep * (t + 1), :]

                # 通过模型获取预测
                self_attn_output, _ = self.model.self_attn_layers(segment)
                prediction, _ = self.model.output_layer(self_attn_output)
                prediction = prediction.squeeze()

                # 转换为和弦标签（只在和弦变化时记录）
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        # 验证索引是否在字典中
                        if prev_chord not in self.idx_to_chord:
                            raise ValueError(
                                f"模型预测的索引 {prev_chord} 不在和弦字典中。"
                                f"可能的原因：模型类型与 large_voca 设置不匹配。"
                            )
                        continue

                    current_chord = prediction[i].item()
                    # 验证索引
                    if current_chord not in self.idx_to_chord:
                        raise ValueError(
                            f"模型预测的索引 {current_chord} 不在和弦字典中。"
                            f"可能的原因：模型类型与 large_voca 设置不匹配。"
                        )

                    if current_chord != prev_chord:
                        # 和弦变化，记录前一个和弦段
                        lines.append(
                            (
                                start_time,
                                feature_per_second * (n_timestep * t + i),
                                self.idx_to_chord[prev_chord],
                            )
                        )
                        start_time = feature_per_second * (n_timestep * t + i)
                        prev_chord = current_chord

                    # 处理最后一个时间段
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != feature_per_second * (n_timestep * t + i):
                            lines.append(
                                (
                                    start_time,
                                    feature_per_second * (n_timestep * t + i),
                                    self.idx_to_chord[prev_chord],
                                )
                            )
                        break

            # 如果没有任何和弦变化（整个音频只有一个和弦），至少返回第一个和弦段
            if not lines and prev_chord is not None:
                total_time = feature_per_second * (num_instance * n_timestep - num_pad)
                if total_time > 0:
                    lines.append((0.0, total_time, self.idx_to_chord[prev_chord]))

        merged_segments = lines

        # 根据输出格式返回
        if output_format == "list":
            return merged_segments
        elif output_format == "lab":
            lab_content = self._format_as_lab(merged_segments)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(lab_content)
            return lab_content
        elif output_format == "json":
            import json

            json_data = [
                {"start": start, "end": end, "chord": chord}
                for start, end, chord in merged_segments
            ]
            json_str = json.dumps(json_data, indent=2)
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "w") as f:
                    f.write(json_str)
            return json_str
        else:
            raise ValueError(
                f"不支持的输出格式: {output_format}. 支持: 'list', 'lab', 'json'"
            )

    def _merge_consecutive_chords(
        self, segments: List[Tuple[float, float, str]]
    ) -> List[Tuple[float, float, str]]:
        """
        合并连续相同和弦的时间段

        Args:
            segments: 原始和弦段列表

        Returns:
            合并后的和弦段列表
        """
        if not segments:
            return []

        merged = []
        current_start, current_end, current_chord = segments[0]

        for start, end, chord in segments[1:]:
            if chord == current_chord and abs(start - current_end) < 0.01:
                # 相同和弦且时间连续，合并
                current_end = end
            else:
                # 不同和弦或时间不连续，保存当前段，开始新段
                merged.append((current_start, current_end, current_chord))
                current_start, current_end, current_chord = start, end, chord

        # 添加最后一段
        merged.append((current_start, current_end, current_chord))

        return merged

    def _format_as_lab(self, segments: List[Tuple[float, float, str]]) -> str:
        """
        将和弦段格式化为 .lab 文件格式

        Args:
            segments: 和弦段列表

        Returns:
            .lab 格式的字符串
        """
        lines = []
        for start, end, chord in segments:
            lines.append(f"{start:.3f} {end:.3f} {chord}\n")
        return "".join(lines)

    def recognize_batch(
        self,
        audio_paths: List[Union[str, Path]],
        output_format: str = "list",
        save_dir: Optional[Union[str, Path]] = None,
        continue_on_error: bool = False,
    ) -> List[Union[List[Tuple[float, float, str]], str, Exception]]:
        """
        批量识别多个音频文件

        Args:
            audio_paths: 音频文件路径列表
            output_format: 输出格式
            save_dir: 保存目录（如果指定，会为每个文件生成对应的输出文件）
            continue_on_error: 如果为 True，遇到错误时继续处理其他文件，错误会作为结果返回

        Returns:
            每个音频文件的识别结果列表。如果 continue_on_error=True，错误会作为 Exception 对象返回
        """
        if not audio_paths:
            return []

        results = []
        for audio_path in audio_paths:
            try:
                save_path = None
                if save_dir:
                    audio_name = Path(audio_path).stem
                    if output_format == "lab":
                        save_path = Path(save_dir) / f"{audio_name}.lab"
                    elif output_format == "json":
                        save_path = Path(save_dir) / f"{audio_name}.json"

                result = self.recognize(
                    audio_path, output_format=output_format, save_path=save_path
                )
                results.append(result)
            except Exception as e:
                if continue_on_error:
                    results.append(e)
                else:
                    raise RuntimeError(
                        f"批量处理时在文件 {audio_path} 处失败: {str(e)}"
                    ) from e

        return results
