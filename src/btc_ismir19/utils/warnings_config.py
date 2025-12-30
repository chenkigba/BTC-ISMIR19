"""
统一的警告配置模块

此模块用于集中管理项目中的警告抑制策略，避免在多个文件中重复配置。

说明：
1. Keras/TensorFlow 的 FutureWarning 来自上游库（keras/src/export/tf2onnx_lib.py），
   这是上游库与 NumPy 2.x 的兼容性问题，会在未来版本中修复。
   抑制这些警告是合理的临时方案。

2. 我们自己的代码已修复所有 NumPy 2.x 兼容性问题（np.int -> np.int64 等），
   这些修复是"治本"的。

3. 当上游库修复后，可以移除相关的警告抑制。
"""

import warnings


def configure_warnings():
    """
    配置项目级别的警告过滤器。

    在项目入口文件的最开始调用此函数，确保在所有导入之前设置警告过滤器。
    """
    # 抑制用户警告（通常是库的兼容性提示）
    warnings.filterwarnings("ignore", category=UserWarning)

    # 抑制来自 Keras/TensorFlow 的 FutureWarning（上游库问题）
    # 这些警告来自 keras/src/export/tf2onnx_lib.py，使用 np.object 检查
    warnings.filterwarnings("ignore", category=FutureWarning, module="keras.*")
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*np\\.object.*")

    # 注意：我们自己的代码已修复所有 NumPy 2.x 兼容性问题，不需要抑制其他 FutureWarning


# 自动配置（当模块被导入时）
configure_warnings()
