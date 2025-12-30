from pathlib import Path


def get_package_resource_path(resource_name: str) -> Path:
    """
    获取包内资源文件的路径

    Args:
        resource_name: 资源文件名（如 'run_config.yaml' 或 'models/btc_model.pt'）

    Returns:
        资源文件的路径
    """
    # 使用包目录定位资源，这在所有安装方式下都可靠
    # （editable install、wheel install、直接运行）
    package_dir = Path(__file__).parent.parent
    return package_dir / resource_name

