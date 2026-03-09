import argparse
import base64
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量 (默认读取 .env)
load_dotenv()


def substitute_env_vars(value):
    """递归替换配置中的环境变量占位符 ${VAR}"""
    if isinstance(value, str):
        # 查找 ${VAR_NAME} 模式
        pattern = re.compile(r"\$\{([^}]+)\}")
        matches = pattern.findall(value)
        for var_name in matches:
            env_val = os.getenv(var_name)
            if env_val:
                value = value.replace(f"${{{var_name}}}", env_val)
            else:
                # print(f"Warning: Environment variable {var_name} not found.")
                pass
        return value
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(v) for v in value]
    return value


def load_config(project_root: Path, config_path: str | None = None):
    """加载并解析 VLM 配置，优先独立配置文件。"""
    candidate_paths = []
    if config_path:
        candidate_paths.append(Path(config_path))
    candidate_paths.append(project_root / "vlm_config.yaml")
    candidate_paths.append(project_root / "settings.yaml")
    candidate_paths.append(project_root / "settings_template.yaml")

    selected_path = None
    for path in candidate_paths:
        if path.exists():
            selected_path = path
            break

    if selected_path is None:
        raise FileNotFoundError(
            "Configuration file not found. Please create vlm_config.yaml"
        )

    with open(selected_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    config = substitute_env_vars(raw_config)
    print(f"Loaded config: {selected_path.name}")
    return config


def resolve_vlm_config(config):
    """解析 VLM 配置：优先 vlm_config.yaml 的 vlm 节点，兼容旧结构。"""
    if isinstance(config, dict) and "vlm" in config and isinstance(config["vlm"], dict):
        vlm_config = config["vlm"]
    elif (
        isinstance(config, dict)
        and "models" in config
        and isinstance(config["models"], dict)
        and "default_vlm_model" in config["models"]
    ):
        print(
            "Warning: using legacy config path models.default_vlm_model, please migrate to vlm_config.yaml::vlm"
        )
        vlm_config = config["models"]["default_vlm_model"]
    else:
        print("Warning: no VLM section found, fallback to env/default values")
        vlm_config = {
            "api_key": os.getenv("GRAPHRAG_VLM_API_KEY"),
            "model": os.getenv("GRAPHRAG_VLM_MODEL", "qwen-vl-plus"),
            "api_base": os.getenv(
                "GRAPHRAG_VLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            "timeout": 120,
            "max_tokens": 1000,
            "temperature": 0.1,
        }

    if not vlm_config.get("api_key"):
        raise ValueError(
            "VLM api_key is missing. Set it in vlm_config.yaml or environment variable."
        )
    if not vlm_config.get("model"):
        raise ValueError("VLM model is missing in config.")

    return vlm_config


def get_vlm_client(config):
    """根据独立 VLM 配置初始化客户端。"""
    vlm_config = resolve_vlm_config(config)
    api_key = vlm_config.get("api_key")
    base_url = vlm_config.get("api_base")
    timeout = vlm_config.get("timeout", 120)

    client = OpenAI(
        api_key=api_key, base_url=base_url if base_url else None, timeout=timeout
    )
    return client, vlm_config


def encode_image(image_path):
    """将图片转换为 Base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_vlm_for_image(client, vlm_config, image_path):
    model_name = vlm_config.get("model", "qwen-vl-plus")
    print(f"Calling VLM ({model_name}) for {image_path.name}...")

    base64_image = encode_image(image_path)
    ext = image_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this academic figure. Explicitly list entities (proper nouns, models, chemical compounds) and their relationships depicted in the diagram. Summarize any quantitative data. Format the output as a concise description.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime};base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=int(vlm_config.get("max_tokens", 1000)),
            temperature=float(vlm_config.get("temperature", 0.1)),
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling API: {e}")
        return f"[ERROR PROCESSING IMAGE: {image_path.name}: {str(e)}]"


def process_text_files(project_root, config_path: str | None = None):
    # 1. 加载配置
    print(f"Loading configuration from {project_root}...")
    try:
        settings = load_config(project_root, config_path=config_path)
        client, vlm_config = get_vlm_client(settings)
        print(f"VLM initialized with model: {vlm_config.get('model', 'unknown')}")
    except Exception as e:
        print(f"Configuration error: {e}")
        return

    input_dir = project_root / "input"

    # 2. 准备正则
    # Pattern: [[FIGURE_REF: assets/images/hash.png | PAGE: 1]]
    pattern = re.compile(r"\[\[FIGURE_REF: (.*?) \| PAGE: (\d+)\]\]")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    # 3. 处理文件
    processed_count = 0
    for text_file in input_dir.glob("*_processed.txt"):
        print(f"Processing {text_file.name}...")

        with open(text_file, "r", encoding="utf-8") as f:
            content = f.read()

        # List to store descriptions for backup
        figure_descriptions = []

        def replace_match(match):
            rel_image_path = match.group(1)
            page_num = match.group(2)

            # 原始路径通常是相对于 project_root 的
            # 比如 "assets/images/xxx.png"
            full_image_path = project_root / rel_image_path.strip()

            if full_image_path.exists():
                description = call_vlm_for_image(client, vlm_config, full_image_path)
                formatted_desc = f"\n=== [FIGURE START: Page {page_num}] ===\n{description}\n=== [FIGURE END] ===\n"

                # Append to backup list
                figure_descriptions.append(f"Image: {rel_image_path}\n{formatted_desc}")

                return formatted_desc
            else:
                print(f"Warning: Image not found at {full_image_path}")
                return match.group(0)

        new_content = pattern.sub(replace_match, content)

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        # Save backup of figure descriptions
        if figure_descriptions:
            figures_file = text_file.with_name(text_file.stem + "_figures_only.txt")
            with open(figures_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(figure_descriptions))
            print(f"Saved figure descriptions backup to {figures_file.name}")

        print(f"Updated {text_file.name} with VLM descriptions.")
        processed_count += 1

    if processed_count == 0:
        print("No processed text files found. Did you run step 1?")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VLM 图像语义增强流水线（独立于 GraphRAG 配置）"
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).parent),
        help="项目根目录，默认当前脚本所在目录",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="VLM 配置文件路径，默认自动搜索 vlm_config.yaml",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root)
    process_text_files(project_root, config_path=args.config)
