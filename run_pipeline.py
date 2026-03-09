#!/usr/bin/env python3
"""
GraphRAG 多模态自动化构建管线
----------------------------------------
该脚本整合了多模态预处理（step1、step2）和 GraphRAG 官方命令（prompt-tune, index），
自动为输入的 PDF 文档构建知识图谱，并将所有中间产物和最终结果存放在 runs/ 下的独立目录中。

增强特性：
- 进度可视化：显示当前步骤序号 / 总步骤数，以及每个步骤的耗时。
- 详细日志：日志同时输出到控制台和运行目录下的 pipeline.log 文件。
- 健壮的错误处理：子进程失败时提供明确的错误信息并退出。
- 早期日志支持：即使在创建运行目录之前也能输出控制台日志。

Usage:
    python run_pipeline.py /path/to/paper.pdf [--run-name NAME] [--skip-prompt-tune] [--skip-index] [--verbose]


Date: 2026-02-22
Version: 2.2.0
"""

import os
import sys
import shutil
import subprocess
import argparse
import time
import re
import logging
from pathlib import Path
from datetime import datetime

# 可选：加载 .env 文件中的环境变量
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# -------------------- 配置与常量 --------------------
TEMPLATES_DIR = Path(__file__).parent / "templates"
RUNS_DIR = Path(__file__).parent / "runs"

# 日志格式
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 步骤序号计数器
step_counter = 0
total_steps = 0

# -------------------- 模块级 logger --------------------
logger = logging.getLogger("graphrag-pipeline")


# -------------------- 工具函数 --------------------
def ensure_dir(path: Path) -> Path:
    """确保目录存在，并返回该路径"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_folder_name(name: str, max_length: int = 50) -> str:
    """
    将字符串转换为安全的文件夹名：
    - 替换 Windows 非法字符（<>:"/\|?*）为下划线
    - 去除开头和结尾的空格和点
    - 将连续下划线压缩为一个
    - 截断到最大长度
    """
    # 替换非法字符
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    # 替换控制字符和除字母数字、空格、连字符外的其他字符
    safe = re.sub(r"[^\w\s\-]", "_", safe)
    # 将空白（包括空格）替换为下划线
    safe = re.sub(r"\s+", "_", safe)
    # 压缩多个下划线
    safe = re.sub(r"_+", "_", safe)
    # 去除首尾下划线和点
    safe = safe.strip("_.")
    if not safe:
        safe = "unnamed"
    if len(safe) > max_length:
        safe = safe[:max_length]
    return safe


def run_subprocess(cmd, cwd=None, env=None, verbose=False, step_name=None):
    """
    运行子进程，实时输出日志，失败时抛出异常。
    如果提供了 step_name，会在开始和结束时打印更友好的信息。
    """
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    if step_name:
        logger.info(f"[{step_name}] Running: {cmd_str}")
    else:
        logger.info(f"Running: {cmd_str}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=not verbose, text=True, check=False
        )
        elapsed = time.time() - start_time
        if result.returncode != 0:
            error_msg = (
                result.stderr.strip()
                if result.stderr
                else f"Command failed with code {result.returncode}"
            )
            logger.error(
                f"[{step_name}] Command failed after {elapsed:.1f}s: {error_msg}"
            )
            if verbose and result.stdout:
                logger.debug(f"[{step_name}] stdout: {result.stdout}")
            raise RuntimeError(f"Command '{cmd_str}' failed: {error_msg}")
        else:
            logger.info(f"[{step_name}] Completed successfully in {elapsed:.1f}s")
            if verbose and result.stdout:
                logger.debug(f"[{step_name}] stdout: {result.stdout}")
        return result
    except FileNotFoundError as e:
        logger.error(
            f"[{step_name}] Command not found: {cmd_str}. Make sure it's installed and in PATH."
        )
        raise
    except Exception as e:
        logger.error(f"[{step_name}] Unexpected error: {e}")
        raise


def setup_logging(run_dir: Path, verbose: bool = False):
    """配置日志：添加文件处理器（日志写入运行目录下的 pipeline.log），同时保留控制台输出"""
    log_file = run_dir / "pipeline.log"
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_file}")


def generate_topic_from_pdf(pdf_path: Path) -> str | None:
    """
    从 PDF 的前几页提取文本，调用 LLM 生成一句话主题（英文）。
    使用与 GraphRAG 相同的环境变量配置（GRAPHRAG_API_KEY, GRAPHRAG_API_BASE, GRAPHRAG_CHAT_MODEL）。
    """
    try:
        import fitz  # PyMuPDF
        from openai import OpenAI
    except ImportError as e:
        logger.warning(f"Missing dependency for topic generation: {e}")
        return None

    api_key = os.getenv("GRAPHRAG_API_KEY")
    if not api_key:
        logger.warning("GRAPHRAG_API_KEY not set; cannot generate topic.")
        return None

    # 提取 PDF 前 3 页文本
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(3, len(doc))):
            page = doc.load_page(i)
            text += page.get_text()
        doc.close()
        if not text.strip():
            logger.warning("No text extracted from PDF for topic generation.")
            return None
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF: {e}")
        return None

    # 截断以避免超长
    if len(text) > 3000:
        text = text[:3000]

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("GRAPHRAG_API_BASE", "https://api.openai.com/v1"),
    )
    model = os.getenv("GRAPHRAG_CHAT_MODEL", "gpt-4o-mini")

    try:
        logger.info("Calling LLM to generate topic...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Summarize the main topic of the provided text in a concise phrase "
                        "(5-10 words) in English. The phrase will be used as a folder name, "
                        "so use only alphanumeric characters, underscores, and hyphens. "
                        "No punctuation or special symbols."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.3,
            max_tokens=30,
            timeout=30,  # 设置超时，避免长时间阻塞
        )
        topic = response.choices[0].message.content.strip()
        # 再次安全化
        topic = re.sub(r"[^\w\-]", "_", topic)
        topic = re.sub(r"_+", "_", topic).strip("_")
        if len(topic) > 50:
            topic = topic[:50]
        logger.info(f"Generated topic: {topic}")
        return topic if topic else None
    except Exception as e:
        logger.warning(f"Topic generation failed: {e}")
        return None


def copy_template(template_name: str, dest_dir: Path, required: bool = False) -> bool:
    """从 templates 目录复制文件到目标目录，返回是否成功"""
    src = TEMPLATES_DIR / template_name
    if src.exists():
        shutil.copy2(src, dest_dir / template_name)
        logger.info(f"Copied template: {template_name}")
        return True
    else:
        msg = f"Template {template_name} not found in {TEMPLATES_DIR}"
        if required:
            logger.error(msg)
            raise FileNotFoundError(msg)
        else:
            logger.warning(msg)
        return False


def print_step_header(step_name: str, step_index: int, total: int):
    """打印步骤标题，带序号和可视化分隔线"""
    logger.info("=" * 60)
    logger.info(f"[{step_index}/{total}] {step_name}")
    logger.info("=" * 60)


# -------------------- 主管线逻辑 --------------------
def main():
    global step_counter, total_steps

    parser = argparse.ArgumentParser(
        description="GraphRAG 多模态自动化构建管线（增强版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
示例:
  python run_pipeline.py "E:\papers\my_paper.pdf"
  python run_pipeline.py "paper.pdf" --run-name "biomed_experiment" --skip-prompt-tune
  python run_pipeline.py "paper.pdf" --verbose
        """,
    )
    parser.add_argument("pdf_path", type=str, help="要处理的 PDF 文件路径")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="自定义运行目录名称（若未提供，则从 PDF 内容自动生成主题）",
    )
    parser.add_argument(
        "--vlm-config", type=str, default=None, help="外部 VLM 配置文件路径（覆盖模板）"
    )
    parser.add_argument(
        "--skip-prompt-tune",
        action="store_true",
        help="跳过 prompt-tune 步骤（使用已有的 prompts）",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="跳过索引构建步骤（仅运行 step1+step2）",
    )
    parser.add_argument("--verbose", action="store_true", help="显示子命令的详细输出")
    args = parser.parse_args()

    # 预先配置控制台日志（在运行目录创建之前也能输出）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # 计算总步骤数（用于进度显示）
    total_steps = 2  # step1, step2 总是执行
    if not args.skip_prompt_tune:
        total_steps += 1
    if not args.skip_index:
        total_steps += 1

    # 1. 验证 PDF 文件
    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    # 2. 确定运行目录名称
    if args.run_name:
        base_name = args.run_name
        logger.info(f"Using user-provided run name: {base_name}")
    else:
        logger.info("Generating topic from PDF...")
        topic = generate_topic_from_pdf(pdf_path)
        if topic:
            base_name = topic
            logger.info(f"Generated topic: {base_name}")
        else:
            base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.info(f"Falling back to timestamp: {base_name}")

    safe_name = safe_folder_name(base_name)
    run_dir = ensure_dir(RUNS_DIR / safe_name)

    # 避免目录冲突（添加数字后缀）
    counter = 1
    original_run_dir = run_dir
    while run_dir.exists() and any(run_dir.iterdir()):  # 非空目录才冲突
        run_dir = RUNS_DIR / f"{safe_name}_{counter}"
        counter += 1
    if counter > 1:
        logger.info(
            f"Directory {original_run_dir} already exists and is not empty, using {run_dir} instead."
        )
    ensure_dir(run_dir)

    # 设置文件日志（在运行目录创建后）
    setup_logging(run_dir, args.verbose)
    logger.info(f"Run directory created: {run_dir}")
    logger.info(f"Pipeline started with arguments: {args}")

    # 3. 复制 PDF 到运行目录下的 raw_pdfs
    raw_pdfs_dir = ensure_dir(run_dir / "raw_pdfs")
    dest_pdf = raw_pdfs_dir / pdf_path.name
    shutil.copy2(pdf_path, dest_pdf)
    logger.info(f"PDF copied to {dest_pdf}")

    # 4. 准备 VLM 配置文件
    if args.vlm_config:
        vlm_config_src = Path(args.vlm_config).resolve()
        if vlm_config_src.exists():
            shutil.copy2(vlm_config_src, run_dir / "vlm_config.yaml")
            logger.info(f"VLM config copied from user-provided file: {vlm_config_src}")
        else:
            logger.error(f"Specified VLM config not found: {vlm_config_src}")
            sys.exit(1)
    else:
        copy_template("vlm_config.yaml", run_dir, required=False)

    # 5. 准备 GraphRAG 配置文件 (settings.yaml)
    settings_copied = copy_template("settings.yaml", run_dir, required=False)
    if not settings_copied:
        logger.warning(
            "No settings.yaml template found; will attempt 'graphrag init' later."
        )

    # 6. 运行 step1 (多模态解析)
    step_counter = 1
    print_step_header("PDF Parsing (Step 1)", step_counter, total_steps)
    step1_script = Path(__file__).parent / "step1_multimodal_fast.py"
    if not step1_script.exists():
        logger.error("step1_multimodal_fast.py not found in the same directory.")
        sys.exit(1)
    try:
        run_subprocess(
            ["python", str(step1_script), "--project-root", str(run_dir)],
            cwd=run_dir,
            verbose=args.verbose,
            step_name=f"Step {step_counter}",
        )
    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        sys.exit(1)

    # 7. 运行 step2 (VLM 图像描述)
    step_counter += 1
    print_step_header("VLM Description Generation (Step 2)", step_counter, total_steps)
    step2_script = Path(__file__).parent / "step2_vlm_extract.py"
    if not step2_script.exists():
        logger.error("step2_vlm_extract.py not found.")
        sys.exit(1)
    try:
        run_subprocess(
            ["python", str(step2_script), "--project-root", str(run_dir)],
            cwd=run_dir,
            verbose=args.verbose,
            step_name=f"Step {step_counter}",
        )
    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        sys.exit(1)

    # 8. 检查 step2 是否生成了 processed 文件
    input_dir = run_dir / "input"
    if not list(input_dir.glob("*_processed.txt")):
        logger.error("No processed text files found after step2. Pipeline aborted.")
        sys.exit(1)

    # 9. 确保 settings.yaml 存在（如果模板未提供，运行 graphrag init）
    if not (run_dir / "settings.yaml").exists():
        logger.info("settings.yaml missing; running 'graphrag init'...")
        step_counter += 1
        print_step_header("GraphRAG Initialization", step_counter, total_steps)
        try:
            run_subprocess(
                ["graphrag", "init", "--root", str(run_dir)],
                cwd=run_dir,
                verbose=args.verbose,
                step_name=f"Step {step_counter}",
            )
        except Exception as e:
            logger.error(f"GraphRAG init failed: {e}")
            sys.exit(1)

    # 10. 运行 prompt-tune（除非跳过）
    if not args.skip_prompt_tune:
        step_counter += 1
        print_step_header(
            f"Prompt Tuning (Step {step_counter})", step_counter, total_steps
        )
        try:
            run_subprocess(
                [
                    "graphrag",
                    "prompt-tune",
                    "--root",
                    str(run_dir),
                    "--no-discover-entity-types",
                ],
                cwd=run_dir,
                verbose=args.verbose,
                step_name=f"Step {step_counter}",
            )
        except Exception as e:
            logger.error(f"Prompt tuning failed: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping prompt tuning as requested.")

    # 11. 运行索引构建（除非跳过）
    if not args.skip_index:
        step_counter += 1
        print_step_header(
            f"Knowledge Graph Indexing (Step {step_counter})", step_counter, total_steps
        )
        try:
            run_subprocess(
                ["graphrag", "index", "--root", str(run_dir), "--verbose"]
                if args.verbose
                else ["graphrag", "index", "--root", str(run_dir)],
                cwd=run_dir,
                verbose=args.verbose,
                step_name=f"Step {step_counter}",
            )
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            sys.exit(1)
    else:
        logger.info("Skipping index build as requested.")

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results are in: {run_dir}")
    logger.info("=" * 60)
    print(f"\n All done! Run directory: {run_dir}")
    print(f" Detailed log: {run_dir / 'pipeline.log'}")


if __name__ == "__main__":
    main()
