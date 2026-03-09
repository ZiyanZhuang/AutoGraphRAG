# AutoGraphRAG: Automated Multimodal Knowledge Graph Construction Pipeline

AutoGraphRAG is a production-ready pipeline that transforms academic PDF documents (including figures and diagrams) into a fully structured knowledge graph using GraphRAG. It integrates multimodal pre[...]

## ✨ Key Features

- **End-to-End Automation** – From raw PDF to queryable knowledge graph in one command.
- **Multimodal Understanding** – Extracts and describes figures via a Vision Language Model (VLM), enriching the knowledge graph with visual information.
- **Intelligent Run Naming** – Automatically generates a concise topic phrase from the PDF content to name the output folder.
- **Isolated Runs** – Every execution creates a new subdirectory under `runs/` (automatically ignored by git) containing inputs, logs, prompts, and outputs.
- **Modular & Extensible** – Built with clear separation of concerns; individual steps can be skipped or replaced.

## 📋 System Requirements

- **Python** 3.10+
- **GraphRAG** 0.5.0+
- **API Access** to an OpenAI‑compatible endpoint.

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ZiyanZhuang/AutoGraphRAG.git
   cd AutoGraphRAG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GRAPHRAG_API_KEY=your_openai_api_key
   GRAPHRAG_API_BASE=https://api.openai.com/v1
   GRAPHRAG_CHAT_MODEL=gpt-4o-mini

   # VLM Configuration (e.g., Qwen-VL)
   GRAPHRAG_VLM_API_KEY=your_vlm_api_key
   GRAPHRAG_VLM_MODEL=qwen-vl-plus
   GRAPHRAG_VLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

## 🚀 Usage

Run the pipeline with a single command:
```bash
python run_pipeline.py /path/to/your/document.pdf
```

### Optional Arguments

- `--run-name NAME`: Specify a custom name for the run directory.
- `--skip-prompt-tune`: Skip the prompt-tuning step if you already have custom prompts.
- `--verbose`: Show detailed output of all sub-commands.

## 📂 Project Structure

- `run_pipeline.py`: The main orchestrator.
- `step1_multimodal_fast.py`: PDF parsing and image extraction.
- `step2_vlm_extract.py`: VLM-powered image description.
- `templates/`: Configuration and prompt templates.
- `settings_template.yaml`: Template for GraphRAG settings.

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

- **Attribution**: You must give appropriate credit.
- **Non-Commercial**: You may not use the material for commercial purposes.
- **ShareAlike**: If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

---
Built with ❤️ for the GraphRAG community.
```

---

