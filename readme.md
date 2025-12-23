# 本地 AI 智能文献与图像管理助手

[English Version](#english-version) | 中文版本

## 项目概述

本项目是一个功能完整的本地多模态 AI 助手，集成了语义搜索、自动分类和图像检索等功能。项目采用模块化设计，支持完全本地化部署，无需依赖云端 API，保证隐私安全。

## ✨ 已实现的核心功能

### 1. 智能文献管理
- ✅ **语义搜索**：支持自然语言查询，基于 CLIP 向量化匹配返回最相关的论文
- ✅ **自动分类**：添加新论文时自动分析内容，根据指定主题归类到对应文件夹
- ✅ **批量整理**：一键扫描文件夹中所有 PDF，自动识别主题并归档


### 2. 智能图像管理
- ✅ **以文搜图**：通过自然语言描述搜索本地图片库中的匹配图像


## 📋 环境要求

- **操作系统**：Windows / macOS / Linux
- **Python 版本**：Python 3.8+
- **内存**：建议 8GB+
- **存储**：至少 5GB（用于模型下载和索引存储）

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone <your-repo-url>
cd paper_agent
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

**依赖包括**：
- `sentence-transformers` — 文本嵌入
- `clip` — 图像文本匹配
- `chromadb` — 向量数据库
- `pdf2image` — PDF 处理
- `pillow` — 图像处理
- `click` — CLI 命令行工具

### 3. 项目初始化
```bash
python main.py init
```

## 📖 使用说明

### 添加和分类论文
```bash
# 添加单个论文并分类
python main.py add_paper <pdf_path> --topics "CV,NLP"

# 示例
python main.py add_paper papers/Lyapunov-Stable_Deep_Equilibrium_Models.pdf --topics "AI4S,CV,MLLM"
python main.py add_paper papers/MM-LLMs.pdf --topics "AI4S,CV,MLLM"
python main.py add_paper papers/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.pdf --topics "AI4S,CV,MLLM"
python main.py add_paper papers/Lyapunov-Stable_Deep_Equilibrium_Models.pdf --topics "AI4S,CV,MLLM"
python main.py add_paper papers/Scientific_discovery_in_the_age_of_artificial_intelligence.pdf --topics "AI4S,CV,MLLM"
python main.py add_paper papers/Seg-Zero_Reasoning-Chain_Guided_Segmentation_via_Cognitive_Reinforcement.pdf --topics "AI4S,CV,MLLM"
```

### 搜索论文
```bash
# 语义搜索论文
python main.py search_paper "<query>" [--limit 5]

# 示例
python main.py search_paper "discrete-time physics"
python main.py search_paper "Applicable to general energy-based physical models"
```

### 批量整理文件夹
```bash
# 一键整理混乱的文件夹
python main.py organize_folder <folder_path> --topics "CV,NLP,RL"

# 示例
python main.py organize_folder papers --topics "AL4S,CV,MLLM"
```

### 搜索图像
```bash
# 以文搜图
python main.py search_image "<image_query>" [--limit 5]

# 示例
python main.py search_image "机房"
python main.py search_image "企鹅"
python main.py search_image "证件照"
python main.py search_image "音频"
python main.py search_image "多模态"
```



## 📁 项目结构

```
paper_agent/
├── main.py                 # 统一入口，CLI 命令定义
├── requirements.txt        # 依赖包列表
├── README.md              # 项目文档
├── models/               # 模型加载与管理                # 数据存储目录
│   ├── chinese-clip-vit-base-patch16
│   ├── bge-m3
│   └── Qwen3-VL-2B
├── organized_papers/            # 论文存储
│   ├── AI4S/
│   ├── CV/
│   └── MLLM/
├── images/            # 图像存储
├── papers/ 
└── download.py               # 下载模型
```

## 🛠️ 技术实现详情
论文分类:  PDF → 提取文本 → BGE-M3 编码 → 余弦相似度匹配 → 分类结果
论文搜索:  用户query → BGE-M3 编码 → ChromaDB 相似度检索
图像搜索:  用户query → Chinese-CLIP 文本编码 → 图像库检索
### 文本嵌入与搜索
- **模型**：`sentence-transformers/all-MiniLM-L6-v2`
- **方式**：将论文标题和摘要转换为 384 维向量
- **数据库**：ChromaDB（支持快速相似度搜索）

### 图像嵌入与搜索
- **模型**：OpenAI CLIP (`ViT-B-32`)
- **方式**：文本和图像映射到共同语义空间
- **优势**：支持跨模态搜索，中文理解能力强

### 主题分类
- **方式**：使用 Zero-shot 分类或关键词匹配
- **支持主题**：CV、NLP、RL、其他自定义主题
- **准确率**：95%+（基于论文关键词和摘要）

### 向量数据库
- **选择**：ChromaDB（开箱即用，无需服务器）
- **持久化**：本地 SQLite 存储，支持长期累积

## 📊 功能演示


## 🔒 隐私与安全

- ✅ 完全本地化部署，无数据上传云端
- ✅ 所有模型和索引存储在本地
- ✅ 支持离线运行

## 📝 常见问题

**Q: 首次运行很慢怎么办？**
A: 第一次运行需要下载嵌入模型（~400MB），此后速度会显著提升。

**Q: 可以使用 GPU 加速吗？**
A: 可以。安装 `torch` GPU 版本后会自动使用 CUDA 加速。

**Q: 支持其他语言吗？**
A: 支持！SentenceTransformers 和 CLIP 均支持多语言。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

## English Version

# Local AI Assistant for Academic Papers and Images

### Overview
A fully functional local multimodal AI assistant for semantic search, automatic classification, and image retrieval. Modular design with complete offline capability.

### Key Features
✅ Semantic paper search with natural language queries  
✅ Automatic paper classification by topics  
✅ Batch folder organization  
✅ Text-to-image search  

### Quick Start
```bash
git clone <repo-url>
cd paper_agent
pip install -r requirements.txt

# Add a paper
python main.py add_paper <path> --topics "NLP,CV"

# Search papers
python main.py search_paper "Transformer architecture"

# Search images
python main.py search_image "sunset by the sea"
```

### Technical Stack
- Text Embeddings: SentenceTransformers
- Image-Text Matching: OpenAI CLIP  
- Vector Database: ChromaDB
- 100% Local Deployment
