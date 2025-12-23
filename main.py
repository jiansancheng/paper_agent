# main.py —— 本地 Qwen3-VL + 本地 Embedding 版本
# python main.py add_paper papers/Lyapunov-Stable_Deep_Equilibrium_Models.pdf --topics "AI4S,CV,MLLM"
# python main.py add_paper papers/MM-LLMs.pdf --topics "AI4S,CV,MLLM"
# python main.py add_paper papers/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.pdf --topics "AI4S,CV,MLLM"
# python main.py add_paper papers/Lyapunov-Stable_Deep_Equilibrium_Models.pdf --topics "AI4S,CV,MLLM"
# python main.py add_paper papers/Scientific_discovery_in_the_age_of_artificial_intelligence.pdf --topics "AI4S,CV,MLLM"
# python main.py add_paper papers/Seg-Zero_Reasoning-Chain_Guided_Segmentation_via_Cognitive_Reinforcement.pdf --topics "AI4S,CV,MLLM"
# python main.py search_paper "discrete-time physics"
# python main.py search_paper "Applicable to general energy-based physical models"
# python main.py organize_folder papers --topics "AL4S,CV,MLLM"
#python main.py search_image "机房"
#python main.py search_image "企鹅"
#python main.py search_image "证件照"
#python main.py search_image "音频"
#python main.py search_image "多模态"
#rm -rf ./embeddings
import os
import shutil
import argparse
import re
import hashlib
import traceback
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
import chromadb
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    ChineseCLIPProcessor, 
    ChineseCLIPModel,
    logging as hf_logging
)
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForImageTextToText, AutoProcessor # 把 AutoModelForImageTextToText 换成这个
# --- 配置 ---
# 压制 Transformers 的啰嗦警告
hf_logging.set_verbosity_error()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 模型路径配置 (请根据实际情况调整) ===
QWEN_MODEL_PATH = "/data0/jycheng/homework/paper_agent/models/Qwen3-VL-2B"
EMBEDDING_MODEL_PATH = "/data0/jycheng/homework/paper_agent/models/bge-m3"
CHINESE_CLIP_PATH = "/data0/jycheng/homework/paper_agent/models/chinese-clip-vit-base-patch16"

# === 全局变量 (懒加载) ===
_qwen_model = None
_qwen_processor = None
_embedding_model = None
_chroma_client = None
_cclip_model = None
_cclip_processor = None
# 1. 确保头部导入包含 AutoModelForImageTextToText
from transformers import AutoModelForImageTextToText, AutoProcessor

def get_qwen_model():
    """
    [修复版] 适配 Qwen3-VL
    不再硬编码 Qwen2 类，而是使用 AutoModel 自动识别模型结构
    """
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor
    
    print(f"🧠 Loading Qwen3-VL from: {QWEN_MODEL_PATH} ...")
    try:
        # 加载 Processor (图像/文本预处理)
        _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
        
        # 【核心修改】使用 AutoModelForImageTextToText
        # trust_remote_code=True 会允许 Qwen3 执行它文件夹里的 python 代码来定义它自己
        _qwen_model = AutoModelForImageTextToText.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True 
        ).eval()
        
        print("✅ Qwen3-VL Loaded Successfully!")
        
    except Exception as e:
        print(f"⚠️ Qwen Load Failed: {e}")
        # 如果 AutoModel 也挂了，打印详细错误栈以便调试
        traceback.print_exc()
        return None, None
        
    return _qwen_model, _qwen_processor
# -----------------------------------------------------------------------------
# 1. 模型加载函数 (Lazy Loading)
# -----------------------------------------------------------------------------

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./embeddings")
    return _chroma_client

def get_embedding_model():
    """加载 BGE-M3 (用于论文文本检索)"""
    global _embedding_model
    if _embedding_model is None:
        print("🔤 Loading BGE-M3 embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    return _embedding_model



def get_chinese_clip():
    """加载 Chinese-CLIP (用于图片搜索，All-in-One)"""
    global _cclip_model, _cclip_processor
    if _cclip_model is None:
        print("🇨🇳 Loading Chinese-CLIP (All-in-One)...")
        try:
            _cclip_model = ChineseCLIPModel.from_pretrained(CHINESE_CLIP_PATH).to(DEVICE).eval()
            _cclip_processor = ChineseCLIPProcessor.from_pretrained(CHINESE_CLIP_PATH)
        except Exception as e:
            print(f"❌ Chinese-CLIP Load Failed: {e}")
            raise e
    return _cclip_model, _cclip_processor

# -----------------------------------------------------------------------------
# 2. 辅助工具函数
# -----------------------------------------------------------------------------

def clean_thinking_content(raw_text: str) -> str:
    """清洗 Qwen 的思维链输出，只保留最终答案"""
    if not raw_text: return ""
    # 移除 <think> 标签
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 移除常见前缀
    if "Answer:" in clean_text: clean_text = clean_text.split("Answer:")[-1]
    elif "Category:" in clean_text: clean_text = clean_text.split("Category:")[-1]
    # 移除 Markdown 和标点
    clean_text = clean_text.replace("*", "").replace("`", "").replace('"', "").replace("'", "").strip()
    return clean_text

def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
def compute_clip_embedding(text=None, image=None):
    """
    [终极修复版] Chinese-CLIP 统一向量计算接口
    修复 IndexError: tuple index out of range
    """
    model, processor = get_chinese_clip()
    
    with torch.no_grad():
        if image is not None:
            # === 图片编码 (保持不变) ===
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            feats = model.get_image_features(**inputs)
            
        elif text is not None:
            # === 文本编码 (手动提取 [CLS] Token) ===
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            
            # 1. 调用底层 text_model
            text_outputs = model.text_model(**inputs)
            
            # 2. 安全获取 last_hidden_state
            # 有些版本是对象，有些是元组，这里做个双重保险
            if hasattr(text_outputs, "last_hidden_state"):
                last_hidden_state = text_outputs.last_hidden_state
            else:
                last_hidden_state = text_outputs[0]
            
            # 3. 提取 [CLS] Token (即序列的第一个 token)
            # Shape: [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            pooled_output = last_hidden_state[:, 0, :]
            
            # 4. 投影到 CLIP 空间
            feats = model.text_projection(pooled_output)
            
        else:
            return None
        
        # L2 归一化 (CLIP 必需)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats[0].cpu().numpy().tolist()

# -----------------------------------------------------------------------------
# 3. 核心功能：论文管理 (Qwen + BGE)
# -----------------------------------------------------------------------------
def classify_paper_with_qwen(text: str, topics: List[str]) -> str:
    """
    [优化版] 使用 BGE-M3 进行基于语义相似度的分类 (比 2B 大模型更准、更快)
    """
    # 1. 获取 Embedding 模型
    emb_model = get_embedding_model()
    
    # 2. 截取摘要 (前 1000 个字符足够判断类别了)
    abstract = text[:1000]
    
    print(f"🧠 Classifying via Embedding Similarity...")
    
    # 3. 计算相似度
    # 编码 "类别词" (如 AI4S, CV, MLLM)
    topic_embeddings = emb_model.encode(topics, normalize_embeddings=True)
    # 编码 "论文摘要"
    paper_embedding = emb_model.encode(abstract, normalize_embeddings=True)
    
    # 4. 计算余弦相似度 (点积)
    # paper_embedding @ topic_embeddings.T
    similarities = np.dot(topic_embeddings, paper_embedding)
    
    # 5. 找出分最高的那个
    best_idx = np.argmax(similarities)
    best_topic = topics[best_idx]
    best_score = similarities[best_idx]
    
    # 打印一下具体的得分，让你看到它为什么选这个
    # 方便你调试：如果 CV 的分也很高，说明分类本身有重叠
    score_debug = ", ".join([f"{t}: {s:.2f}" for t, s in zip(topics, similarities)])
    print(f"📊 Scores: [{score_debug}]")
    print(f"✅ Picked: {best_topic}")
    
    return best_topic
def add_paper(pdf_path: str, topics: List[str]):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # 0. 查重逻辑
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="papers")
    existing = collection.get(where={"filename": pdf_path.name}, limit=1)
    if existing['ids']:
        print(f"⏩ Skipped (Already indexed): {pdf_path.name}")
        return

    print(f"📄 Processing: {pdf_path.name}")
    
    # === 1. 读取 PDF 并提取文本用于分类 ===
    try:
        reader = PdfReader(str(pdf_path))
        # 提取前两页的内容用于分类就够了 (通常摘要在第一页)
        full_text_for_classify = ""
        for i in range(min(2, len(reader.pages))):
            full_text_for_classify += reader.pages[i].extract_text() or ""
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return

    # === 2. 分类 (使用 Embedding 快速分类) ===
    print("🧠 Classifying...")
    topic = classify_paper_with_qwen(full_text_for_classify, topics)
    print(f"✅ Classified as: {topic}")

    # === 3. 移动文件 ===
    target_dir = Path("organized_papers") / topic
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / pdf_path.name
    if not dest.exists():
        shutil.copy(pdf_path, dest)
    print(f"✅ Saved to: {dest}")

    # === 4. 按页切片并建立索引 (保留页码信息) ===
    print("✂️  Chunking & Indexing per page...")
    emb_model = get_embedding_model()
    
    CHUNK_SIZE = 500
    OVERLAP = 50
    file_hash = hashlib.md5(str(dest.resolve()).encode('utf-8')).hexdigest()
    
    batch_chunks = []
    batch_ids = []
    batch_metadatas = []

    # 遍历每一页
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text: continue
        
        # 清洗一下，去掉多余换行，方便阅读
        page_text = page_text.replace('\n', ' ')
        
        # 在当前页内进行切片
        for i in range(0, len(page_text), CHUNK_SIZE - OVERLAP):
            chunk = page_text[i : i + CHUNK_SIZE]
            if len(chunk) < 50: continue # 太短的忽略
            
            chunk_id = f"{file_hash}_p{page_idx+1}_{i}" # ID里也加上页码
            
            batch_chunks.append(chunk)
            batch_ids.append(chunk_id)
            # 【关键】这里记录 page 字段
            batch_metadatas.append({
                "filename": pdf_path.name,
                "topic": topic,
                "path": str(dest),
                "page": page_idx + 1, # 记录页码 (从1开始)
                "chunk_index": i
            })

    # 批量入库
    if batch_chunks:
        # 分批 embedding 防止爆显存
        batch_size = 32
        for i in range(0, len(batch_chunks), batch_size):
            end = i + batch_size
            sub_chunks = batch_chunks[i:end]
            sub_ids = batch_ids[i:end]
            sub_metas = batch_metadatas[i:end]
            
            embeddings = emb_model.encode(sub_chunks, normalize_embeddings=True).tolist()
            collection.add(embeddings=embeddings, documents=sub_chunks, metadatas=sub_metas, ids=sub_ids)
            
        print(f"✅ Indexed {len(batch_chunks)} chunks from {len(reader.pages)} pages.")

def organize_folder(folder_path: str, topics: List[str]):
    source_dir = Path(folder_path)
    pdfs = list(source_dir.glob("**/*.pdf"))
    print(f"📂 Found {len(pdfs)} PDFs in {folder_path}")
    for pdf in pdfs:
        try:
            add_paper(str(pdf), topics)
        except Exception as e:
            print(f"❌ Skip {pdf.name}: {e}")
def search_paper(query: str, top_k: int = 5, simple: bool = False): # <--- 改动1: 增加 simple 参数
    emb_model = get_embedding_model()
    query_emb = emb_model.encode([query], normalize_embeddings=True)[0].tolist()
    
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="papers")
    
    # 如果是简洁模式，我们可能想多要把一点候选，然后去重
    n_results = top_k * 3 if simple else top_k
    results = collection.query(query_embeddings=[query_emb], n_results=n_results)
    
    print(f"\n🔍 Results for: '{query}'\n" + "="*60)
    if not results['ids'][0]:
        print("No matches.")
        return

    # === 新增：简洁列表模式 ===
    if simple:
        seen_files = set() # 用于去重
        print("📂 Relevant Files List (Unique):")
        
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            filename = meta['filename']
            
            # 如果这个文件之前没出现过，就打印
            if filename not in seen_files:
                print(f"📄 {filename}")
                print(f"   path: {meta['path']}")
                seen_files.add(filename)
                
        print("="*60)
        return
    # ==========================

    # 原有的详细模式
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        content = results['documents'][0][i]
        page_num = meta.get('page', 'Unknown')
        
        print(f"📄 File: {meta['filename']}")
        print(f"🏷️  Topic: {meta['topic']}")
        print(f"📖 Page:  {page_num}")
        print(f"📍 Context: ...{content}...")
        print("-" * 60)
# -----------------------------------------------------------------------------
# 4. 核心功能：图片搜索 (Chinese-CLIP All-in-One)
# -----------------------------------------------------------------------------

def add_images_clip(image_dir: str = "images"):
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="images_chinese_clip")
    
    # 预加载模型检查
    get_chinese_clip()
    
    img_dir_path = Path(image_dir)
    if not img_dir_path.exists():
        print(f"❌ Folder not found: {image_dir}")
        return

    image_paths = [p for p in img_dir_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    new_count = 0

    for img_path in image_paths:
        # 1. MD5 去重
        path_str = str(img_path.resolve())
        img_id = hashlib.md5(path_str.encode('utf-8')).hexdigest()
        
        try:
            if collection.get(ids=[img_id])['ids']: continue
        except: pass
        
        # 2. 编码入库
        try:
            print(f"⚡ Indexing: {img_path.name}...")
            image_obj = Image.open(img_path).convert("RGB")
            emb = compute_clip_embedding(image=image_obj)
            
            collection.add(
                embeddings=[emb],
                metadatas=[{"path": str(img_path), "method": "chinese_clip"}],
                ids=[img_id]
            )
            new_count += 1
        except Exception as e:
            print(f"❌ Error {img_path.name}: {e}")

    if new_count > 0: print(f"✅ Added {new_count} new images.")
    else: print("✅ Image index up-to-date.")

def search_image(query: str, top_k: int = 3):
    print(f"🇨🇳 Searching with Chinese-CLIP for: '{query}'")
    add_images_clip() # 自动更新索引
    
    query_emb = compute_clip_embedding(text=query)
    
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="images_chinese_clip")
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    
    print(f"\n🖼️  Results:\n" + "="*60)
    if not results['ids'][0]:
        print("No images found.")
        return

    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i] if 'distances' in results else None
        
        # 将欧氏距离转换为相似度分数 (0-1 之间)
        # ChromaDB 默认使用欧氏距离，距离越小越相似
        # 相似度 = 1 / (1 + distance)
        similarity = 1 / (1 + distance) if distance is not None else 0.0
        
        print(f"{i+1}. {os.path.basename(meta['path'])}")
        print(f"   Path: {meta['path']}")
        print(f"   ⭐ Similarity: {similarity:.2%}")
        print()
    print("="*60)
# -----------------------------------------------------------------------------
# 5. CLI 主程序
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Paper & Image Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. Add Paper
    p_add = subparsers.add_parser("add_paper")
    p_add.add_argument("path", type=str)
    p_add.add_argument("--topics", type=str, default="AI_for_Science,CV,MLLM")

    # 2. Organize Folder
    p_org = subparsers.add_parser("organize_folder")
    p_org.add_argument("folder_path", type=str)
    p_org.add_argument("--topics", type=str, default="AI_for_Science,CV,MLLM")

    # 3. Search Paper
    p_search_p = subparsers.add_parser("search_paper")
    p_search_p.add_argument("query", type=str)
    # 新增开关，不带这个参数就是 False，带了就是 True
    p_search_p.add_argument("--simple", action="store_true", help="Only list filenames")

    # 4. Search Image
    p_search_i = subparsers.add_parser("search_image")
    p_search_i.add_argument("query", type=str)

    args = parser.parse_args()

    if args.command == "add_paper":
        topics = [t.strip() for t in args.topics.split(",")]
        add_paper(args.path, topics)
    elif args.command == "organize_folder":
        topics = [t.strip() for t in args.topics.split(",")]
        organize_folder(args.folder_path, topics)
    elif args.command == "search_paper":
        # 传递 simple 参数
        search_paper(args.query, simple=args.simple)
    elif args.command == "search_image":
        search_image(args.query)

if __name__ == "__main__":
    main()