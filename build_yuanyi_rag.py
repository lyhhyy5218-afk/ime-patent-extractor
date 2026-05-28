#!/usr/bin/env python3
"""
元易创新方法 RAG 知识库构建脚本
将元易书稿、概述、应用详解、应用实例等知识文档分块，用dashscope embedding建ChromaDB向量库。
"""
import os
import re
import json
import time
import chromadb
from chromadb.config import Settings
import dashscope
from dashscope import TextEmbedding

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
DASHSCOPE_API_KEY = 'sk-4af4ab13c47d4408aabaa93c32273a48'
CHROMA_DIR = '/tmp/ime-patent-extractor/yuanyi_rag_chroma'
COLLECTION_NAME = 'yuanyi_knowledge'
CHUNK_SIZE = 500      # 每块字符数
CHUNK_OVERLAP = 100   # 块间重叠

dashscope.api_key = DASHSCOPE_API_KEY

# ═══════════════════════════════════════════
# 知识源文件（已提取和清洗的高质量文本）
# ═══════════════════════════════════════════
KNOWLEDGE_DIR = '/tmp/ime-patent-extractor/yuanyi_knowledge'
KNOWLEDGE_SOURCES = {
    '九维定义': os.path.join(KNOWLEDGE_DIR, '01_九维定义.txt'),
    '九法定义': os.path.join(KNOWLEDGE_DIR, '02_九法定义.txt'),
    '维法耦合详解': os.path.join(KNOWLEDGE_DIR, '03_维法耦合详解.txt'),
    '元易书稿全文': '/tmp/元易书稿.txt',
}


def load_text(filepath):
    """加载文本文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def clean_text(text):
    """清洗文本：去除页码、多余空行"""
    # 去除独立一行的数字（页码）
    text = re.sub(r'\n\d+\n', '\n', text)
    # 去除连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_chunks(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    按语义段落分块，保留来源信息
    策略：先按段落分割，再将过长的段落按chunk_size切割
    """
    chunks = []
    # 按双换行分段落
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前块+新段落不超长，合并
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            # 保存当前块
            if current_chunk:
                chunks.append({
                    'id': f'{source_name}_{chunk_idx}',
                    'text': current_chunk,
                    'source': source_name,
                    'chunk_idx': chunk_idx,
                })
                chunk_idx += 1

            # 如果单段落超长，按chunk_size切割
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    piece = para[start:end]
                    chunks.append({
                        'id': f'{source_name}_{chunk_idx}',
                        'text': piece,
                        'source': source_name,
                        'chunk_idx': chunk_idx,
                    })
                    chunk_idx += 1
                    start = end - overlap
            else:
                current_chunk = para

    # 最后一块
    if current_chunk:
        chunks.append({
            'id': f'{source_name}_{chunk_idx}',
            'text': current_chunk,
            'source': source_name,
            'chunk_idx': chunk_idx,
        })

    return chunks


def get_embeddings_batch(texts, batch_size=6):
    """调用dashscope获取文本embedding"""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = TextEmbedding.call(
            model='text-embedding-v3',
            input=batch,
            dimension=1024,
        )

        if resp.status_code == 200:
            batch_embs = [item['embedding'] for item in resp.output['embeddings']]
            all_embeddings.extend(batch_embs)
        else:
            print(f"  Embedding错误 (batch {i}): {resp.message}")
            all_embeddings.extend([[0.0] * 1024] * len(batch))

        if i + batch_size < len(texts):
            time.sleep(0.3)

    return all_embeddings


def build_rag():
    """构建RAG知识库主流程"""
    print("=" * 60)
    print("元易创新方法 RAG 知识库构建")
    print("=" * 60)

    # Step 1: 加载所有知识源
    all_chunks = []
    for name, filepath in KNOWLEDGE_SOURCES.items():
        if not os.path.exists(filepath):
            print(f"  [跳过] {name}: {filepath} 不存在")
            continue
        print(f"\n[加载] {name}: {filepath}")
        text = load_text(filepath)
        text = clean_text(text)
        chunks = split_into_chunks(text, name)
        print(f"  生成 {len(chunks)} 个知识块")
        all_chunks.extend(chunks)

    print(f"\n总知识块数: {len(all_chunks)}")

    if not all_chunks:
        print("没有知识块可处理，退出")
        return

    # Step 2: 生成embedding
    print(f"\n[Embedding] 生成向量 ({len(all_chunks)} 条)...")
    texts = [c['text'] for c in all_chunks]
    embeddings = get_embeddings_batch(texts)
    print(f"  完成，维度: {len(embeddings[0])}")

    # Step 3: 存入ChromaDB
    print(f"\n[ChromaDB] 存储到 {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 如果已有collection先删除
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "元易创新方法九维九法知识库"}
    )

    # 批量插入
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[c['id'] for c in batch],
            documents=[c['text'] for c in batch],
            embeddings=embeddings[i:i + batch_size],
            metadatas=[{'source': c['source'], 'chunk_idx': c['chunk_idx']} for c in batch],
        )

    print(f"  存入 {len(all_chunks)} 条知识块")

    # Step 4: 验证
    print("\n[验证] 测试检索...")
    test_queries = [
        "在转子铁芯开设通风槽属于什么维度和法则",
        "什么是分解与去除法则",
        "结构维的定义是什么",
        "组合与集成法则有哪些具体方法",
    ]
    for q in test_queries:
        results = query_rag(collection, q, top_k=2)
        print(f"\n  Q: {q}")
        for r in results:
            print(f"    → [{r['source']}] {r['text'][:80]}...")

    print("\n" + "=" * 60)
    print("RAG知识库构建完成！")
    print(f"路径: {CHROMA_DIR}")
    print(f"知识块: {len(all_chunks)}")
    print("=" * 60)

    return collection


def query_rag(collection, query, top_k=3):
    """查询RAG知识库"""
    resp = TextEmbedding.call(
        model='text-embedding-v3',
        input=[query],
        dimension=1024,
    )
    query_emb = resp.output['embeddings'][0]['embedding']

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
    )

    output = []
    for i in range(len(results['ids'][0])):
        output.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'distance': results['distances'][0][i],
        })
    return output


def load_rag():
    """加载已有的RAG知识库"""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"已加载RAG知识库: {collection.count()} 条知识块")
    return collection


if __name__ == '__main__':
    build_rag()
