from sentence_transformers import SentenceTransformer
import chromadb
import os
from sentence_transformers import SentenceTransformer
import chromadb
import os
import torch
import re

class RAGRetriever:
    def __init__(self, embed_model="shibing624/text2vec-base-chinese", persist_dir="./chroma_db"):
        if torch.cuda.is_available():
            device = 'cuda'
            print(" 使用 GPU 进行向量编码")
        else:
            device = 'cpu'
            print(" 使用 CPU 进行向量编码")
        self.embedder = SentenceTransformer(embed_model, device=device)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("doc_chunks")
        print(f" 知识库已加载，当前块数: {self.collection.count()}")

    def add_document(self, document, doc_id="doc", chunk_size=200, overlap=100, person_name=None):
        """
        将文档切块并添加到向量库，可指定人名
        """
        if not document or len(document.strip()) == 0:
            print(f" 文档 {doc_id} 为空，跳过")
            return 0

        chunks = []
        start = 0
        while start < len(document):
            end = min(start + chunk_size, len(document))
            chunks.append(document[start:end])
            start += chunk_size - overlap

        if not chunks:
            return 0

        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {"doc_id": doc_id, "chunk_index": i, "length": len(chunk)}
            if person_name:
                meta["person_name"] = person_name
            metadatas.append(meta)

        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f" 文档 '{doc_id}' 已入库，新增 {len(chunks)} 个块")
        if person_name:
            print(f"   👤 关联人名: {person_name}")
        return len(chunks)

    def retrieve(self, question, top_k=4, person_name=None):
        """
        检索最相关的 top_k 个块，可指定人名过滤
        """
        if not question or not question.strip():
            return []
        where_filter = {}
        if person_name:
            where_filter["person_name"] = person_name
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        if not results['documents'] or not results['documents'][0]:
            return []
        docs = []
        for i, text in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            doc_id = metadata.get('doc_id', 'unknown')
            docs.append((doc_id, text))
        return docs

    def extract_person_name_from_text(self, text):
        """
        从文本中提取姓名（支持“姓名：XXX”或“名字：XXX”）
        """
        patterns = [
            r'(?:姓名|名字)[：:]\s*([\u4e00-\u9fa5]{2,4})',
            r'^([\u4e00-\u9fa5]{2,4})\s*[,，、]'  # 开头的人名
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None

    def extract_person_name_from_question(self, question):
        """
        从问题中提取可能的人名（中文姓名）
        """
        # 匹配“某某的”或“某某是”等结构
        pattern = r'([\u4e00-\u9fa5]{2,4})(?:的|在|是|有|和)'
        matches = re.findall(pattern, question)
        return matches[0] if matches else None

    def get_all_doc_ids(self):
        all_meta = self.collection.get(include=["metadatas"])
        if all_meta and all_meta['metadatas']:
            doc_ids = set([m['doc_id'] for m in all_meta['metadatas']])
            return list(doc_ids)
        return []

    def count_documents(self):
        return self.collection.count()

    def clear_all(self):
        self.client.delete_collection("doc_chunks")
        self.collection = self.client.get_or_create_collection("doc_chunks")
        print("️ 知识库已清空")

    def delete_document(self, doc_id):
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f" 文档 '{doc_id}' 已删除，移除 {len(results['ids'])} 个块")
            return len(results['ids'])
        return 0