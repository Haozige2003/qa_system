from sentence_transformers import SentenceTransformer
import chromadb
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from peft import PeftModel


class RAGRetriever:
    def __init__(self, embed_model="shibing624/text2vec-base-chinese", persist_dir="./chroma_db"):
        # 硬件检测
        if torch.cuda.is_available():
            device = 'cuda'
            print("使用 GPU 进行向量编码")
        else:
            device = 'cpu'
            print("使用 CPU 进行向量编码")

        # 向量模型
        self.embedder = SentenceTransformer(embed_model, device=device)

        # LoRA NER 模型
        self.ner_model = None
        self.ner_tokenizer = None
        try:
            base_model = AutoModelForTokenClassification.from_pretrained("bert-base-chinese", num_labels=28)
            self.ner_model = PeftModel.from_pretrained(base_model, "./ner_lora_output/lora_adapter")
            self.ner_tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            self.ner_model.eval()
            if torch.cuda.is_available():
                self.ner_model.to('cuda')
            print("LoRA-NER 模型加载成功（仅用于人名提取兜底）")
        except Exception as e:
            print(f"LoRA-NER 模型加载失败，将完全依赖正则提取人名: {e}")
            self.ner_model = None
            self.ner_tokenizer = None

        # 向量数据库
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("doc_chunks")
        print(f"知识库已加载，当前块数: {self.collection.count()}")

    #  入库
    def add_document(self, document, doc_id="doc", chunk_size=200, overlap=100, person_name=None):
        if not document or len(document.strip()) == 0:
            print(f"文档 {doc_id} 为空，跳过")
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
        print(f"文档 '{doc_id}' 已入库，新增 {len(chunks)} 个块")
        if person_name:
            print(f"关联人名: {person_name}")
        return len(chunks)

    #  检索
    def retrieve(self, question, top_k=4, person_name=None):
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

    # LoRA 推理
    def _extract_person_name_with_lora(self, input_text):
        """给一段文本，用 LoRA 模型推理，返回第一个完整人名，否则返回 None"""
        if self.ner_model is None or self.ner_tokenizer is None:
            return None

        try:
            inputs = self.ner_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

            # 从模型读取标签映射，读不到就用默认 7 类（兜底）
            id2label = self.ner_model.config.id2label
            if not id2label:
                id2label = {
                    0: 'O', 1: 'B-PER', 2: 'I-PER',
                    3: 'B-ORG', 4: 'I-ORG',
                    5: 'B-LOC', 6: 'I-LOC'
                }

            person_name = ""
            for token, label_id in zip(tokens, predictions):
                label = id2label.get(label_id.item(), 'O')
                if label == 'B-PER':
                    person_name = token
                elif label == 'I-PER' and person_name:
                    person_name += token
                elif label == 'O' and person_name:
                    if len(person_name) >= 2:
                        return person_name
                    else:
                        person_name = ""
            if person_name and len(person_name) >= 2:
                return person_name
        except Exception:
            pass
        return None

    # 提取人名（正文）
    def extract_person_name_from_text(self, text):
        # 正则优先
        patterns = [
            r'(?:姓名|名字)[：:]\s*([\u4e00-\u9fa5]{2,4})',
            r'^([\u4e00-\u9fa5]{2,4})\s*[,，、]'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)

        # LoRA 兜底
        return self._extract_person_name_with_lora(text[:256])

    # 提取人名（问题）
    def extract_person_name_from_question(self, question):
        # 正则优先
        pattern = r'([\u4e00-\u9fa5]{2,4})(?:的|在|是|有|和)'
        matches = re.findall(pattern, question)
        if matches:
            return matches[0]

        # LoRA 兜底（问题太短，不用切片）
        return self._extract_person_name_with_lora(question)

    # 辅助工具
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
        print("知识库已清空")

    def delete_document(self, doc_id):
        results = self.collection.get(where={"doc_id": doc_id})
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"文档 '{doc_id}' 已删除，移除 {len(results['ids'])} 个块")
            return len(results['ids'])
        return 0
