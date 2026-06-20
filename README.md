# 多格式文档问答系统（RAG）

基于 RAG（Retrieval-Augmented Generation）架构的多格式文档问答系统。支持上传 PDF、Word、Excel、PPT、TXT 等多格式文档，自动构建本地知识库，通过语义检索 + 大模型生成实现精准问答。

**项目特点**：
- 多格式文档解析 + **四层解析兜底机制**，处理真实办公文档
- **跨文档归属隔离检索**，避免多份文档内容混淆
- 多 API 端点调度（DeepSeek + 阿里云通义），高可用设计
- 基于 Gradio 的可视化界面，上传即用


## 项目结构

```
├── app_pro.py              # 主程序（Gradio 界面 + API 调度）
├── rag_retriever.py        # 检索核心类（ChromaDB + Embedding）
├── chroma_db/              # 向量数据库持久化目录（运行后自动生成）
└── .kreuzberg/             # 文档解析库缓存目录（运行后自动生成）
```


## 环境依赖

```
Python 3.8+
torch
transformers
sentence-transformers
chromadb
gradio
openai
kreuzberg
pypdfium2
PyPDF2
python-docx
openpyxl
python-pptx
```

安装命令：

```bash
pip install torch transformers sentence-transformers chromadb gradio openai kreuzberg pypdfium2 PyPDF2 python-docx openpyxl python-pptx
```


## 快速开始

### 1. 配置 API Key

在 `rag_app.py` 中设置你的 API Key：

```python
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-你的阿里云Key")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-你的DeepSeekKey")
```

系统会**自动遍历多个端点**，单个 API 失败时自动切换。

### 2. 启动服务

```bash
python rag_app.py
```

终端会输出一个本地地址（如 `http://127.0.0.1:7860`），在浏览器中打开即可访问。

### 3. 使用流程

1. **上传文档**：点击上传按钮，可选择多个文件（PDF/Word/Excel/PPT/TXT）
2. **点击“入库”**：系统自动提取文本、分块、生成向量索引
3. **输入问题**：在问题框输入问题，点击“回答”
4. **查看结果**：系统返回答案并展示检索到的相关片段


## 支持的文档格式

| 格式 | 解析方式 |
|------|----------|
| `.txt` / `.md` / `.csv` / `.json` / `.py` | 直接读取（UTF-8/GBK 自动适配） |
| `.pdf` | 四层兜底：Kreuzberg → pypdfium2 → PyPDF2 |
| `.docx` | Kreuzberg → python-docx |
| `.pptx` | Kreuzberg → python-pptx |
| `.xlsx` | Kreuzberg → openpyxl |

> 解析失败时自动切换下一层方案，**单文件处理时间控制在 2 秒以内**。


## 核心设计

### 1. 四层解析兜底机制

针对不同格式文档的编码差异，设计了分层解析策略：

```
用户上传文件
     ↓
Kreuzberg 通用解析器（首选）
     ↓ 失败
格式专用解析器（pypdfium2 / python-docx / openpyxl / python-pptx）
     ↓ 失败
基础解析器（PyPDF2 等）
     ↓ 失败
返回空字符串（记录失败日志）
```

这一设计显著提升了真实办公场景下的文档解析成功率。

### 2. 跨文档归属隔离检索

**场景痛点**：上传多份简历或合同时，用户提问“他的毕业院校是哪里？”系统不知道“他”是谁，可能从错误文档中检索。

**解决方案**：
- 入库时通过 `doc_id` 元数据标记每个文本块的来源文档
- 支持自动提取**人名**作为元数据标签（正则匹配“姓名：XXX”或开头中文名）
- 提问时**先按人名过滤检索**，召回不足时自动回退全库检索

**适用场景**：
- 多份简历批量筛选
- 合同版本管理（v1 / v2 / v3）
- 多作者稿件区分

### 3. 高可用 LLM 调用层

```python
API_ENDPOINTS = [
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", ...},
    {"name": "阿里云", "base_url": "https://dashscope.aliyuncs.com/...", ...}
]
```

- 遍历多个 API 端点，单个失败自动切换
- 单端点失败时指数退避重试（2 秒 → 4 秒）
- Temperature = 0.3 + “仅使用检索片段回答”约束，控制模型幻觉


## 检索策略详解

| 步骤 | 说明 |
|------|------|
| 文本分块 | `chunk_size=500, overlap=150`，保持上下文连贯性 |
| 向量化 | 使用 `shibing624/text2vec-base-chinese` 中文 Embedding 模型 |
| 索引存储 | ChromaDB 持久化，重启数据不丢失 |
| 语义检索 | 基于余弦相似度召回 Top-K 相关块 |
| 元数据过滤 | 支持按 `doc_id` / `person_name` 等字段过滤 |
| 回退策略 | 按人名过滤召回 < 2 条时，自动全库检索补充 |


## 交互界面

基于 Gradio 搭建，包含以下模块：

| 模块 | 功能 |
|------|------|
| 文件上传 | 支持多文件批量上传 |
| 入库按钮 | 解析、分块、向量化、入库一站式完成 |
| 状态栏 | 显示入库进度、文档块数、错误日志 |
| 问题输入 | 用户提问 |
| 答案输出 | 显示 LLM 生成的回答 |
| 检索片段展示 | 显示被召回的原始文本块，便于溯源 |


## 功能自测

- [x] PDF/Word/Excel/PPT/TXT 解析入库
- [x] 同名文档自动覆盖更新
- [x] 按人名过滤检索
- [x] 召回不足时自动回退全库
- [x] 多 API 端点自动切换
- [x] Gradio 界面交互


## 未来优化方向

- [ ] 支持更多 Embedding 模型（如 BGE、OpenAI Embedding）
- [ ] 增加文档摘要生成功能
- [ ] 支持混合检索（关键词 + 向量）
- [ ] 添加对话历史记忆功能


##  作者

倪皓轩 - [GitHub](https://github.com/Haozige2003)
