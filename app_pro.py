import gradio as gr
import os
import asyncio
import time
import re
from openai import OpenAI
from rag_retriever import RAGRetriever
from kreuzberg import extract_file

retriever = RAGRetriever(persist_dir="./chroma_db")

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-xxx")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-yyy")

API_ENDPOINTS = [
    {"name": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "api_key": DEEPSEEK_API_KEY, "model": "deepseek-chat"},
    {"name": "阿里云", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": DASHSCOPE_API_KEY, "model": "deepseek-r1"}
]

def call_llm(prompt):
    for ep in API_ENDPOINTS:
        if not ep["api_key"]:
            continue
        for attempt in range(2):
            try:
                c = OpenAI(api_key=ep["api_key"], base_url=ep["base_url"], timeout=60)
                resp = c.chat.completions.create(
                    model=ep["model"],
                    messages=[{"role": "system", "content": "根据事实回答"}, {"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=500
                )
                return resp.choices[0].message.content.strip()
            except:
                time.sleep(2 ** attempt)
    return "API调用失败"

def read_file(file):
    if not file:
        return ""
    ext = os.path.splitext(file.name)[1].lower()
    if ext in ['.txt', '.md', '.csv', '.json', '.py']:
        for enc in ['utf-8', 'gbk']:
            try:
                with open(file.name, 'r', encoding=enc) as f:
                    return re.sub(r'\s+', ' ', f.read()).strip()
            except:
                continue
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(extract_file(file.name))
        loop.close()
        if res.content and res.content.strip():
            return re.sub(r'\s+', ' ', res.content).strip()
    except:
        pass
    try:
        if ext == '.pdf':
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(file.name)
                text = "".join([page.get_text() for page in pdf])
                pdf.close()
                if text.strip():
                    return re.sub(r'\s+', ' ', text).strip()
            except:
                pass
            import PyPDF2
            with open(file.name, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join([p.extract_text() or "" for p in reader.pages])
                if text.strip():
                    return re.sub(r'\s+', ' ', text).strip()
        elif ext == '.docx':
            import docx
            doc = docx.Document(file.name)
            text = "".join([p.text for p in doc.paragraphs if p.text.strip()])
            if text.strip():
                return re.sub(r'\s+', ' ', text).strip()
        elif ext == '.pptx':
            from pptx import Presentation
            prs = Presentation(file.name)
            text = "".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
            if text.strip():
                return re.sub(r'\s+', ' ', text).strip()
        elif ext == '.xlsx':
            import openpyxl
            wb = openpyxl.load_workbook(file.name, read_only=True, data_only=True)
            rows = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join([str(c) for c in row if c])
                    if row_text:
                        rows.append(row_text)
            text = "".join(rows)
            if text.strip():
                return re.sub(r'\s+', ' ', text).strip()
    except:
        pass
    return ""

def add_files(files):
    if not files:
        return "未选择文件", retriever.count_documents()
    logs, ok = [], 0
    for f in files:
        content = read_file(f)
        if not content or len(content) < 20:
            logs.append(f" {os.path.basename(f.name)} 解析失败")
            continue
        name = retriever.extract_person_name_from_text(content)
        doc_id = os.path.splitext(os.path.basename(f.name))[0]
        if name:
            doc_id = f"{name}简历"
        retriever.delete_document(doc_id)
        cnt = retriever.add_document(content, doc_id=doc_id, person_name=name, chunk_size=500, overlap=150)
        ok += 1
        logs.append(f" {os.path.basename(f.name)} → {cnt}块" + (f" ({name})" if name else ""))
    total = retriever.count_documents()
    return f"成功 {ok} 个，总块数 {total}\n" + "\n".join(logs), total

def ask(question):
    if not question:
        return "请输入问题", ""
    name = retriever.extract_person_name_from_question(question)
    docs = []
    if name:
        docs = retriever.retrieve(question, person_name=name, top_k=4)
        # 如果结果少于2块，回退全库检索，扩大召回
        if len(docs) < 2:
            extra = retriever.retrieve(question, top_k=6)
            seen = set([d[0] for d in docs])
            for d in extra:
                if d[0] not in seen:
                    docs.append(d)
                    seen.add(d[0])
    else:
        docs = retriever.retrieve(question, top_k=6)
    if not docs:
        return "未找到相关信息", ""
    context = "\n\n".join([f"【{doc_id}】\n{text}" for doc_id, text in docs])
    ans = call_llm(f"根据以下片段回答，只使用提供的信息：\n{context}\n问题：{question}")
    return ans, context

def refresh():
    total = retriever.count_documents()
    ids = retriever.get_all_doc_ids()
    return f"知识库：{total}块，文档：{', '.join(ids) if ids else '空'}"

def clear():
    retriever.clear_all()
    return "已清空", 0

with gr.Blocks() as demo:
    gr.Markdown("# 文档问答（多格式）")
    with gr.Row():
        with gr.Column():
            files = gr.File(label="上传文件（可多选）", file_count="multiple")
            btn_up = gr.Button("入库")
            btn_clr = gr.Button("清空")
            btn_ref = gr.Button("刷新")
            status = gr.Textbox(label="状态", lines=6)
        with gr.Column():
            question = gr.Textbox(label="问题")
            btn_ask = gr.Button("回答")
            answer = gr.Textbox(label="答案", lines=6)
            context = gr.Textbox(label="检索片段", lines=8)
    btn_up.click(add_files, [files], [status, status])
    btn_clr.click(clear, None, [status, status])
    btn_ref.click(refresh, None, status)
    btn_ask.click(ask, [question], [answer, context])

if __name__ == "__main__":
    demo.launch()