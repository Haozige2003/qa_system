import gradio as gr
import re
import os
import asyncio
import time
from openai import OpenAI
from ner_utils import extract_entities
from kreuzberg import extract_file

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

API_ENDPOINTS = [
    {"name": "DeepSeek官方", "base_url": "https://api.deepseek.com/v1", "api_key": DEEPSEEK_API_KEY, "model": "deepseek-chat"},
    {"name": "DeepSeek华东", "base_url": "https://api-east.deepseek.com/v1", "api_key": DEEPSEEK_API_KEY, "model": "deepseek-chat"},
    {"name": "阿里云百炼", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": DASHSCOPE_API_KEY, "model": "deepseek-r1"}
]
VALID_ENDPOINTS = [ep for ep in API_ENDPOINTS if ep["api_key"]]

class Client:
    def __init__(self, endpoints, timeout=60, retries=2):
        self.endpoints = endpoints
        self.timeout = timeout
        self.retries = retries
    def request(self, prompt):
        for ep in self.endpoints:
            for attempt in range(self.retries + 1):
                try:
                    c = OpenAI(api_key=ep["api_key"], base_url=ep["base_url"], timeout=self.timeout)
                    resp = c.chat.completions.create(
                        model=ep["model"],
                        messages=[{"role": "system", "content": "根据事实回答。"}, {"role": "user", "content": prompt}],
                        temperature=0.3, max_tokens=500
                    )
                    print(f" {ep['name']}")
                    return resp.choices[0].message.content.strip(), ep["name"]
                except Exception as e:
                    print(f" {ep['name']}: {str(e)[:80]}")
                    time.sleep(2 ** attempt)
        raise Exception("所有节点失败")
client = Client(VALID_ENDPOINTS)

def fallback_answer(question, entities):
    if not entities:
        return "未抽到实体"
    q = question.lower()
    if any(k in q for k in ["公司","企业","工作"]):
        orgs = list(set(e['text'] for e in entities if e['type']=='ORG'))
        return f"组织：{', '.join(orgs)}" if orgs else "无"
    if any(k in q for k in ["职位","职务"]):
        titles = list(set(e['text'] for e in entities if e['type']=='TITLE'))
        return f"职位：{', '.join(titles)}" if titles else "无"
    if any(k in q for k in ["姓名","是谁"]):
        names = list(set(e['text'] for e in entities if e['type']=='NAME'))
        return f"人物：{', '.join(names)}" if names else "无"
    return "无法回答（大模型不可用）"

def chunk_doc(document, max_len=500):
    """仅用于实体抽取的分块，不改变问答用的全文"""
    sents = re.split(r'([。！？；])', document)
    chunks, cur = [], ""
    for i in range(0, len(sents), 2):
        sent = sents[i] + (sents[i+1] if i+1<len(sents) else "")
        sent = sent.strip()
        if not sent: continue
        if len(cur)+len(sent) <= max_len:
            cur += sent
        else:
            if cur: chunks.append(cur)
            cur = sent
    if cur: chunks.append(cur)
    return chunks

def answer_question_long(question, document):
    if not document.strip() or not question.strip():
        return "请提供文档和问题", []
    # 分块抽实体（用于展示）
    chunks = chunk_doc(document)
    all_entities, seen = [], set()
    for ch in chunks:
        for e in extract_entities(ch):
            key = (e['type'], e['text'])
            if key not in seen:
                seen.add(key)
                all_entities.append(e)
    # 全文截断（DeepSeek 支持长文本，但控制 token 用 6000 字符足够）
    MAX_CHARS = 6000
    if len(document) > MAX_CHARS:
        full_text = document[:MAX_CHARS] + "\n...(文档过长，已截断)"
    else:
        full_text = document
    entities_str = "\n".join(f"- {e['type']}: {e['text']}" for e in all_entities) if all_entities else "无"
    prompt = f"""基于以下【文档】和【抽取的实体】回答问题。如果信息不足，请先列出已有相关事实（例如从抽取的实体中提取），然后说明哪些信息缺失。

【文档】
{full_text}

【实体】
{entities_str}

问题：{question}
回答："""
    try:
        ans, _ = client.request(prompt)
        return ans, all_entities
    except Exception as e:
        return fallback_answer(question, all_entities), all_entities

def read_file(file):
    if not file:
        return ""
    ext = os.path.splitext(file.name)[1].lower()
    try:
        if ext in ['.txt','.md','.csv','.json','.py','.html','.xml']:
            with open(file.name, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(extract_file(file.name))
            loop.close()
            return res.content if res.content else "解析失败"
    except:
        return "文件读取失败"

def process(question, doc):
    ans, ents = answer_question_long(question, doc)
    ent_disp = "\n".join(f"- {e['type']}: {e['text']}" for e in ents) if ents else "无实体"
    return ans, ent_disp

with gr.Blocks(title="智能问答系统") as demo:
    gr.Markdown("# 信息抽取+智能问答系统")
    with gr.Row():
        with gr.Column():
            file = gr.File(label="上传文档", file_types=None)
            doc_box = gr.Textbox(label="文档内容", lines=12)
            q_box = gr.Textbox(label="问题", lines=2)
            btn = gr.Button("生成答案")
        with gr.Column():
            ans_box = gr.Textbox(label="答案", lines=6)
            ent_box = gr.Textbox(label="抽取实体", lines=10)
    file.change(fn=read_file, inputs=file, outputs=doc_box)
    btn.click(fn=process, inputs=[q_box, doc_box], outputs=[ans_box, ent_box])

if __name__ == "__main__":
    demo.launch()
