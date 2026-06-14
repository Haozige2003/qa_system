import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from peft import PeftModel

MODEL_PATH = r"D:\test\pythonProject\qa_system\ner_lora_output\lora_adapter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 强制离线加载 BERT 模型（避免联网检查）
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese", local_files_only=True)
base_model = BertForTokenClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=28,
    local_files_only=True
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.to(device)
model.eval()

# 标签映射（根据训练的 label_list 生成）
id2label = {
    0: 'B-CONT', 1: 'B-EDU', 2: 'B-LOC', 3: 'B-NAME', 4: 'B-ORG', 5: 'B-PRO', 6: 'B-RACE', 7: 'B-TITLE',
    8: 'E-CONT', 9: 'E-EDU', 10: 'E-LOC', 11: 'E-NAME', 12: 'E-ORG', 13: 'E-PRO', 14: 'E-RACE', 15: 'E-TITLE',
    16: 'M-CONT', 17: 'M-EDU', 18: 'M-LOC', 19: 'M-NAME', 20: 'M-ORG', 21: 'M-PRO', 22: 'M-RACE', 23: 'M-TITLE',
    24: 'O', 25: 'S-NAME', 26: 'S-ORG', 27: 'S-RACE'
}

def extract_entities(text):
    """从文本中抽取实体，返回列表 [{'type': 'NAME', 'text': '张三'}, ...]"""
    if not text.strip():
        return []
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().numpy())
    labels = [id2label.get(p, "O") for p in preds]

    entities = []
    current = None
    for token, lbl in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        clean_token = token.replace("##", "")
        if lbl.startswith("S-"):
            entities.append({"type": lbl[2:], "text": clean_token})
            current = None
        elif lbl.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": lbl[2:], "text": clean_token}
        elif lbl.startswith(("I-", "M-")) and current and current["type"] == lbl[2:]:
            current["text"] += clean_token
        elif lbl.startswith("E-") and current and current["type"] == lbl[2:]:
            current["text"] += clean_token
            entities.append(current)
            current = None
        elif lbl == "O":
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities