from flask import Flask, render_template, request
# -*- coding:utf-8 -*-

import logging
import sys, json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
app = Flask(__name__)


model_path = "D:\Python\Pingan\LLM\model\Qwen2-1.5B-Instruct"
logging.info(': start running ...')
# api调用获取
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("model load over !")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.route('/', methods=['POST'])
def post_example():
    data = request.get_json() # 获取POST请求中的data参数
    try:
        # 自己模块 修改
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    except Exception as e:
        logging.error("Error content: %s \t ", text, '--e---', e)
        response = []
    return response


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8002, debug=True)
