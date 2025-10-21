import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name="KORMo-Team/KORMo-10B-sft"
tokenizer=AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model=AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

def generate_text(prompt, max_length=5000):
    messages=[
        {"role":"user","content":prompt}
    ]

    chat_prompt=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs=tokenizer(chat_prompt,return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output_ids=model.generate(
            **inputs,
            max_new_tokens=max_length
        )

    response=tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_toknes=True)
    return response

if __name__=="__main__":
    prompt="오늘 저녁에 뭐먹지? 추천해주세요."
    result=generate_text(prompt)
    print(f"입력: {prompt}")
    print(f"출력: {result}")