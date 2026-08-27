from llama_cpp import Llama
from datetime import datetime

MODEL_PATH = r"D:\Downloads\qwen2.5-3b-instruct-q4_k_m.gguf"

print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_batch=512,
    n_gpu_layers=-1,
    verbose=True
)

print("Model loaded successfully!")

messages = [
    {
        "role": "system",
        "content": "شما یک تولیدکننده دیتاست آموزشی با کیفیت برای مدل‌های زبانی بزرگ هستید."
    },
    {
        "role": "user",
        "content": """
/no_think

یک نمونه مکالمه چند نوبتی به زبان فارسی تولید کن.

موضوع:
پزشکی

قوانین:
- مکالمه طبیعی و شبیه گفتگوی واقعی انسان باشد.
- کاربر یک سوال واضح مطرح کند.
- دستیار پاسخ کامل، دقیق و آموزشی ارائه دهد.
- اطلاعات ساختگی تولید نکن.
- خروجی فقط JSON معتبر باشد.
- هیچ متن اضافی قبل یا بعد از JSON ننویس.

فرمت خروجی:

{
  "messages": [
    {
      "role": "user",
      "content": "سوال کاربر"
    },
    {
      "role": "assistant",
      "content": "پاسخ دستیار"
    }
  ]
}

اکنون یک نمونه تولید کن.
"""
    }
]

print("Generating response...\n")

start_time = datetime.now()

output = llm.create_chat_completion(
    messages=messages,
    max_tokens=500,
    temperature=0.75,
    stream=True
)

print("Response:\n")

response = ""

for chunk in output:
    content = chunk["choices"][0]["delta"].get("content", "")
    if content:
        response += content
        print(content, end="", flush=True)

end_time = datetime.now()
elapsed = end_time - start_time

print("\n\nDone.")

print("Final response length:", len(response))
print(
    "Generation time:",
    f"{elapsed.seconds // 3600:02d}:"
    f"{(elapsed.seconds % 3600) // 60:02d}:"
    f"{elapsed.seconds % 60:02d}"
)