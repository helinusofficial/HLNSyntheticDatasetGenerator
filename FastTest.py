from llama_cpp import Llama
from datetime import datetime

MODEL_PATH = r"D:\TFSProjects\HelinusCollections\AI\AllModels\Gemma3_Models\gemma-3-4b-it-Q4_K_M.gguf"

print("Loading model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_batch=512,
    n_gpu_layers=-1,
    verbose=False
)

print("Model loaded successfully!")

messages = [
    {
        "role": "system",
        "content": "شما یک تولیدکننده دیتاست مکالمات طبیعی فارسی هستید."
    },
    {
        "role": "user",
        "content": """
/no_think

یک مکالمه طبیعی و واقعی بین دو نفر به زبان فارسی تولید کن.

موضوع مکالمه یکی از موضوعات روزمره باشد.

قوانین:
- مکالمه کاملاً مرتبط با موضوع باشد.
- هر دو نفر مانند انسان واقعی و طبیعی صحبت کنند.
- سوال و پاسخ‌ها واضح و منطقی باشند.
- مکالمه چندنوبتی باشد.
- از جملات بی‌معنی یا ساختگی استفاده نکن.
- فقط JSON معتبر خروجی بده و هیچ متن اضافی ننویس.

فرمت:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""
    }
]
print("Generating response...\n")

start_time = datetime.now()

output = llm.create_chat_completion(
    messages=messages,
    max_tokens=3000,
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