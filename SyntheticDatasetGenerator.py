import os
from llama_cpp import Llama
from datetime import datetime
import json
import pandas as pd
import random

class PersianConversationGenerator:
    def __init__(self,logger, config):
        self.config = config
        self.logger=logger
        self.model_path = self.config.model_path
        self.output_file = self.config.output_file

        self.topics = self.config.topics.copy()
        random.shuffle(self.topics)
        self.topic_index = 0

        self.topic_index = 0
        self.logger.info("Loading model...")
        self.llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            n_batch=self.config.n_batch,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=self.config.verbose
        )
        self.logger.info("Model loaded successfully!")

    def get_next_topic(self):
        topic = self.topics[self.topic_index]
        self.topic_index += 1
        if self.topic_index >= len(self.topics):
            self.topic_index = 0

        return topic

    def get_random_variation(self):
        return {
            "conversation_style": random.choice([
                "صمیمی و روزمره",
                "رسمی و مؤدبانه",
                "نیمه‌رسمی",
                "آرام و دوستانه",
                "کمی جدی",
                "کنجکاوانه",
                "مستقیم و کوتاه",
                "توضیحی و مفصل"
            ]),

            "opening_style": random.choice([
                "مکالمه با یک سؤال شروع شود.",
                "مکالمه با بیان یک موضوع یا مشکل شروع شود.",
                "مکالمه با یک واکنش به یک اتفاق شروع شود.",
                "مکالمه با یک توضیح کوتاه شروع شود.",
                "مکالمه با یک درخواست شروع شود.",
                "مکالمه با بیان یک تجربه شروع شود.",
                "مکالمه با یک جمله طبیعی و غیرمستقیم شروع شود."
            ]),

            "interaction_style": random.choice([
                "افراد بیشتر سؤال و جواب داشته باشند.",
                "یکی از افراد بیشتر توضیح دهد.",
                "هر دو نفر تقریباً به یک اندازه صحبت کنند.",
                "در مکالمه یک سوءتفاهم کوچک ایجاد و سپس برطرف شود.",
                "یکی از افراد نظر متفاوتی داشته باشد.",
                "مکالمه به‌صورت طبیعی از یک نکته به نکته دیگری مرتبط با موضوع برسد.",
                "یکی از افراد در ابتدا اطلاعات کاملی ندهد و طرف مقابل با سؤال‌های مناسب موضوع را روشن کند."
            ]),

            "response_style": random.choice([
                "پاسخ‌ها کوتاه و طبیعی باشند.",
                "پاسخ‌ها ترکیبی از جمله‌های کوتاه و متوسط باشند.",
                "گاهی پاسخ‌ها توضیحی و چندجمله‌ای باشند.",
                "طول پاسخ‌ها متغیر باشد و همه پاسخ‌ها یک اندازه نباشند."
            ])
        }

    def generate_conversation(self, conversation_index, total_conversations, max_tokens=None,
                          temperature=None):

        if max_tokens is None:
            max_tokens = self.config.max_tokens

        if temperature is None:
            temperature = self.config.temperature

        max_turns = random.randint(
            self.config.min_turns,
            self.config.max_turns
        )

        max_messages = max_turns * 2

        topic = self.get_next_topic()
        self.logger.info(f"Conversation [{conversation_index}/{total_conversations}] | Topic: {topic}")
        variation = self.get_random_variation()

        messages = [
            {
                "role": "system",
                "content": self.config.system_prompt
            },
            {
                "role": "user",
                "content": self.config.conversation_prompt.format(
                    topic=topic,
                    max_turns=max_turns,
                    max_messages=max_messages,
                    conversation_style=variation["conversation_style"],
                    opening_style=variation["opening_style"],
                    interaction_style=variation["interaction_style"],
                    response_style=variation["response_style"]
                )
            }
        ]

        start_time = datetime.now()
        output = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stream=True
        )
        response = ""
        for chunk in output:
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                response += content

        if self.config.Show_Generated_Output:
            self.logger.info(response)

        elapsed = datetime.now() - start_time
        total_seconds = int(elapsed.total_seconds())

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        self.logger.info(
            f"Conversation [{conversation_index}/{total_conversations}] | "
            f"Generation time: {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        return topic, response

    def generate_dataset(self):
        dataset = []
        if os.path.exists(self.output_file):
            try:
                existing_df = pd.read_parquet(self.output_file)
                dataset = existing_df.to_dict("records")
                self.logger.info(f"Resuming from {len(dataset)} conversations")
            except Exception:
                dataset = []

        self.logger.info("=" * 60)
        self.logger.info(f"Generating {self.config.num_conversations} conversations")
        self.logger.info(f"Output: {self.output_file}")
        self.logger.info("=" * 60)

        for i in range(len(dataset), self.config.num_conversations):
            try:
                topic, response = self.generate_conversation(
                    conversation_index=i + 1,
                    total_conversations=self.config.num_conversations,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                clean_response = response.strip()

                if clean_response.startswith("```"):
                    clean_response = clean_response.replace("```json", "")
                    clean_response = clean_response.replace("```", "")
                    clean_response = clean_response.strip()

                conversation = json.loads(clean_response)

                if "messages" not in conversation:
                    self.logger.info("Invalid response: messages not found")
                    continue

                if not isinstance(conversation["messages"], list):
                    self.logger.info("Invalid response: messages is not a list")
                    continue

                conversation["messages"] = conversation["messages"][:32]

                dataset.append(
                    {
                        "id": i + 1,
                        "topic": topic,
                        "messages": json.dumps(
                            conversation["messages"],
                            ensure_ascii=False
                        )
                    }
                )
                if (i + 1) % self.config.save_interval == 0 or (i + 1) == self.config.num_conversations:
                    df = pd.DataFrame(dataset)
                    df.to_parquet(self.config.output_temp_file, index=False)
                    os.replace(self.config.output_temp_file, self.output_file)
                    self.logger.info(f"Dataset saved: {len(dataset)} conversations")

            except json.JSONDecodeError:
                self.logger.info("Error: Model returned invalid JSON.")
                self.logger.info("Conversation skipped.")
            except Exception as e:
                self.logger.info(f"Error: {e}")
                self.logger.info("Conversation skipped.")

        df = pd.DataFrame(dataset)

        log_text = "\n" + "=" * 60 + "\n"
        log_text += "DATASET COMPLETED\n"
        log_text += "=" * 60 + "\n"
        log_text += f"Generated: {len(df)} conversations\n"
        log_text += f"File: {self.output_file}\n"
        log_text += "=" * 60
        self.logger.info(log_text)

        return df