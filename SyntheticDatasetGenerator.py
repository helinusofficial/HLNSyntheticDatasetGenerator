from llama_cpp import Llama
from datetime import datetime
import json
import pandas as pd


class PersianConversationGenerator:
    def __init__(self,logger, config):
        self.config = config
        self.logger=logger
        self.model_path = self.config.model_path
        self.output_file = self.config.output_file

        self.topics = self.config.topics
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

    def generate_conversation(self, conversation_index, total_conversations, max_turns=None, max_tokens=None,
                              temperature=None):
        if max_turns is None:
            max_turns = self.config.max_turns
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature
        max_messages = max_turns * 2

        topic = self.get_next_topic()
        self.logger.info(f"Conversation [{conversation_index}/{total_conversations}] | Topic: {topic}")
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": self.config.conversation_prompt.format(topic=topic, max_turns=max_turns,
                                                                               max_messages=max_messages)}
        ]
        start_time = datetime.now()
        output = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature,
                                                 stream=True)
        response = ""
        for chunk in output:
            content = chunk["choices"][0]["delta"].get("content", "")
            if content:
                response += content
                if self.config.Show_Generated_Output:
                    self.logger.info(content, end="", flush=True)
        elapsed = datetime.now() - start_time
        self.logger.info(f"\nGeneration time: {elapsed.seconds // 60:02d}:{elapsed.seconds % 60:02d}")
        return topic, response

    def generate_dataset(self):
        if self.config.max_tokens is None:
            max_tokens = self.config.max_tokens
        if self.config.temperature is None:
            temperature = self.config.temperature
        dataset = []
        self.logger.info("=" * 60)
        self.logger.info(f"Generating {self.config.num_conversations} conversations")
        self.logger.info(f"Output: {self.output_file}")
        self.logger.info("=" * 60)

        for i in range(self.config.num_conversations):
            try:
                topic, response = self.generate_conversation(conversation_index=i + 1,
                                                             total_conversations=self.config.num_conversations,
                                                             max_turns=self.config.max_turns, max_tokens=self.config.max_tokens,
                                                             temperature=self.config.temperature)
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

                dataset.append({
                    "id": i + 1,
                    "topic": topic,
                    "messages": json.dumps(conversation["messages"], ensure_ascii=False)
                })

                self.logger.info("Conversation saved successfully.")
            except json.JSONDecodeError:
                self.logger.info("Error: Model returned invalid JSON.")
                self.logger.info("Conversation skipped.")
            except Exception as e:
                self.logger.info(f"Error: {e}")
                self.logger.info("Conversation skipped.")

        df = pd.DataFrame(dataset)

        df.to_parquet(self.output_file, index=False)

        log_text = "\n" + "=" * 60 + "\n"
        log_text += "DATASET COMPLETED\n"
        log_text += "=" * 60 + "\n"
        log_text += f"Generated: {len(df)} conversations\n"
        log_text += f"File: {self.output_file}\n"
        log_text += "=" * 60
        self.logger.info(log_text)

        return df