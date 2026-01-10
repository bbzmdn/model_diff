import os
from dotenv import load_dotenv

load_dotenv()

base_model_id = os.getenv("BASE_MODEL_ID")
chat_model_id = os.getenv("CHAT_MODEL_ID")
hidden_size = int(os.getenv("HIDDEN_SIZE"))
target_layer = int(os.getenv("TARGET_LAYER"))

behavioral_prompts_filepath = os.getenv("BEHAVIORAL_PROMPTS_FILEPATH")
steering_vectors_filepath = os.getenv("STEERING_VECTORS_FILEPATH")
diffsae_checkpoint_path = os.getenv("DIFFSAE_CHECKPOINT_PATH")
