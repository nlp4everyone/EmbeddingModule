from dotenv import load_dotenv
import os,json
load_dotenv()
cache_folder = "cache_folder"
embedding_model_folder = os.path.join(cache_folder,"embedding_model")

# Embedding model params
# Define params
embedding_service = "COHERE"
embedding_model_name = "embed-english-light-v3.0"
chat_service = "GEMINI"
chat_model = "models/gemini-pro"

# Model config
GROQ_KEY = os.getenv("GROQ_KEY") # 30 requests/min
OPENAI_KEY = os.getenv("OPENAI_KEY") # 5$ starter bundle
TOGETHER_KEY = os.getenv("TOGETHER_KEY") # 25$ starter bundle
GEMINI_KEY = os.getenv("GEMINI_KEY") # Free to use
VOYAGE_KEY = os.getenv("VOYAGE_KEY") # Free for 50M first tokens
NOMIC_KEY = os.getenv("NOMIC_KEY") # Free for 50M first tokens
COHERE_KEY = os.getenv("COHERE_KEY") # 1000 request per month (multilingual)
JINA_KEY = os.getenv("JINA_KEY") # Free for 50M first tokens
LLAMAPARSE_KEY = os.getenv("LLAMAPARSE_KEY") # https://cloud.llamaindex.ai/parse

# Load service
if not os.path.exists("config/llama_index_config.json"):
    raise Exception("Llama index config is not existed!")

# Load config
with open("config/llama_index_config.json",'r') as f:
    supported_services = json.load(f)

# Add key service
supported_services["GROQ"]["KEY"] = GROQ_KEY
supported_services["OPENAI"]["KEY"] = OPENAI_KEY # List model: https://platform.openai.com/docs/models/continuous-model-upgrades
supported_services["TOGETHER"]["KEY"] = TOGETHER_KEY  # List model: https://docs.together.ai/docs/inference-models
supported_services["GEMINI"]["KEY"] = GEMINI_KEY # List model: https://ai.google.dev/models/gemini
supported_services["QDRANT"]["KEY"] = ""  # Qdrant Embedding: https://qdrant.github.io/fastembed/examples/Supported_Models/
supported_services["VOYAGE"]["KEY"] = VOYAGE_KEY  # Embedding: https://docs.voyageai.com/docs/pricing
supported_services["NOMIC"]["KEY"] = NOMIC_KEY # Embedding: https://docs.nomic.ai/atlas/models/text-embedding
supported_services["COHERE"]["KEY"] = COHERE_KEY # Embedding: https://docs.nomic.ai/atlas/models/text-embedding
supported_services["JINA"]["KEY"] = JINA_KEY