# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import params
from typing import Optional, Literal
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from system_components import Logger
from embedding_modules.llamaindex.base_embedding_module import StandardlizedEmbeddingModule
import os



class EmbeddingModule(StandardlizedEmbeddingModule):
    def __init__(self,
                 model_name: str = "default",
                 service_name: Literal["FastEmbed","HuggingFace"] = "FastEmbed",
                 batch_size: int = 10,
                 max_length: int = 512,
                 cached_folder = params.embedding_model_folder,
                 num_threads :Optional[int] = None):
        """Define embedding service with specified params"""

        super().__init__(batch_size = batch_size,max_length= max_length)
        # Define variable
        self._cached_folder = cached_folder


        # Create folder
        os.makedirs(cached_folder,exist_ok=True)

        service_unsupported_msg = "HuggingFace temporally turned off"
        # Hugging Face
        if service_name == "HuggingFace":
            # self._embedding_model = HuggingFaceEmbedding(cache_folder=self._embedding_model_folder,embed_batch_size=self.batch_size)
            Logger.exception(service_unsupported_msg)
            raise Exception(service_unsupported_msg)
        # Fast Embed
        elif service_name == "FastEmbed":
            self._model_name = "BAAI/ bge-small-en-v1.5" if model_name == "default" else model_name
            self._embedding_model = FastEmbedEmbedding(model_name = self._model_name,
                                                       max_length = self.max_length,
                                                       cache_dir = self._cached_folder,
                                                       threads = num_threads)
        else:
            service_unsupported_msg = f"Service {service_name} is not supported!"
            Logger.exception(service_unsupported_msg)
            raise Exception(service_unsupported_msg)

        # Logging status
        init_message = f"Initiate {service_name} with model: {self._embedding_model.model_name}, batch size {self.batch_size}"
        Logger.info(init_message)


