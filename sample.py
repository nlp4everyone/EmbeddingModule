from embedding_modules.llamaindex.intergrations import IntergrationsEmbeddingModule

# Define object
embbedding_module = IntergrationsEmbeddingModule(service_name = "GEMINI")
embedding_model = embbedding_module.get_embedding_model()
embedding_text = embedding_model.get_text_embedding("Hello")
print(embedding_text)