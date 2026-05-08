import pytest
from assistant.components.embedders.hf_model_embedder_factory import HFModelEmbedderFactory

class TestHFModelEmbedderFactory:
    def test_successful_query_embedding(self, embedder_factory, mock_huggingface_embeddings):
        embedder = embedder_factory.create_embedder()
        user_query = "???"
        
        vector = embedder.embed_query(user_query)
        
        assert vector is not None
        assert isinstance(vector, (list, tuple))
        assert len(vector) > 0
        
        mock_huggingface_embeddings.return_value.embed_query.assert_called_once_with(user_query)

    def test_deterministic_embedding(self, embedder_factory, mock_huggingface_embeddings):
        embedder = embedder_factory.create_embedder()
        text = "TEXT"
        
        vec1 = embedder.embed_query(text)
        vec2 = embedder.embed_query(text)
        
        list1 = list(vec1) if hasattr(vec1, "__iter__") else [vec1]
        list2 = list(vec2) if hasattr(vec2, "__iter__") else [vec2]
        
        assert list1 == list2
