import torch

class TestHFModelGenerator:
    class TestGeneration:
        def test_successful_generation(self, generator_instance):
            prompt = "prompt"
            result = generator_instance(prompt, max_new_tokens=50, temperature=0.5)

            assert isinstance(result, str)
            assert len(result) > 0
            assert result == "generated answer"

        def test_deterministic_result(self, generator_instance):
            prompt = "prompt"
            torch.manual_seed(42)

            result1 = generator_instance(prompt, max_new_tokens=50, temperature=0.0)
            
            torch.manual_seed(42)
            result2 = generator_instance(prompt, max_new_tokens=50, temperature=0.0)

            assert result1 == result2
