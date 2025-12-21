from core.generator import Generator

class DefaultGenerator(Generator):
    def __init__(self):
        pass

    def __call__(self, input_data: str) -> str:
        return input_data.to_string()
