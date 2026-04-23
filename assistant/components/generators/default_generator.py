from assistant.core.generator import Generator

class DefaultGenerator(Generator):
    def __init__(self):
        pass

    def __call__(self, input_data) -> str:
        return str(input_data)
