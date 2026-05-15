import pytest

class TestChunker:
    class TestSplittingLogic:
        def test_correct_splitting_with_size_and_overlap(self, chunker):
            text = "X" * 200
            chunks = chunker.split_text(text)

            assert len(chunks) > 1

            for idx, chunk in enumerate(chunks):
                assert len(chunk) <= 50

            for idx in range(len(chunks) - 1):
                expected_suffix = chunks[idx][-10:]
                assert chunks[idx + 1].startswith(expected_suffix)

    class TestEdgeCases:
        @pytest.mark.parametrize("input_text, case_name", [
            ("", "empty"),
            ("     ", "spaces"),
            ("\n\n\n", "newlines"),
            (" \t \n ", "whitespaces"),
        ])
        def test_empty_and_whitespace_handling(self, chunker, input_text, case_name):
            result = chunker.split_text(input_text)
            assert isinstance(result, list)

    class TestInterfaceContract:
        def test_strict_return_format(self, chunker):
            text = "Long long long long long long long long long long long long long long long text."
            result = chunker.split_text(text)

            assert isinstance(result, list)
            assert all(isinstance(item, str) for item in result)
            assert len(result) > 0

            desc = chunker.describe()
            assert isinstance(desc, str)
            assert "50" in desc
