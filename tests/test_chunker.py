import pytest

from app.services.chunker import Chunker


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestPlainText:
    def test_short_text_yields_single_chunk(self):
        # Regression: this used to loop forever whenever text was shorter than chunk_size.
        chunks = Chunker(chunk_size=100, chunk_overlap=10).chunk_text("hello world")
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].start_time is None

    def test_long_text_respects_size_and_overlap(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        text = words(400)
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        assert all(c.token_count <= 50 for c in chunks)
        assert [c.index for c in chunks] == list(range(len(chunks)))
        # consecutive chunks share exactly `chunk_overlap` tokens
        tokens = chunker.encoding.encode(text)
        assert chunks[1].text == chunker.encoding.decode(tokens[40:90])

    def test_covers_entire_text(self):
        chunker = Chunker(chunk_size=30, chunk_overlap=5)
        text = words(200)
        chunks = chunker.chunk_text(text)
        assert chunks[-1].text.endswith("w199")

    def test_english_translation_is_aligned_proportionally(self):
        chunker = Chunker(chunk_size=20, chunk_overlap=0)
        original = words(100, "o")
        english = words(50, "e")  # translation half as long
        chunks = chunker.chunk_text(original, text_english=english)

        assert chunks[-1].text_english is not None
        assert chunks[-1].text_english.strip().endswith("e49")
        assert chunks[0].text_english.strip().startswith("e0")

    @pytest.mark.parametrize("size, overlap", [(-5, 0), (10, 10), (10, -1)])
    def test_rejects_invalid_configuration(self, size, overlap):
        with pytest.raises(ValueError):
            Chunker(chunk_size=size, chunk_overlap=overlap)


class TestTimestamped:
    @staticmethod
    def segments(n: int, words_each: int = 10) -> list[dict]:
        return [{"start": i * 5.0, "end": i * 5.0 + 5.0, "text": words(words_each, f"s{i}_")} for i in range(n)]

    def test_chunks_follow_segment_boundaries_with_times(self):
        chunks = Chunker(chunk_size=60, chunk_overlap=0).chunk_text("", timestamps=self.segments(12))

        assert len(chunks) > 1
        assert chunks[0].start_time == 0.0
        assert chunks[-1].end_time == 60.0
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert nxt.start_time == prev.end_time  # no gaps, no overlap

    def test_first_chunk_keeps_zero_start_time(self):
        # 0.0 is a real timestamp, not "missing"
        chunks = Chunker(chunk_size=500, chunk_overlap=0).chunk_text("", timestamps=self.segments(2))
        assert chunks[0].start_time == 0.0

    def test_overlap_repeats_trailing_segments(self):
        segs = self.segments(12, words_each=2)  # small segments so one fits in the overlap
        chunks = Chunker(chunk_size=30, chunk_overlap=12).chunk_text("", timestamps=segs)
        assert len(chunks) > 1
        assert chunks[1].start_time < chunks[0].end_time
        assert chunks[1].text.split()[0] in chunks[0].text

    def test_no_trailing_chunk_made_only_of_overlap(self):
        segs = self.segments(6)
        chunker = Chunker(chunk_size=60, chunk_overlap=25)
        chunks = chunker.chunk_text("", timestamps=segs)
        assert chunks[-1].end_time == segs[-1]["end"]
        assert len({c.end_time for c in chunks}) == len(chunks)

    def test_english_split_across_segments(self):
        segs = self.segments(4, words_each=2)
        chunks = Chunker(chunk_size=500, chunk_overlap=0).chunk_text(
            "", text_english="one two three four five six seven eight", timestamps=segs
        )
        assert chunks[0].text_english == "one two three four five six seven eight"
