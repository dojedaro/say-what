import pytest

from app.services.extractor import VideoExtractor, timestamp_url


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://www.youtube.com/watch?v=a3TTFErF3FY", ("youtube", "a3TTFErF3FY")),
        ("https://youtu.be/a3TTFErF3FY?si=abc", ("youtube", "a3TTFErF3FY")),
        ("https://www.youtube.com/embed/a3TTFErF3FY", ("youtube", "a3TTFErF3FY")),
        ("https://www.tiktok.com/@some.user/video/7234567890123456789", ("tiktok", "7234567890123456789")),
        ("https://www.instagram.com/reel/Cxyz123_-/", ("instagram", "Cxyz123_-")),
    ],
)
def test_detect_platform(url, expected):
    assert VideoExtractor.detect_platform(url) == expected


@pytest.mark.parametrize("url", ["https://vimeo.com/123", "not a url", ""])
def test_detect_platform_rejects_unsupported(url):
    with pytest.raises(ValueError, match="Unsupported"):
        VideoExtractor.detect_platform(url)


def test_timestamp_url():
    assert timestamp_url("youtube", "abc", 0.0) == "https://www.youtube.com/watch?v=abc&t=0s"
    assert timestamp_url("youtube", "abc", 61.7) == "https://www.youtube.com/watch?v=abc&t=61s"
    assert timestamp_url("youtube", "abc", None) is None
    assert timestamp_url("tiktok", "abc", 10) is None
