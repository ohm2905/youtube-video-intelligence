import pytest
from backend.app.services.youtube import extract_video_id, format_duration


def test_extract_video_id_standard():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_with_params():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&feature=youtu.be"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_short():
    url = "https://youtu.be/dQw4w9WgXcQ?t=10"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_embed():
    url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_shorts():
    url = "https://youtube.com/shorts/dQw4w9WgXcQ"
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_extract_video_id_raw():
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    with pytest.raises(ValueError):
        extract_video_id("https://google.com/search?q=test")


def test_format_duration():
    assert format_duration(None) is None
    assert format_duration(59) == "00:59"
    assert format_duration(3661) == "01:01:01"
