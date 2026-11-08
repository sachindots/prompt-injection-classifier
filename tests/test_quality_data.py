from injclf.quality_data import sample


def test_exam_reserves_normalized_text_even_outside_sample():
    seen = {"already used"}
    records = [
        {"text": " Already  USED ", "label": 1},
        {"text": "first public example", "label": 0},
        {"text": "second public example", "label": 0},
        {"text": "attack", "label": 1},
    ]
    exam = sample(records, seen, 1, 42)
    assert len(exam) == 2
    assert all(text.strip().lower() != "already  used" for text, _ in exam)
    assert sample(records, seen, 10, 42) == []
