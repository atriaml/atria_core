from __future__ import annotations

import numpy as np
import pytest

from atria_core.types._generic._elements import ElementArray, OCRLevel


def _hierarchy() -> ElementArray:
    # page(0) -> line(1) -> word(2), word(3)
    return ElementArray(
        ids=np.array([0, 1, 2, 3]),
        parent_ids=np.array([-1, 0, 1, 1]),
        levels=np.array(
            [
                OCRLevel.page.value,
                OCRLevel.line.value,
                OCRLevel.word.value,
                OCRLevel.word.value,
            ]
        ),
        bboxes=np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.1, 0.1, 0.6, 0.2],
                [0.1, 0.1, 0.3, 0.2],
                [0.35, 0.1, 0.6, 0.2],
            ]
        ),
        texts=np.array(["", "", "hello", "world"], dtype=object),
    )


def test_from_words_flat() -> None:
    ea = ElementArray.from_words(["hello", "world"], [[0.1, 0.1, 0.3, 0.2], [0.35, 0.1, 0.6, 0.2]])
    assert list(ea.levels) == [OCRLevel.word.value, OCRLevel.word.value]
    assert list(ea.parent_ids) == [-1, -1]
    assert ea.joined_text() == "hello world"


def test_at_filters_by_level() -> None:
    ea = _hierarchy()
    words = ea.at(OCRLevel.word)
    assert list(words.texts) == ["hello", "world"]
    assert len(words) == 2


def test_segment_bboxes_gathers_parent_line_box() -> None:
    ea = _hierarchy()
    segments = ea.segment_bboxes(OCRLevel.word)
    assert np.array_equal(segments, [[0.1, 0.1, 0.6, 0.2], [0.1, 0.1, 0.6, 0.2]])


def test_parent_bbox_root_falls_back_to_own_box() -> None:
    ea = _hierarchy()
    parents = ea.parent_bbox()
    # page (id 0) is root -> its own box
    assert np.array_equal(parents[0], ea.bboxes[0])


def test_joined_text_default_word_level() -> None:
    ea = _hierarchy()
    assert ea.joined_text() == "hello world"


def test_to_dict_from_dict_roundtrip() -> None:
    ea = _hierarchy()
    data = ea.to_dict()
    restored = ElementArray.from_dict(data)
    assert restored.to_dict() == data


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="expected"):
        ElementArray(bboxes=np.zeros((2, 4)), texts=np.array(["only one"], dtype=object))


def test_rejects_unnormalized_bbox() -> None:
    with pytest.raises(ValueError, match="normalized"):
        ElementArray(
            bboxes=np.array([[0.0, 0.0, 2.0, 2.0]]), texts=np.array(["x"], dtype=object)
        )


def test_rejects_non_ndarray_texts() -> None:
    with pytest.raises(TypeError, match="texts"):
        ElementArray(texts=["x"])  # type: ignore[arg-type]


def test_at_produces_slice_with_dangling_parent_ids_uncheck() -> None:
    # at() doesn't call validate_hierarchy() -- a word-only slice's
    # parent_ids point at line ids no longer present, by design.
    ea = _hierarchy()
    words = ea.at(OCRLevel.word)
    with pytest.raises(ValueError, match="dangling"):
        words.validate_hierarchy()


def test_validate_hierarchy_rejects_dangling_parent_id() -> None:
    ea = ElementArray(
        ids=np.array([0]),
        parent_ids=np.array([99]),
        texts=np.array(["x"], dtype=object),
    )
    with pytest.raises(ValueError, match="dangling"):
        ea.validate_hierarchy()


def test_validate_hierarchy_passes_for_well_formed_tree() -> None:
    _hierarchy().validate_hierarchy()


def test_empty_array_is_falsy_len() -> None:
    ea = ElementArray()
    assert len(ea) == 0
    assert ea.joined_text() == ""
    assert ea.segment_bboxes().shape == (0, 4)
