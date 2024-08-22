"""Tests for `dataset.example` module."""

from collections import namedtuple

import numpy as np
import pytest
import torch

from mmlearn.datasets.core.data_collator import DefaultDataCollator
from mmlearn.datasets.core.example import Example, find_matching_indices


def test_example():
    """Test `dataset.example.Example`."""
    # happy path
    an_example = Example()
    assert len(an_example) == 0

    an_example.text = "Hello"
    assert len(an_example) == 1
    assert an_example["text"] == "Hello"
    assert an_example.text == "Hello"

    init_dict = {
        "text": "Hello",
        "number": 123,
        "list": [1, 2, 3],
        "tensor": torch.tensor(1),
        "point": namedtuple("Point", ["x", "y"])(1, 2),
        "mapping": {"a": 1, "b": 2},
        "nested_mapping": {"a": {"b": 1}},
    }
    init_example = Example(init_dict=init_dict)
    assert len(init_example) == 7
    assert init_dict == dict(init_example)

    # test example_id
    init_example.dataset_index = 1
    init_example.example_index = 2
    init_example.create_ids()

    assert all(
        key in init_example.example_ids
        and torch.equal(init_example.example_ids[key], torch.tensor([1, 2]))
        for key in init_dict
    )

    # error path
    with pytest.raises(TypeError, match="'int' object is not iterable.*"):
        example = Example(123)

    example = Example({"text": torch.tensor(2)})
    with pytest.raises(AttributeError):
        example.non_existent_attribute  # noqa: B018


def test_collate_example_list():
    """Test `dataset.example.collate_example_list`."""
    img_class_example = Example(
        {"image": torch.tensor(1), "class_label": torch.tensor(2)},
    )
    img_text_pair = Example({"image": torch.tensor(3), "text": "Hello"})
    audio_text_pair = Example({"audio": torch.tensor(4), "text": "World"})
    nested_example = Example(
        {
            "an_int": 1,
            "a_float": 1.0,
            "a_list": [1, 2, 3],
            "a_tuple": (1, 2, 3),
            "a_tensor": torch.tensor(1),
            "a_mapping": {"a": 1, "b": 2},
            "a_nested_mapping": {"a": {"b": 1}},
            "a_double_nested_mapping": {"a": {"b": {"c": 1}}},
            "a_namedtuple": namedtuple("Point", ["x", "y"])(1, 2),
            "a_numpy_array": np.array([1, 2, 3]),
        },
    )
    expected_result = {
        "image": torch.tensor([1, 3]),
        "class_label": torch.tensor([2]),
        "text": ["Hello", "World"],
        "audio": torch.tensor([4]),
        "an_int": torch.tensor([1]),
        "a_float": torch.tensor([1.0], dtype=torch.float64),
        "a_list": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
        "a_tuple": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])],
        "a_tensor": torch.tensor([1]),
        "a_mapping": {"a": torch.tensor([1]), "b": torch.tensor([2])},
        "a_nested_mapping": {"a": {"b": torch.tensor([1])}},
        "a_double_nested_mapping": {"a": {"b": {"c": torch.tensor([1])}}},
        "a_namedtuple": namedtuple("Point", ["x", "y"])(
            torch.tensor([1]),
            torch.tensor([2]),
        ),
        "a_numpy_array": torch.tensor([[1, 2, 3]]),
    }
    result = DefaultDataCollator()(
        [img_class_example, img_text_pair, audio_text_pair, nested_example],
    )
    for key in expected_result:
        assert key in result
        if isinstance(expected_result[key], torch.Tensor):
            assert torch.equal(result[key], expected_result[key])
        elif isinstance(expected_result[key], list) and isinstance(
            expected_result[key][0],
            torch.Tensor,
        ):
            assert all(
                torch.equal(result[key][i], expected_result[key][i])
                for i in range(len(expected_result[key]))
            )
        elif isinstance(expected_result[key], dict):
            for sub_key in expected_result[key]:
                assert sub_key in result[key]
                if isinstance(expected_result[key][sub_key], torch.Tensor):
                    assert torch.equal(
                        result[key][sub_key],
                        expected_result[key][sub_key],
                    )
                elif isinstance(expected_result[key][sub_key], dict):
                    for sub_sub_key in expected_result[key][sub_key]:
                        if isinstance(
                            expected_result[key][sub_key][sub_sub_key],
                            torch.Tensor,
                        ):
                            assert torch.equal(
                                result[key][sub_key][sub_sub_key],
                                expected_result[key][sub_key][sub_sub_key],
                            )
                        elif isinstance(
                            expected_result[key][sub_key][sub_sub_key],
                            dict,
                        ):
                            for sub_sub_sub_key in expected_result[key][sub_key][
                                sub_sub_key
                            ]:
                                assert torch.equal(
                                    result[key][sub_key][sub_sub_key][sub_sub_sub_key],
                                    expected_result[key][sub_key][sub_sub_key][
                                        sub_sub_sub_key
                                    ],
                                )
        else:
            assert result[key] == expected_result[key]


def test_find_matching_indices():
    """Test `dataset.example.find_matching_indices`."""
    first_example_ids = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)])
    second_example_ids = torch.tensor([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    expected_result = (torch.tensor([2, 3]), torch.tensor([0, 1]))
    result = find_matching_indices(first_example_ids, second_example_ids)
    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], expected_result[0])
    assert torch.equal(result[1], expected_result[1])

    # duplicates
    first_example_ids = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)])
    second_example_ids = torch.tensor([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    expected_result = (torch.tensor([2, 3, 4]), torch.tensor([0, 1, 1]))
    result = find_matching_indices(first_example_ids, second_example_ids)
    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], expected_result[0])
    assert torch.equal(result[1], expected_result[1])

    # no matches
    first_example_ids = torch.tensor([(0, 0), (0, 1), (1, 0), (1, 1)])
    second_example_ids = torch.tensor([(2, 0), (2, 1), (2, 2)])
    expected_result = (torch.tensor([]), torch.tensor([]))
    result = find_matching_indices(first_example_ids, second_example_ids)
    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)
    assert torch.equal(result[0], expected_result[0])
    assert torch.equal(result[1], expected_result[1])

    first_example_ids = [(0, 0), (0, 1), (1, 0), (1, 1)]
    second_example_ids = torch.tensor([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    with pytest.raises(TypeError, match="Expected inputs to be tensors, but got.*"):
        find_matching_indices(first_example_ids, second_example_ids)

    first_example_ids = torch.tensor([(0,), (0,), (1,), (1,)])
    second_example_ids = torch.tensor([(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
    with pytest.raises(
        ValueError,
        match=r"Expected argument `first_example_ids` to be a tensor of shape \(N, 2\).*",
    ):
        find_matching_indices(first_example_ids, second_example_ids)
