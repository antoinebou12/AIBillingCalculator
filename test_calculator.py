import itertools
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pytest
from calculator import *
import os

def test_word_to_token():
    assert word_to_token("Hello") == 1
    assert word_to_token("world") == 1
    assert word_to_token("Hello World") == 2

def test_calculate_cost():
    # round to 2 decimal places
    assert round(calculate_cost("gpt4_8k", 1, 5000), 3) == 0.6
    assert round(calculate_cost("gpt4_32k", 3, 5000), 3) == 10.8
    assert round(calculate_cost("chat_gpt", 10, 50000), 3) == 1.333
    assert round(calculate_cost("ada", 8, 20000), 3) == 0.032
    assert round(calculate_cost("embedding_curie", 1, 1000), 3) == 0.001


def test_calculate_cost_ai21():
    assert round(calculate_cost_ai21("jumbo", 4, 15000), 3) == 1.2
    assert round(calculate_cost_ai21("grande", 6, 20000), 3) == 1.6
    assert round(calculate_cost_ai21("large", 2, 5000), 3) == 0.04


def test_lower_bound_prompt_size():
    assert round(calculate_cost("gpt4_8k", LOWER_BOUND_PROMPT_SIZE, MONTHLY_MESSAGES), 3) > 0


def test_upper_bound_prompt_size():
    assert round(calculate_cost("gpt4_8k", UPPER_BOUND_PROMPT_SIZE, MONTHLY_MESSAGES), 3) > 0
    assert round(calculate_cost("gpt4_8k", UPPER_BOUND_PROMPT_SIZE, MONTHLY_MESSAGES), 3) > round(calculate_cost("gpt4_8k", LOWER_BOUND_PROMPT_SIZE, MONTHLY_MESSAGES), 3)


def test_messages_per_day():
    assert calculate_cost("gpt4_8k", 1, MESSAGES_PER_DAY * 30) > 0
    assert calculate_cost("gpt4_8k", 5, MESSAGES_PER_DAY * 30) > calculate_cost("gpt4_8k", 5, MESSAGES_PER_DAY * 15)


@pytest.fixture
def mock_calculate_cost():
    with patch('calculator.calculate_cost') as mock:
        mock.return_value = 10.0
        yield mock

@pytest.fixture
def mock_calculate_cost_ai21():
    with patch('calculator.calculate_cost_ai21') as mock:
        mock.return_value = 20.0
        yield mock

def test_export_cost_to_csv(mock_calculate_cost, mock_calculate_cost_ai21):
    with TemporaryDirectory() as tempdir:
        csv_file = os.path.join(tempdir, 'costs.csv')
        export_cost_to_csv(csv_file)

        assert os.path.isfile(csv_file)

        with open(csv_file, 'r') as f:
            header(
                f, mock_calculate_cost, mock_calculate_cost_ai21
            )

def header(f, mock_calculate_cost, mock_calculate_cost_ai21):
    reader = csv.reader(f)
    headers = next(reader)
    assert headers == [
        'Model', 'Prompt Size (k words)', 'Messages per Day',
        'Tokens per Month', 'Cost per Month ($)'
    ]

    openai_models = [
        "gpt4_8k",
        "gpt4_32k",
        "chat_gpt",
        "ada",
        "babbage",
        "curie",
        "davinci",
        "embedding_ada",
        "embedding_curie",
        "image_1024",
        "image_512",
        "image_256",
        "whisper",
    ]

    for model, prompt_size in itertools.product(openai_models, range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1)):
        expected_cost = 10.0
        for messages_per_day in range(1, 26):
            expected_tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
            row = next(reader)
            assert row[0] == model
            assert int(row[1]) == prompt_size
            assert int(row[2]) == messages_per_day
            assert int(row[3]) == expected_tokens_per_month
            assert float(row[4]) == expected_cost

    expected_cost = 20.0
    ai21_models = [
        "jumbo",
        "grande",
        "large",
    ]

    for model in ai21_models:
        for prompt_size, messages_per_day in itertools.product(range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1), range(1, 26)):
            expected_tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
            row = next(reader)
            assert row[0] == model
            assert int(row[1]) == prompt_size
            assert int(row[2]) == messages_per_day
            assert int(row[3]) == expected_tokens_per_month
            assert float(row[4]) == expected_cost


def test_calculate_costs(capsys):
    calculate_costs(1, 5, 25)
    captured = capsys.readouterr()
    assert "Calculating costs for different models..." in captured.out
    assert "Model: gpt4_8k" in captured.out
    assert "Model: davinci" in captured.out
    assert "Model: image_256" in captured.out
