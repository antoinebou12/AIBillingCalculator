import itertools
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pytest
from calculator import *
import os

# Tests for word_to_token_size function
def test_word_to_token_size():
    word = "hello"
    token_count = word_to_token_size(word)

    # GPT-4 tokenizes "hello" as a single token
    assert token_count == 1

    word = "chatbot"
    token_count = word_to_token_size(word)

    # GPT-4 tokenizes "chatbot" as a single token
    assert token_count == 1

    word = "conversational"
    token_count = word_to_token_size(word)

    # GPT-4 tokenizes "conversational" as two tokens
    assert token_count == 2

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


def test_calculate_api_billing():
    # Test with default values (all zero requests)
    assert calculate_api_billing() == 0

    # Test with various API request quantities
    assert calculate_api_billing(paraphrase_requests=1000) == 1
    assert calculate_api_billing(summarize_requests=1000) == 5
    assert calculate_api_billing(grammar_correction_requests=1000) == 0.5
    assert calculate_api_billing(text_improvement_requests=1000) == 0.5
    assert calculate_api_billing(text_segmentation_requests=1000) == 1
    assert calculate_api_billing(contextual_answers_requests=1000) == 5

    # Test with a combination of API request quantities
    total_cost = calculate_api_billing(
        paraphrase_requests=1000,
        summarize_requests=500,
        grammar_correction_requests=2000,
        text_improvement_requests=2000,
        text_segmentation_requests=1000,
        contextual_answers_requests=500,
    )
    assert total_cost == 9.0

def test_export_cost_to_df(mock_calculate_cost, mock_calculate_cost_ai21):
    df = export_cost_to_df()

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Model", "Prompt Size (k words)", "Messages per Day", "Tokens per Month", "Cost per Month ($)"]

    # Check row count
    openai_models = 13
    ai21_models = 3
    prompt_sizes = UPPER_BOUND_PROMPT_SIZE
    messages_per_day = 25
    expected_row_count = (openai_models + ai21_models) * prompt_sizes * messages_per_day
    assert len(df) == expected_row_count

    # Check row content (use the mocked cost values)
    openai_cost = 10.0
    ai21_cost = 20.0
    for _, row in df.iterrows():
        if row['Model'] in ['jumbo', 'grande', 'large']:
            assert row['Cost per Month ($)'] == ai21_cost
        else:
            assert row['Cost per Month ($)'] == openai_cost