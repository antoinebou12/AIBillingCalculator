import itertools
from tokenizers import Tokenizer
import typer
from rich.console import Console
import tiktoken
import csv
import pandas as pd

console = Console()

# Create CLI app with typer
app = typer.Typer()

# Define constants
# 1 token ~= 4 chars in English
# 1 token ~= ¾ words
# 100 tokens ~= 75 words
# 1-2 sentence ~= 30 tokens
# 1 paragraph ~= 100 tokens
# 1,500 words ~= 2048 tokens

TOKENS_PER_K_WORDS = 1000 / 750  # 1k tokens is equivalent to ~750 words
MESSAGES_PER_DAY = 25
MONTHLY_MESSAGES = MESSAGES_PER_DAY * 24 * 30
LOWER_BOUND_PROMPT_SIZE = 1  # 1k words
UPPER_BOUND_PROMPT_SIZE = 10  # 10k words
TOKENS_PER_MESSAGE = 1000
TOKENS_LIMIT = 4000


def word_to_token(word):
    """
    Convert a word to the number of tokens it would take to generate it.

    Parameters
    ----------
    word : str
        The word to convert.

    Returns
    -------
    tokens : int
        The number of tokens it would take to generate the word.
    """
    return len(tiktoken.encoding_for_model("gpt-4").encode(word))


# Define functions to calculate cost
def calculate_cost(model, prompt_size, tokens_per_month):
    """
    Calculate the cost of using a model for a given prompt size and number of tokens per month.

    Parameters
    ----------
    model : str
        The model to use.
        prompt_size : int
        The size of the prompt in k words.
        tokens_per_month : int
        The number of tokens to generate per month.

    Returns
    -------
        cost : float
            The cost of using the model for the given prompt size and number of tokens per month.
    Raises
    ------
        ValueError
            If the model is not supported.
    """
    if model == "gpt4_8k":
        price_per_token_prompt = 0.03
        price_per_token_completion = 0.06
        tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
        tokens_per_completion = tokens_per_prompt
        cost = (
            tokens_per_month * tokens_per_prompt * price_per_token_prompt
            + tokens_per_month * tokens_per_completion * price_per_token_completion
        ) / 1000
    elif model == "gpt4_32k":
        price_per_token_prompt = 0.06
        price_per_token_completion = 0.12
        tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
        # GPT-4 32K has 4x the completion response tokens
        tokens_per_completion = tokens_per_prompt * 4
        cost = (
            tokens_per_month * tokens_per_prompt * price_per_token_prompt
            + tokens_per_month * tokens_per_completion * price_per_token_completion
        ) / 1000
    elif model == "chat_gpt":
        price_per_token = 0.002
        tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
        cost = tokens_per_month * tokens_per_prompt * price_per_token / 1000
    elif model in ["ada", "babbage", "curie", "davinci"]:
        if model == "ada":
            price_per_training_token = 0.0004
            price_per_usage_token = 0.0016
        elif model == "babbage":
            price_per_training_token = 0.0006
            price_per_usage_token = 0.0024
        elif model == "curie":
            price_per_training_token = 0.0030
            price_per_usage_token = 0.0120
        elif model == "davinci":
            price_per_training_token = 0.0300
            price_per_usage_token = 0.1200
        cost = tokens_per_month * price_per_usage_token / 1000
    elif model == "embedding_ada":
        price_per_token = 0.0004
        cost = tokens_per_month * price_per_token / 1000
    elif model == "embedding_curie":
        price_per_token = 0.0006
        cost = tokens_per_month * price_per_token / 1000
    elif model == "image_1024":
        price_per_image = 0.020
        cost = price_per_image * tokens_per_month
    elif model == "image_512":
        price_per_image = 0.018
        cost = price_per_image * tokens_per_month
    elif model == "image_256":
        price_per_image = 0.016
        cost = price_per_image * tokens_per_month
    elif model == "whisper":
        price_per_minute = 0.006
        cost = price_per_minute * tokens_per_month / 60
    else:
        console.print("Invalid model. Please choose a valid model.", style="bold red")
        return None
    return cost


# Define functions to calculate cost for AI21 models
def calculate_cost_ai21(model, prompt_size, tokens_per_month):
    """
    Calculate the cost of using a model for a given prompt size and number of tokens per month.

    Parameters
    ----------
    model : str
        The model to use.
        prompt_size : int
        The size of the prompt in k words.
        tokens_per_month : int
        The number of tokens to generate per month.

    Returns
    -------
        cost : float
            The cost of using the model for the given prompt size and number of tokens per month.
    Raises
    ------
        ValueError
            If the model is not supported.
    """

    if model == "jumbo":
        price_per_token = 0.015
    elif model == "grande":
        price_per_token = 0.01
    elif model == "large":
        price_per_token = 0.003
    else:
        console.print("Invalid model. Please choose a valid model.", style="bold red")
        return None

    tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
    return tokens_per_month * tokens_per_prompt * price_per_token / 1000


def export_cost_to_csv(file):
    """
    Export the cost of using OpenAI and AI21 models for a given prompt size and number of messages per day to a CSV file.

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    console.print("Exporting costs to CSV file...", style="bold magenta")
    with open(file, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "Model",
                "Prompt Size (k words)",
                "Messages per Day",
                "Tokens per Month",
                "Cost per Month ($)",
            ]
        )
        for model in [
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
        ]:
            for prompt_size, messages_per_day in itertools.product(range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1), range(1, 26)):
                tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
                cost = calculate_cost(model, prompt_size, tokens_per_month)
                csv_writer.writerow(
                    [model, prompt_size, messages_per_day, tokens_per_month, cost]
                )
        for model in ["jumbo", "grande", "large"]:
            for prompt_size, messages_per_day in itertools.product(range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1), range(1, 26)):
                tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
                cost = calculate_cost_ai21(model, prompt_size, tokens_per_month)
                csv_writer.writerow(
                    [model, prompt_size, messages_per_day, tokens_per_month, cost]
                )
    console.print("Costs exported to CSV file.", style="bold green")

    return None

def export_cost_to_df(file=None):
    """
    Export the cost of using OpenAI and AI21 models for a given prompt size and number of messages per day to a Pandas DataFrame.

    Parameters
    ----------
        None
    Returns
    -------
        df : pandas.DataFrame
            The DataFrame containing the cost of using OpenAI and AI21 models for a given prompt size and number of messages per day.
    """

    df = pd.DataFrame(
        columns=[
            "Model",
            "Prompt Size (k words)",
            "Messages per Day",
            "Tokens per Month",
            "Cost per Month ($)",
        ]
    )
    for model in [
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
        ]:
        for prompt_size, messages_per_day in itertools.product(range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1), range(1, 26)):
            tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
            cost = calculate_cost(model, prompt_size, tokens_per_month)
            df = df.append(
                {
                    "Model": model,
                    "Prompt Size (k words)": prompt_size,
                    "Messages per Day": messages_per_day,
                    "Tokens per Month": tokens_per_month,
                    "Cost per Month ($)": cost,
                },
                ignore_index=True,
            )

    for model in ["jumbo", "grande", "large"]:
        for prompt_size, messages_per_day in itertools.product(range(LOWER_BOUND_PROMPT_SIZE, UPPER_BOUND_PROMPT_SIZE + 1), range(1, 26)):
            tokens_per_month = messages_per_day * TOKENS_PER_MESSAGE
            cost = calculate_cost_ai21(model, prompt_size, tokens_per_month)
            df = df.append(
                {
                    "Model": model,
                    "Prompt Size (k words)": prompt_size,
                    "Messages per Day": messages_per_day,
                    "Tokens per Month": tokens_per_month,
                    "Cost per Month ($)": cost,
                },
                ignore_index=True,
            )

    console.print("Costs exported to DataFrame.", style="bold green")
    if file:
        df.to_csv(file, index=False)
    return df

@app.command()
def calculate_costs(
    lower_bound_prompt_size: int = LOWER_BOUND_PROMPT_SIZE,
    upper_bound_prompt_size: int = UPPER_BOUND_PROMPT_SIZE,
    messages_per_day: int = 25,
):
    """
    Calculate the cost of using OpenAI and AI21 models for a given prompt size and number of messages per day.

    Parameters
    ----------
    lower_bound_prompt_size : int
        The lower bound of the prompt size in k words.
        upper_bound_prompt_size : int
        The upper bound of the prompt size in k words.
        messages_per_day : int
        The number of messages to generate per day.

    Returns
    -------
        None
    """
    console.print("Calculating costs for different models...", style="bold magenta")

    for model in [
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
    ]:
        cost_lower_bound = calculate_cost(
            model, lower_bound_prompt_size, MONTHLY_MESSAGES
        )
        cost_upper_bound = calculate_cost(
            model, upper_bound_prompt_size, MONTHLY_MESSAGES
        )

        if cost_lower_bound is not None and cost_upper_bound is not None:
            console.print(f"Model: {model}", style="bold")
            console.print(f"Lower bound cost: ${cost_lower_bound:.2f}")
            console.print(f"Upper bound cost: ${cost_upper_bound:.2f}")
            console.print("\n")


if __name__ == "__main__":
    app()
