import typer
from rich.console import Console

console = Console()

# Define constants
TOKENS_PER_K_WORDS = 1000 / 750  # 1k tokens is equivalent to ~750 words
MONTHLY_MESSAGES = messages_per_day * 24 * 30

# Define functions to calculate cost
def calculate_cost(model, prompt_size, tokens_per_month):
    if model == "gpt4_8k":
        price_per_token_prompt = 0.03
        price_per_token_completion = 0.06
        tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
        tokens_per_completion = tokens_per_prompt
        cost = (tokens_per_month * tokens_per_prompt * price_per_token_prompt
                + tokens_per_month * tokens_per_completion * price_per_token_completion) / 1000
    elif model == "gpt4_32k":
        price_per_token_prompt = 0.06
        price_per_token_completion = 0.12
        tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
        tokens_per_completion = tokens_per_prompt * 4  # GPT-4 32K has 4x the completion response tokens
        cost = (tokens_per_month * tokens_per_prompt * price_per_token_prompt
                + tokens_per_month * tokens_per_completion * price_per_token_completion) / 1000
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

# Define functions to calculate cost
def calculate_cost_ai21(model, prompt_size, tokens_per_month):
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
    cost = tokens_per_month * tokens_per_prompt * price_per_token / 1000
    return cost

# Create CLI app with typer
app = typer.Typer()

@app.command()
def calculate_costs(lower_bound_prompt_size: int = LOWER_BOUND_PROMPT_SIZE, 
                    upper_bound_prompt_size: int = UPPER_BOUND_PROMPT_SIZE, 
                    messages_per_day: int = 25):
    console.print("Calculating costs for different models...", style="bold magenta")

    for model in ["gpt4_8k", "gpt4_32k", "chat_gpt", "ada", "babbage", "curie", "davinci",
                  "embedding_ada", "embedding_curie", "image_1024", "image_512", "image_256", "whisper"]:
        cost_lower_bound = calculate_cost(model, lower_bound_prompt_size, MONTHLY_MESSAGES)
        cost_upper_bound = calculate_cost(model, upper_bound_prompt_size, MONTHLY_MESSAGES)

        if cost_lower_bound is not None and cost_upper_bound is not None:
            console.print(f"Model: {model}", style="bold")
            console.print(f"Lower bound cost: ${cost_lower_bound:.2f}")
            console.print(f"Upper bound cost: ${cost_upper_bound:.2f}")
            console.print("\n")
