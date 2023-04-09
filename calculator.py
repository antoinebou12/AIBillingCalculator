import typer
from rich.console import Console

console = Console()

# Define constants
MONTHLY_MESSAGES = 25 * 24 * 30  # 25 messages every 3 hours for 30 days
TOKENS_PER_K_WORDS = 1000 / 750  # 1k tokens is equivalent to ~750 words
GPT4_PRICE_PER_TOKEN_PROMPT = 0.03  # GPT-4 price per 1k prompt request tokens
GPT4_PRICE_PER_TOKEN_COMPLETION = 0.06  # GPT-4 price per 1k completion response tokens
GPT35_TURBO_PRICE_PER_TOKEN = 0.002  # GPT-3.5-turbo price per token
LOWER_BOUND_PROMPT_SIZE = 50
UPPER_BOUND_PROMPT_SIZE = 200

# Define functions to calculate cost
def calculate_cost_gpt4_8k(prompts_per_month, prompt_size, price_per_token_prompt, price_per_token_completion):
    tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
    tokens_per_completion = tokens_per_prompt
    return (prompts_per_month * tokens_per_prompt * price_per_token_prompt
            + prompts_per_month * tokens_per_completion * price_per_token_completion) / 1000

def calculate_cost_gpt4_32k(prompts_per_month, prompt_size, price_per_token_prompt, price_per_token_completion):
    tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
    tokens_per_completion = tokens_per_prompt * 4  # GPT-4 32K has 4x the completion response tokens
    return (prompts_per_month * tokens_per_prompt * price_per_token_prompt
            + prompts_per_month * tokens_per_completion * price_per_token_completion) / 1000

def calculate_cost_gpt35_turbo(prompts_per_month, prompt_size, price_per_token):
    tokens_per_prompt = prompt_size * TOKENS_PER_K_WORDS
    return prompts_per_month * tokens_per_prompt * price_per_token / 1000

# Create CLI app with typer
app = typer.Typer()

@app.command()
def calculate_costs(lower_bound_prompt_size: int = LOWER_BOUND_PROMPT_SIZE, upper_bound_prompt_size: int = UPPER_BOUND_PROMPT_SIZE):
    # Calculate costs for lower bound of average prompt size (50 tokens per message)
    gpt4_8k_lower_bound_cost = calculate_cost_gpt4_8k(MONTHLY_MESSAGES, lower_bound_prompt_size,
                                                      GPT4_PRICE_PER_TOKEN_PROMPT, GPT4_PRICE_PER_TOKEN_COMPLETION)
    gpt4_32k_lower_bound_cost = calculate_cost_gpt4_32k(MONTHLY_MESSAGES, lower_bound_prompt_size,
                                                        GPT4_PRICE_PER_TOKEN_PROMPT, GPT4_PRICE_PER_TOKEN_COMPLETION)
    gpt35_turbo_lower_bound_cost = calculate_cost_gpt35_turbo(MONTHLY_MESSAGES, lower_bound_prompt_size,
                                                              GPT35_TURBO_PRICE_PER_TOKEN)

    # Calculate costs for upper bound of average prompt size (200 tokens per message)
    gpt4_8k_upper_bound_cost = calculate_cost_gpt4_8k(MONTHLY_MESSAGES, upper_bound_prompt_size,
                                                      GPT4_PRICE_PER_TOKEN_PROMPT, GPT4_PRICE_PER_TOKEN_COMPLETION)
    gpt4_32k_upper_bound_cost = calculate_cost_gpt4_32k(MONTHLY_MESSAGES, upper_bound_prompt_size,
                                                        GPT4_PRICE_PER_TOKEN_PROMPT, GPT4_PRICE_PER_TOKEN_COMPLETION)
