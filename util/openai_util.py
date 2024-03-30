import openai
import loguru

logger = loguru.logger
COST_FACTORS = {
    "gpt-3.5-turbo": 0.002 / 1000, # 0.002 per 1000 tokens
}

def calculate_cost(tokens: int, model: str = "gpt-3.5-turbo") -> float:
    if model not in COST_FACTORS:
        raise ValueError(f"Unknown model: {model}")
    cost_factor = COST_FACTORS[model]
    return tokens * cost_factor

cost = 0
maximum_cost = 0.1 # Maximum cost in each run

def completion_with_chatgpt(text: str, model: str = "gpt-3.5-turbo") -> str:
    global cost
    if cost >= maximum_cost:
        logger.warning(f"Cost exceeds the limit: {cost} used, {maximum_cost} allowed")
        raise ValueError("Cost exceeds the limit")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", # Don't change this
        messages=[
            {"role": "user", "content": text},
        ],
    )
    result = response["choices"][0]["message"]["content"]
    tokens = response["usage"]["total_tokens"]
    this_cost = calculate_cost(tokens)
    cost += this_cost
    logger.info(f"Tokens used: {tokens}, estimated cost: {this_cost}")
    return result

def estimated_cost():
    return cost
