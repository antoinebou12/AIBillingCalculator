# AIBillingCalculator
The AIBillingCalculator is a command-line tool that calculates the cost of using different OpenAI language models, based on the number of messages, prompt size, and pricing for each model.

## Installation
To use the ChatGPTBillingCalculator, you will need to have Python 3.x installed on your system. You can install the required dependencies by running the following command in your terminal:

```shell
pip install -r requirements.txt
```

## Usage
To run the ChatGPTBillingCalculator, navigate to the project directory in your terminal and run the following command:

```shell
python main.py calculate_costs --model [MODEL_NAME] --prompt-size [PROMPT_SIZE] --messages-per-day [MESSAGES_PER_DAY]
```

Replace [MODEL_NAME] with the name of the OpenAI model you want to calculate the cost for. The available models are:

-gpt4_8k
-gpt4_32k
-chat_gpt
-ada
-babbage
-curie
-davinci
-embedding_ada
-embedding_curie
-image_1024
-image_512
-image_256
-whisper

Replace [PROMPT_SIZE] with the average size of the prompt in tokens (e.g., 50). Replace [MESSAGES_PER_DAY] with the number of messages per day that you expect to send (e.g., 25).

The command will output the monthly cost of using the selected model based on the given parameters.

## Contributing
If you would like to contribute to the ChatGPTBillingCalculator, please fork the repository and submit a pull request.

## License
The ChatGPTBillingCalculator is licensed under the MIT License. See the LICENSE file for more information.
