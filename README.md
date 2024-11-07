# Promptim

Promptim is an experimental **prompt** opt**im**ization library to help you systematically improve your AI systems.

Promptim automates the process of improving prompts on specific tasks. You provide initial prompt, a dataset, and custom evaluators (and optional human feedback), and `promptim` runs an optimization loop to produce a refined prompt that aims to outperform the original.

For setup and usage details, see the Quick Start guide below.

![Optimization](./static/optimizer.gif)

## Quick start

Let's try prompt optimization on a simple tweet generation task.

### 1. Install

First install the CLI.

```shell
pip install -U promptim
```

And make sure you have a valid [LangSmith API Key](https://smith.langchain.com/) in your environment. For the quick start task, we will use Anthropic's Claude model for our optimizer and for the target system.

```shell
LANGSMITH_API_KEY=CHANGEME
ANTHROPIC_API_KEY=CHANGEME
```

### 2. Create task

Next, create a task to optimize over. Run the following command to generate a template:

```shell
promptim create task ./my-tweet-task \
    --name my-tweet-task \
    --prompt langchain-ai/tweet-generator-example-with-nothing:starter \
    --dataset https://smith.langchain.com/public/6ed521df-c0d8-42b7-a0db-48dd73a0c680/d \
    --description "Write informative tweets on any subject." \
    -y
```
This command will generate starter code, complete with the task's:
1. Name: Provide a useful name for the task (like "ticket classifier" or "report generator"). You may use the default here.
2. Prompt: This is an identifier in the LangSmith prompt hub. Use the following public prompt to start.
3. Dataset: This is the name (or public URL) for the dataset we are optimizing over. Optionally, it can have train/dev/test splits to report separate metrics throughout the training process.
4. Description: This is a high-level description of the purpose for this prompt. The optimizer uses this to help focus its improvements.

Once you've completed the template creation, you should have two files in the `my-tweet-task` directory:

```shell
└── my-tweet-task
    ├── config.json
    └── task.py
```

We can ignore the `config.json` file for now (we'll discuss that later). The last thing we need to do before training is create an evaluator.

### 3. Define evaluators

Next we need to quantify prompt performance on our task. What does "good" and "bad" look like? We do this using evaluators.

Open the evaluator stub written in `my-tweet-task/task.py` and find the line that assigns a score to a prediction:

```python
    # Implement your evaluation logic here
    score = len(str(predicted.content)) < 180  # Replace with actual score
```

We are going to make this evaluator penalize outputs with hashtags. Update that line to be:
```python
    score = int("#" not in result)
```

Next, update the evaluator name. We do this using the `key` field in the evaluator response.
```python
    "key": "tweet_omits_hashtags",
```

To help the optimizer know the ideal behavior, we can add additional instructions in the `comment` field in the response.

Update the "comment" line to explicitly give pass/fail comments:
```python
        "comment": "Pass: tweet omits hashtags" if score == 1 else "Fail: omit all hashtags from generated tweets",
```

And now we're ready to train! The final evaluator should look like:

```python
def example_evaluator(run: Run, example: Example) -> dict:
    """An example evaluator. Larger numbers are better."""
    predicted: AIMessage = run.outputs["output"]

    result = str(predicted.content)
    score = int("#" not in result)
    return {
        "key": "tweet_omits_hashtags",
        "score": score,
        "comment": "Pass: tweet omits hashtags" if score == 1 else "Fail: omit all hashtags from generated tweets",
    }

```

### 4. Train

To start optimizing your prompt, run the `train` command:

```shell
promptim train --task ./my-tweet-task/config.json
```

You will see the progress in your terminal. once it's completed, the training job will print out the final "optimized" prompt in the terminal, as well as a link to the commit in the hub.

### Explanation

Whenever you run `promptim train`, promptim first loads the prompt and dataset specified in your configuration. It then evaluates your prompt on the dev split (if present; full dataset otherwise) using the evaluator(s) configured above. This gives us baseline metrics to compare against throughout the optimization process.

After computing a baseline, `promptim` begins optimizing the prompt by looping over minibatches of training examples. For each minibatch, `promptim` computes the metrics and then applies a **metaprompt** to suggest changes to the current prompt. It then applies that updated prompt to the next minibatch of training examples and repeats the process. It does this over the entire **train** split (if present; full dataset otherwise).

After `promptim` has consumed the whole `train` split, it computes metrics again on the `dev` split. If the metrics show improvement (average score is greater), then the updated prompt is retained for the next round. If the metrics are the same or worse than the current best score, the prompt is discarded.

This process is repeated `--num-epochs` times before the process terminates.

## How to:

### Add human labels

To add human labeling using the annotation queue:

1. Set up an annotation queue:
   When running the `train` command, use the `--annotation-queue` option to specify a queue name:
   ```
   promptim train --task ./my-tweet-task/config.json --annotation-queue my_queue
   ```

2. During training, the system will pause after each batch and print out instructions on how to label the results. It will wait for human annotations.

3. Access the annotation interface:
   - Open the LangSmith UI
   - Navigate to the specified queue (e.g., "my_queue")
   - Review and label as many examples as you'd like, adding notes and scores

4. Resume:
   - Type 'c' in the terminal
   - The training loop will fetch your annotations and include them in the metaprompt's next optimizatin pass

This human-in-the-loop approach allows you to guide the prompt optimization process by providing direct feedback on the model's outputs.

## Reference

### CLI Arguments

The current CLI arguments are as follows. They are experimental and may change in the future:

```shell
Usage: promptim [OPTIONS] COMMAND [ARGS]...

  Optimize prompts for AI tasks using automated evaluation and feedback.

  Promptim helps improve prompts for various AI tasks by running an
  optimization loop. You provide an initial prompt, a dataset, and custom
  evaluators. Promptim then iteratively refines the prompt to improve
  performance on your specific task.

  To get started, create a task configuration or use a pre-defined one, then
  run the 'train' command to begin optimization.

  Example:     promptim train --task ./my-task/config.json

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  create  Commands for creating new tasks.
  train   Train and optimize prompts for different tasks.
```

#### create


```shell
Usage: promptim create [OPTIONS] COMMAND [ARGS]...

  Commands for creating new tasks and examples.

Options:
  --help  Show this message and exit.

Commands:
  example  Clone a pre-made tweet generation task
  task     Walkthrough to create a new task directory from your own prompt and dataset
```

`promptim create task`

```shell
Usage: promptim create task [OPTIONS] PATH

  Create a new task directory with config.json and task file for a custom
  prompt and dataset.

Options:
  --name TEXT         Name for the task. If not provided, the directory name
                      will be used as default. This name will be used in the
                      config.json file.
  --prompt TEXT       Name of the prompt in LangSmith to be optimized.
                      If not provided, you'll be prompted to select or create
                      one. This will be used as the initial prompt for
                      optimization.
  --description TEXT  Description of the task for the optimizer. This helps
                      guide the optimization process by providing context
                      about the task's objectives and constraints.
  --dataset TEXT      Name or public URL of the dataset in LangSmith to be used for
                      training and evaluation. If not provided, you'll be
                      prompted to select or create one. This dataset will be
                      used to test and improve the prompt.
  -y, --yes           Automatically answer yes to all CLI prompts. Use with
                      caution as it skips confirmation steps and uses defaults
                      where applicable.
  --help              Show this message and exit.
```


#### train

```shell
Usage: promptim train [OPTIONS]

  Train and optimize prompts for different tasks.

Options:
  --task TEXT              Task to optimize. Specify a pre-defined task name
                           or path to a custom config file. The task defines
                           the dataset, evaluators, and initial prompt to
                           optimize. Example:
                           'examples/tweet_writer/config.json' for a custom
                           task, or 'sentiment_analysis' for a pre-defined
                           task.
  --batch-size INTEGER     Number of examples to process in each optimization
                           iteration. Larger batches may improve stability but
                           are limited by the metaprompter's maximum context
                           window size.
  --train-size INTEGER     Maximum number of training examples to use per
                           epoch. Useful for limiting optimization time on
                           large datasets. If smaller than total available
                           data, a random subset will be used each epoch.
  --epochs INTEGER         Number of complete passes through the training
                           data. More epochs may improve results but increase
                           runtime.
  --debug                  Enable debug mode for verbose logging and
                           sequential processing.
  --annotation-queue TEXT  Name of the LangSmith annotation queue for manual
                           review of optimization results. The queue will be
                           cleared and updated on each batch.
  --no-commit              Prevent committing the optimized prompt to the
                           LangChain Hub. Use this for local experimentation.
  --help                   Show this message and exit.
```

### Configuration

The schema for your `config.json` file can be found in [config-schema.json](./config-schema.json).

It contains the following arguments:

- `name` (string, required): The name of your task.
- `dataset` (string, required): The name of the dataset in LangSmith to be used for training and evaluation.
- `initial_prompt` (object, required): Configuration for the initial prompt to be optimized.
  - `identifier` (string, optional): If optimizing a prompt from the hub. Do not provide if using a prompt string directly.
  - `model_config` (object, optional): Configuration for the model used in optimization.
  - `which` (integer, default: 0): Which message in the prompt to optimize.
- `description` (string, optional): A detailed explanation of the task's objectives and constraints.
- `evaluator_descriptions` (object, optional): A mapping of evaluator names to their descriptions.
- `optimizer` (object, optional): Configuration for the optimization process.
  - `model` (object, required): Configuration for the model used in optimization, including model name and parameters.
- `evaluators` (string, required): Path to the Python file and variable name containing the evaluator functions for the task. Example: `./task/evaluators.py:evaluators`
- `system` (string, optional): Path to the Python file defining the custom system for making predictions. If not provided, one will be constructed for you (contaiing just a prompt and LLM). Example: `./task/my_system.py:chain`


Below is an example `config.json` file:

```json
{
  "name": "Tweet Generator",
  "dataset": "tweet_dataset",
  "initial_prompt": {
    "prompt_str": "Write a tweet about {topic} in the style of {author}",
    "which": 0
  },
  "description": "Generate engaging tweets on various topics in the style of different authors",
  "evaluator_descriptions": {
    "engagement_score": "Measures the potential engagement of the tweet",
    "style_match": "Evaluates how well the tweet matches the specified author's style"
  },
  "evaluators": "./tweet_evaluators.py:evaluators",
  "optimizer": {
    "model": {
      "name": "gpt-3.5-turbo",
      "temperature": 0.7
    }
  }
}
```
