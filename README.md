# Promptim

Experimental **prompt** opt**im**ization library.

## Quick start

Let's try prompt optimization on a simple task to generate tweets.

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

Next, create a task to optimize over.

```shell
promptim create task ./my-tweet-task
```

Each task requires a few things. When the CLI requests, provide the corresponding values.
1.  name: provide a useful name for the task (like "ticket classifier" or "report generator"). You may use the default here.
2.  prompt: this is an identifier in the LangSmith prompt hub. Use the following public prompt to start:
```
langchain-ai/tweet-generator-example-with-nothing:starter
```
Hit "Enter" to confirm cloning into your workspace (so that you can push optimized commits to it).
3. dataset: this is the name (or public URL) for the dataset we are optimizing over. Optionally, it can have train/dev/test splits to report separate metrics throughout the training process.
```
https://smith.langchain.com/public/6ed521df-c0d8-42b7-a0db-48dd73a0c680/d
```
4.  description: this is a high-level description of the purpose for this prompt. The optimizer uses this to help focus its improvements.
```
Write informative tweets on any subject.
```

Once you've completed the template creation, you should have two files in hte `my-tweet-task` directory:

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

To help the optimizer know the ideal behavior, we can add additional instrutions in the `comment` field in the response.

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

Whenever you you run `promptim train`, promptim first loads the prompt and dataset specified in your configuration. It then evaluates your prompt on the dev split (if present; full dataset otherwise) using the evaluator(s) configured above. This gives us baseline metrics to compare against throughout the optimization process.

After computing a baseline, `promptim` begins optimizing the prompt by looping over minibatches of training examples. For each minibatch, `promptim` computes the metrics and then applies a **metaprompt** to suggest changes to the current prompt. It then applies that updated prompt to the next minibatch of training examples and repeats the process. It does this over the entire **train** split (if present, full dataset otherwise).

After `promptim` has consumed the whole `train` split, it computes metrics again on the `dev` split. If the metrics show improvement (average score is greater), then the updated prompt is retained for the next round. If the metrics are the same or worse than the current best score, the prompt is discarded.

This process is repeated `--num-epochs` times before the process terminates.

## How to

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

  Optimize prompts for different tasks.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  create  Commands for creating new tasks and examples.
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
  --name TEXT         Name for the task.
  --prompt TEXT       Name of the prompt in LangSmith
  --description TEXT  Description of the task for the optimizer.
  --dataset TEXT      Name of the dataset in LangSmith
  --help              Show this message and exit.
```


#### train

```shell
Usage: promptim train [OPTIONS]

  Train and optimize prompts for different tasks.

Options:
  --task TEXT              Task to optimize. You can pick one off the shelf or
                           select a path to a config file. Example:
                           'examples/tweet_writer/config.json
  --batch-size INTEGER     Batch size for optimization
  --train-size INTEGER     Training size for optimization
  --epochs INTEGER         Number of epochs for optimization
  --debug                  Enable debug mode
  --annotation-queue TEXT  The name of the annotation queue to use. Note: we
                           will delete the queue whenever you resume training
                           (on every batch).
  --no-commit              Do not commit the optimized prompt to the hub
  --help                   Show this message and exit.
```


We have created a few off-the-shelf tasks you can choose from:

- tweet: write tweets
- simpleqa: really hard Q&A
- scone: NLI

![run](./static/optimizer.gif)
```
