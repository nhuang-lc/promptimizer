import asyncio
import argparse

from prompt_optimizer.trainer import PromptOptimizer
from langchain_anthropic import ChatAnthropic
import langsmith as ls
from prompt_optimizer.tasks.scone import scone_task
from prompt_optimizer.tasks.tweet_generator import tweet_task
from prompt_optimizer.tasks.metaprompt import metaprompt_task
from prompt_optimizer.tasks.simpleqa import simpleqa_task
from prompt_optimizer.tasks.ticket_classification import ticket_classification_task


tasks = {
    "scone": scone_task,
    "tweet": tweet_task,
    "metaprompt": metaprompt_task,
    "simpleqa": simpleqa_task,
    "ticket-classification": ticket_classification_task,
}


optimizer = PromptOptimizer(
    model=ChatAnthropic(model="claude-3-5-sonnet-20241022", max_tokens_to_sample=8192),
)


async def run(
    task_name: str,
    batch_size: int,
    train_size: int,
    epochs: int,
    use_annotation_queue: str | None = None,
    debug: bool = False,
):
    task = tasks.get(task_name)
    if not task:
        raise ValueError(f"Unknown task: {task_name}")

    with ls.tracing_context(project_name="Optim"):
        return await optimizer.optimize_prompt(
            task,
            batch_size=batch_size,
            train_size=train_size,
            epochs=epochs,
            use_annotation_queue=use_annotation_queue,
            debug=debug,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize prompts for different tasks."
    )
    parser.add_argument("task", choices=list(tasks), help="Task to optimize")
    parser.add_argument(
        "--batch-size", type=int, default=40, help="Batch size for optimization"
    )
    parser.add_argument(
        "--train-size", type=int, default=40, help="Training size for optimization"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for optimization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--use-annotation-queue",
        type=str,
        default=None,
        help="The name of the annotation queue to use. Note: we will delete the queue whenever you resume training (on every batch).",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run(
            args.task,
            args.batch_size,
            args.train_size,
            args.epochs,
            args.use_annotation_queue,
            args.debug,
        )
    )
    print(results)
