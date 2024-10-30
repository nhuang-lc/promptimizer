import random
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Optional
from uuid import UUID

import langsmith as ls
import numpy as np
from langchain_core.language_models import BaseChatModel
from langsmith.evaluation._arunner import ExperimentResultRow
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field
from rich import print as richprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text
from langsmith.evaluation import _runner, _arunner


def ltq():
    return lambda x: x


_runner._load_tqdm = ltq
_arunner._load_tqdm = ltq


def _noop(*args, **kwargs):
    pass


_runner.print = _noop
_arunner.print = _noop


DEFAULT_METAPROMPT = """You are an expert prompt engineer tasked with improving prompts for AI tasks.
You will use all means necessary to optimize the scores for the provided prompt so that the resulting model can
perform well on the target task.

## Current prompt

The following is the current best-performing prompt:

<current_prompt>
{current_prompt}
</current_prompt>

## Previous Prompt Attempts

You previously attempted to use the following prompts, but they earned worse scores than the current one:
<other_attempts>
{other_attempts}
</other_attempts>

Reflect on your previous attempts to ensure you search for and identify better patterns.

## Annotated results:
<results>
{annotated_results}
</results>

## Task description:
<task_description>
{task_description}
</task_description>

Unless otherwise specified, higher scores are better (try to maximize scores). Aim for perfect scores across all examples.

In your head, search through all edits, planning the optimization step-by-step:
1. Analyze the current results and where they fall short
2. Identify patterns in successful vs unsuccessful cases
3. Propose specific improvements to address the shortcomings
4. Generate an improved prompt that maintains all required formatting

The improved prompt must:
- Keep all original input variables
- Maintain any special formatting or delimiters
- Focus on improving the specified metrics
- Be clear and concise.
- Avoid repeating mistakes.

Use prompting strategies as appropriate for the task. For logic and math, consider encourage more chain-of-thought reasoning, 
or include reasoning trajectories to induce better performance. For creative tasks, consider adding style guidelines.
Or consider including examplars.

Output your response in this format:
<analysis>
Your step-by-step analysis here...
</analysis>

<improved_prompt>
Your improved prompt here...
</improved_prompt>"""

SystemType = Callable[[str, dict], dict]
"""Takes the current prompt and the example inputs and returns the results."""


@dataclass(kw_only=True)
class Task:
    """Represents a specific task for prompt optimization."""

    name: str
    description: str = ""
    evaluator_descriptions: dict = field(default_factory=dict)
    train_dataset_name: str
    dev_dataset_name: str
    test_dataset_name: str
    evaluators: list[Callable[[Run, Example], dict]]
    initial_prompt: str
    system: SystemType
    baseline_experiment: UUID | None = None

    def describe(self):
        descript = self.description if self.description else self.name
        evaluator_desc = "\n".join(
            [f"- {key}: {value}" for key, value in self.evaluator_descriptions.items()]
        )
        return f"{descript}\n\nDescription of scores:\n{evaluator_desc}"


class OptimizedPromptOutput(BaseModel):
    """Schema for the optimized prompt output."""

    analysis: str = Field(
        description="First, analyze the current results and plan improvements to reconcile them."
    )
    improved_prompt: str = Field(description="The improved prompt text")


class PromptOptimizer:
    """A framework for optimizing meta-prompts through multi-task evaluation."""

    def __init__(
        self,
        model: BaseChatModel,
        meta_prompt: Optional[str] = None,
        seed: int = 42,
    ):
        self.model = model
        self.client = ls.Client()
        self.meta_prompt = meta_prompt or DEFAULT_METAPROMPT
        random.seed(seed)
        self.rng = random.Random(seed)

    async def optimize_prompt(
        self,
        task: Task,
        train_size: Optional[int] = None,
        batch_size: int = 40,
        epochs: int = 1,
        debug: bool = False,
    ) -> tuple[str, float]:
        """Optimizes a prompt for a specific task through multiple iterations."""
        current_prompt = task.initial_prompt
        best_score = 0
        best_prompt = task.initial_prompt
        other_attempts = []

        # Print the original prompt
        richprint(
            Panel.fit(
                f"[bold cyan]Original Prompt:[/bold cyan]\n\n{task.initial_prompt}",
                title="Starting Prompt",
                border_style="bold",
            )
        )

        with Progress() as progress:
            main_task = progress.add_task("[cyan]Optimizing prompt...", total=100)

            # Step 1: Get baseline scores
            progress.update(
                main_task, advance=10, description="[cyan]Getting baseline scores..."
            )
            if task.baseline_experiment:
                baseline_scores = await self._fetch_baseline_metrics(
                    task.baseline_experiment
                )
            else:
                baseline_experiment_results = await self._evaluate_prompt(
                    current_prompt, task, task.dev_dataset_name, debug=debug
                )
                baseline_scores = await self.calculate_scores(
                    baseline_experiment_results
                )
            best_score = sum(baseline_scores.values()) / len(baseline_scores)
            baseline_scores_output = "[cyan]Baseline scores:\n"
            for metric, score in baseline_scores.items():
                baseline_scores_output += f"  {metric}: {score:.4f}\n"
            baseline_scores_output += f"Overall baseline score: {best_score:.4f}"
            progress.console.print(baseline_scores_output)
            progress.console.print()

            # Step 2: Train
            progress.update(
                main_task, advance=10, description="[cyan]Training prompt..."
            )
            train_examples = list(
                self.client.list_examples(dataset_name=task.train_dataset_name)
            )
            epoch_task = progress.add_task("[green]Epochs", total=epochs)
            for epoch in range(epochs):
                self.rng.shuffle(train_examples)
                if train_size:
                    train_examples = train_examples[:train_size]

                batches = [
                    train_examples[i : i + batch_size]
                    for i in range(0, len(train_examples), batch_size)
                ]

                batch_task = progress.add_task(
                    f"[yellow]Epoch {epoch+1} batches", total=len(batches)
                )
                all_train_scores = []
                for batch in batches:
                    results = await self._evaluate_prompt(
                        current_prompt, task, batch, debug=debug
                    )
                    train_scores = await self.calculate_scores(results)
                    train_score = sum(train_scores.values()) / len(train_scores)
                    all_train_scores.append(train_score)
                    progress.console.print(
                        f"Batch train score: {train_score:.4f}", end="\n"
                    )
                    progress.console.print()
                    improved = await self.apply_metaprompt(
                        current_prompt=current_prompt,
                        other_attempts=other_attempts,
                        meta_prompt=self.meta_prompt,
                        task=task,
                        results=results,
                    )
                    current_prompt = improved.improved_prompt
                    progress.update(batch_task, advance=1)

                console = Console()

                train_scores_panel = Panel(
                    Text(", ".join(f"{score:.4f}" for score in all_train_scores)),
                    title="Train Scores",
                    expand=False,
                    border_style="bold",
                )
                console.print(train_scores_panel)
                # Evaluate on dev set after each epoch
                progress.update(main_task, description="[cyan]Evaluating on dev set...")
                dev_results = await self._evaluate_prompt(
                    current_prompt, task, task.dev_dataset_name, debug=debug
                )
                dev_scores = await self.calculate_scores(dev_results)
                dev_score = sum(dev_scores.values()) / len(dev_scores)

                if dev_score > best_score:
                    if best_prompt not in other_attempts:
                        other_attempts.append(best_prompt)
                    best_score = dev_score
                    best_prompt = current_prompt
                    progress.console.print(
                        f"New best score: {best_score:.4f} (surpassed previous best)"
                    )
                else:
                    other_attempts.append(current_prompt)
                    current_prompt = best_prompt
                    progress.console.print(
                        f"Score {dev_score:.4f} did not surpass best score {best_score:.4f}"
                    )
                progress.console.print()

                progress.console.print(
                    Panel(
                        f"[bold]Epoch {epoch+1}[/bold]\n"
                        f"Dev score: [cyan]{dev_score:.4f}[/cyan]\n"
                        f"Best score: [green]{best_score:.4f}[/green]",
                        title="Training Progress",
                        expand=False,
                        border_style="bold",
                    )
                )
                progress.console.print()
                progress.update(epoch_task, advance=1)

            # Step 3: Test
            progress.update(
                main_task, advance=10, description="[cyan]Running final tests..."
            )
            initial_test_results = await self._evaluate_prompt(
                task.initial_prompt, task, task.test_dataset_name, debug=debug
            )
            final_test_results = await self._evaluate_prompt(
                best_prompt, task, task.test_dataset_name, debug=debug
            )

            progress.update(
                main_task, advance=10, description="[cyan]Optimization complete!"
            )
        # Print final report
        richprint(
            Panel.fit(
                f"[bold green]Optimization Results:[/bold green]\n\n"
                f"[cyan]Initial Prompt Performance:[/cyan]\n"
                f"{await self.calculate_scores(initial_test_results)}\n\n"
                f"[cyan]Optimized Prompt Performance:[/cyan]\n"
                f"{await self.calculate_scores(final_test_results)}",
                title="Final Report",
                border_style="bold",
            )
        )

        # Print prompt diff
        _print_rich_diff(task.initial_prompt, best_prompt, title="Final Prompt Updates")
        return best_prompt, best_score

    async def _evaluate_prompt(
        self, prompt: str, task: Task, data: str | list, debug: bool = False
    ) -> list[ExperimentResultRow]:
        """Evaluates a prompt against a task's dataset and evaluators."""

        async def predict(inputs: dict):
            return await task.system(prompt, inputs)

        results = await ls.aevaluate(
            predict,
            data=data,
            evaluators=task.evaluators,
            max_concurrency=0 if debug else None,
        )
        return [r async for r in results]

    async def calculate_scores(
        self, results: list[ExperimentResultRow]
    ) -> dict[str, float]:
        """Calculates aggregate scores from evaluation results, grouped by key."""

        scores = defaultdict(list)
        for result in results:
            for res in result["evaluation_results"]["results"]:
                if res.score is not None:
                    scores[res.key].append(res.score)

        return {
            key: sum(values) / len(values) if values else 0.0
            for key, values in scores.items()
        }

    async def apply_metaprompt(
        self,
        current_prompt: str,
        meta_prompt: str,
        task: Task,
        results: list[ExperimentResultRow],
        other_attempts: list | None = None,
    ) -> OptimizedPromptOutput:
        annotated_results = self._format_results(results)
        improved = await self._generate_improved_prompt(
            current_prompt,
            meta_prompt,
            annotated_results,
            task,
            other_attempts=other_attempts,
        )
        return improved

    def _format_results(self, results: list[ExperimentResultRow]) -> str:
        """Formats evaluation results for inclusion in the meta-prompt."""
        formatted = []
        i = 0
        for result in results:
            formatted.append(f"Example {i+1}:")
            formatted.append(f'Input: {result["run"].inputs}')
            formatted.append(f'Output: {result["run"].outputs}')
            formatted.append("Evaluations:")
            for eval in result["evaluation_results"]["results"]:
                formatted.append(f"- {eval.key}: {eval.score}")
                if eval.comment:
                    formatted.append(f"  Comment: {eval.comment}")
            formatted.append("")
            i += 1
        return "\n".join(formatted)

    async def _generate_improved_prompt(
        self,
        current_prompt: str,
        meta_prompt: str,
        annotated_results: str,
        task: Task,
        other_attempts: list | None = None,
    ) -> OptimizedPromptOutput:
        """Generates an improved prompt using the meta-prompt."""
        chain = self.model.with_structured_output(OptimizedPromptOutput)
        inputs = meta_prompt.format(
            current_prompt=current_prompt,
            annotated_results=annotated_results,
            task_description=task.describe(),
            other_attempts=(
                "\n\n---".join([p for p in other_attempts]) if other_attempts else "N/A"
            ),
        )
        prompt_output: OptimizedPromptOutput = await chain.ainvoke(inputs)

        _print_rich_diff(
            current_prompt, prompt_output.improved_prompt, "Updated Prompt"
        )

        return prompt_output

    async def _fetch_baseline_metrics(self, experiment_id: UUID) -> dict:
        """Fetches metrics for a baseline experiment."""
        # Implementation to fetch metrics from LangSmith using the experiment ID
        test_results = self.client.get_test_results(project_id=experiment_id)
        metric_cols = [
            col for col in test_results.columns if col.startswith("feedback.")
        ]
        return {col: test_results[col].mean() for col in metric_cols}


def _colorize_diff(diff):
    for op, i1, i2, j1, j2 in diff.get_opcodes():
        if op == "equal":
            yield diff.a[i1:i2]
        elif op == "insert":
            yield f"[green]{diff.b[j1:j2]}[/green]"
        elif op == "delete":
            yield f"[red]{diff.a[i1:i2]}[/red]"
        elif op == "replace":
            yield f"[red]{diff.a[i1:i2]}[/red][green]{diff.b[j1:j2]}[/green]"


def _print_rich_diff(original: str, updated: str, title: str = ""):
    diff = SequenceMatcher(None, original, updated)
    colorized_diff = "".join(_colorize_diff(diff))
    panel = Panel(
        colorized_diff, title=title or "Prompt Diff", expand=False, border_style="bold"
    )
    richprint(panel)
