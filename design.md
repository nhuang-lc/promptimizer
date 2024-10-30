# Design Notes

Something like this:
```python
@dataclass
class Task:
    """Represents a specific task for prompt optimization."""

    name: str
    train_datasets: str
    dev_dataset_name: str
    test_dataset_name: str
    evaluators: list[Callable[[Run, Example], dict]] 
    initial_prompt: str
    system: Runnable # Prompt + LLM most likely, where the prompt
    max_iterations: int = 1
    baseline_experiment: UUID | None
```

Trainer loop:

1. If baseline_experiment is set, fetch metrics for that, otherwise, run current prompt on dev dataset and get baseline scores. 
2. Train:
```python
- list examples in train dataset
for x in epochs:
    - Shuffle examples list & truncate to train size (default None) & split into batches
    - For batch in batches: 
        - Run aevaluate on batch
        - Format feedback
        - Use current metaprompt to update the task-specific prompt
```
3. Test
    - Run initial/baseline prompt on test dataset and get scores
    - Run baseline prompt on test dataset and get scores
    - Print out comparisons



Then for optimizing the metaprompt, this itself can be framed as a task,but the langsmith dataset would be kinda more of a reference than actual values....

The metaprompt task would be like:
- train dataset: each example's inputs is the name of a sub-task(?) and subset of the train dataset?
- system would be some object that:
initializes with a map of the train dataset to the task objects for the tasks it's trying to optimize on.
prelim - loads a batch of the subtask-specific examples based on the subset in that example. Looks up evaluators, etc. for this task from the map on the object.
the evaluator looks at the run outputs
1. Run initial task prompt over that batch
2. Run task evals on the results
3. Run metaprompt to generate new prompt
4. Run task evals on the new results
5. return a dict of {original: ..., new: ..., original_scores: ..., new_scores: ....}
6. eavluator takes those outputs and combines into a singel score. Compare the results and assert they are monitonically improving. And/Or could run a preference evaluator LLM-as-judge.
    so basically this task's evaluator would be 1 if better, 0.5 if same, 0 if worse (or something; this is just an example)