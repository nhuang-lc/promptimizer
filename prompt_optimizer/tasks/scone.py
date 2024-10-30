from typing import Literal

from pydantic import BaseModel, Field
from prompt_optimizer.trainer import Task
from langchain_openai import ChatOpenAI


def exact_match(run, example):
    """Evaluate the exact match correctness of the NLI result."""
    try:
        predicted = run.outputs["is_entailed"]
        expected = example.outputs["answer"]
        score = expected.lower() == predicted.lower()
    except Exception:
        try:
            expected = example.outputs["answer"]
            expected_bool = {"no": False, "yes": True}.get(expected.strip().lower())
            score = run.outputs["output"].is_entailed == expected_bool
        except Exception:
            score = 0
    return {
        "key": "exact_match",
        "score": int(score),
    }


predictor_llm = ChatOpenAI(model="gpt-4o-mini")


class SubmitOutput(BaseModel):
    reasoning: str = Field(
        description="First, think step-by-step to determine the correct answer before submitting your response."
    )
    is_entailed: Literal["Yes", "No"]


async def scone_system(prompt: str, inputs: dict):
    extracted = await predictor_llm.with_structured_output(SubmitOutput).ainvoke(
        [
            ("system", prompt),
            (
                "user",
                "Context: {context}\n\nStatement: {question}\n\nIs this statement entailed? Answer 'Yes' or 'No'".format(
                    **inputs
                ),
            ),
        ]
    )
    return extracted.model_dump(mode="json")


scone_task = Task(
    name="Scone (NLI)",
    train_dataset_name="scone-train2",
    dev_dataset_name="scone-dev2",
    test_dataset_name="scone-test-one-scoped",
    initial_prompt="""Determine if the statement is entailed by the context. Answer 'Yes' if entailed, 'No' otherwise.""",
    evaluators=[exact_match],
    evaluator_descriptions={
        "exact_match": "Directly compares the expected against the predicted outputs. 1 if correct, 0 if incorrect."
    },
    system=scone_system,
)
