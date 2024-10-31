import functools
import logging

from langchain_openai import ChatOpenAI
from prompt_optimizer.trainer import Task
from pydantic import BaseModel, Field
from trustcall import create_extractor

logger = logging.getLogger(__name__)


class Grade(BaseModel):
    """Call to submit your grade."""

    reasoning: str = Field(
        description="First, explain your thought process on why you are giving the provided grade."
    )
    score: int = Field(
        ge=0, le=5, description="Then, submit your score on a scale from 0 to 5."
    )

    @property
    def normalized(self):
        return self.score / 5


judge = create_extractor(
    ChatOpenAI(model="gpt-4o-mini"), tools=[Grade], tool_choice=Grade.__name__
)
utemplate = """Grade the following:
Predicted: {predicted}
Reference example: {reference}"""


async def summary_quality(run, example):
    predicted = run.outputs.get("summary")
    rubric = """Grade the quality of summary. If it fails any criteria, give a 0. If it's perfect, give a 5.
Criteria:
- Must not include idle words like "the email is about X"
- Preferred format is <person> from <org> needs/wants X
"""
    reference = example.outputs["summary"]
    result = await judge.ainvoke(
        [
            ("system", rubric),
            ("user", utemplate.format(predicted=predicted, reference=reference)),
        ]
    )
    grade: Grade = result["responses"][0]
    pf = "Pass" if grade.score >= 4 else "Fail"
    return {"score": grade.normalized, "comment": f"{pf}: {grade.reasoning}"}


def accuracy_check(run, example, key: str):
    predicted = run.outputs.get(key)
    reference = example.outputs.get(key)
    if reference is None:
        return {
            "key": f"{key}-correctness",
            "comment": "Skipping - reference label not found.",
        }
    score = (
        predicted == reference
        if not isinstance(reference, list)
        else predicted in reference
    )
    pf = "Pass" if score else "Fail"
    return {
        "key": f"{key}-correctness",
        "score": score,
        "comment": f"{pf}",
    }  #: Expected {reference}. Got: {predicted}. Why did you get this wrong? Think deeply and update associations."}


classifiers = [
    functools.partial(accuracy_check, key=key)
    for key in [
        "category",
        "support_category",
        "ticket_status",
        "requires_response",
        "non_support_category",
    ]
]
evaluators = [summary_quality, *classifiers]

schema = {
    "name": "categorizeEmail",
    "description": "Category for email classification.",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "description": "5-10 word title or summary for the issue/error/or other content.",
                "type": "string",
            },
            "category": {
                "description": "Top-level category of the email.",
                "allOf": [
                    {
                        "title": "EmailCategory",
                        "description": "Top-level categories for email classification.",
                        "enum": [
                            "technical_support",
                            "security_reports",
                            "billing_support",
                            "inbound_leads",
                            "spam",
                            "other",
                        ],
                        "type": "string",
                    }
                ],
            },
            "non_support_category": {
                "description": "If the email is not technical support, specify the subcategory (sales_leads, marketing_seo_promotional, other_spam, other).\nExamples would be if the top level category is inbound_leads, this might be sales_leads or security_reports. Or if the top level category is spam, this might be marketing_seo_promotional. Or if the request is for self-hosting, this might be sales_leads.\nIf the email is about lawsuits or legal issues, use legal_correspondance.",
                "allOf": [
                    {
                        "title": "NonSupportCategory",
                        "description": "Subcategories for non-support emails.",
                        "enum": [
                            "sales_leads",
                            "compliance_reports",
                            "marketing_seo_promotional",
                            "legal_correspondance",
                            "other_spam",
                            "other",
                        ],
                        "type": "string",
                    }
                ],
            },
            "support_category": {
                "description": "If the email is support-related, specify the subcategory.",
                "allOf": [
                    {
                        "title": "SupportCategory",
                        "description": "Product area (for sub-categorization of support emails).",
                        "enum": [
                            "langchain_py_open_source",
                            "langchain_js_open_source",
                            "langgraph_py_open_source",
                            "langgraph_js_open_source",
                            "langsmith_privacy",
                            "langsmith_application",
                            "langsmith_account_billing",
                            "langsmith_data_deletion",
                            "langsmith_account_management",
                            "langsmith_sdk",
                            "langsmith_prompt_hub",
                            "langserve_oss",
                            "langmem",
                        ],
                        "type": "string",
                    }
                ],
            },
            "organization": {
                "description": "The company or organization the sender represents. Should be someone outside the LangChain org.",
                "type": "string",
            },
            "is_paying_customer": {
                "description": "Whether the email is from a paying customer (based on context). Only applicable for langchain_development and langsmith_sdk categories.",
                "type": "boolean",
            },
            "sentiment": {
                "description": "The sentiment of the sender. We need to respond quickly to paying customers with negative experiences.",
                "default": "neutral",
                "enum": ["negative", "neutral", "positive"],
                "type": "string",
            },
            "requires_response": {
                "description": "Whether the email requires a response from the support team. Yes if our support or sales team should send a follow-up message.  No if the message body is empty, if it's clear the email is an automated one (e.g., a noreply email), if it's spam,  if the user's issue has been resolved already, or if a member of the LangChain team has already responded.",
                "type": "boolean",
            },
            "ticket_status": {
                "description": "The status we should set the ticket to. Typically 'open' unless the response confirms the issue is resolved (closed), or if it's spam (closed) or if it's not something we can do right now (on_hold).",
                "enum": ["open", "waiting_on_customer", "on_hold", "closed"],
                "type": "string",
            },
            "priority": {
                "description": "The priority of the ticket. Urgent is for issues that need immediate attention - either impact a paying customer or all customers.High is for issues that need to be resolved quickly. Medium is for issues that should be resolved within a few days.Low is for issues that can be resolved when time permits.",
                "enum": ["low", "medium", "high", "urgent"],
                "type": "string",
            },
        },
        "required": ["summary", "category", "requires_response"],
    },
}

extractor = create_extractor(
    ChatOpenAI(model="gpt-4o-mini"), tools=[schema], tool_choice="categorizeEmail"
)


async def ticket_classification_system(prompt: str, inputs: dict):
    response = await extractor.ainvoke(prompt.format(input=inputs))
    return response["responses"][0].model_dump(mode="json")


ticket_classification_task = Task(
    name="Ticket Classification",
    description="A task to classify customer support tickets",
    train_dataset_name="ticket-classification-train",
    dev_dataset_name="ticket-classification-dev",
    test_dataset_name="ticket-classification-test",
    initial_prompt="""Classify the following customer support ticket:
{input}""",
    evaluators=evaluators,
    evaluator_descriptions={
        "summary_quality": "Evaluates the quality of the summary",
        "category-correctness": "Checks if the category is correct",
        "support_category-correctness": "Checks if the support category is correct",
        "ticket_status-correctness": "Checks if the ticket status is correct",
        "requires_response-correctness": "Checks if the requires_response field is correct",
        "non_support_category-correctness": "Checks if the non-support category is correct",
    },
    system=ticket_classification_system,
)

if __name__ == "__main__":
    import random

    import langsmith as ls

    c = ls.Client()
    examples = list(
        c.list_examples(dataset_name="customer-support-bot.test_extraction")
    )

    random.shuffle(examples)
    full = examples.copy()
    train, dev, test = [], [], []
    for ds, size, name in zip(
        [train, dev, test], [41, 20, 20], ["train", "dev", "test"]
    ):
        for i in range(size):
            ds.append(full.pop())
        dname = f"ticket-classification-{name}"
        try:
            dataset = c.create_dataset(dataset_name=dname)
        except Exception:
            c.delete_dataset(dataset_name=dname)
            dataset = c.create_dataset(dataset_name=dname)

        outputs = [e.outputs for e in ds]
        for o in outputs:
            for k, v in o.pop("outputs", {}).items():
                o[k] = v
        c.create_examples(
            inputs=[e.inputs for e in ds],
            outputs=outputs,
            dataset_id=dataset.id,
        )
