# Templates hold (question, answer, query) triplets

SEP = "\n##\n"  # separator between few-shot examples

TEMPLATES = {
    # Binary classification tasks
    "piqa": (
        "Goal: {goal}\n",
        "Answer: {}",
        "Answer:"
    ),
    "superglue-wic": (
        "{sentence1}\n{sentence2}\nquestion: Is the word '{word}' used in the same sense in the two sentences above?\n",
        "answer: {}",
        "answer:"
    ),
    "superglue-boolq": (
        "{passage}\nquestion: {question}?\n",
        "answer: {}",
        "answer:"
    ),
    "superglue-cb": (
        "{premise}\nquestion: {hypothesis}. true, false, or neither?\n",
        "answer: {}",
        "answer:"
    ),
    "superglue-wsc": (
        "Passage: {text}\nQuestion: In the passage above, does the pronoun '{span2_text}' refer to {span1_text}?\n",
        "Answer: {}",
        "Answer:"
    ),
    "superglue-rte": (
        "{premise}\nquestion: {hypothesis}. true or false?\n",
        "answer: {}",
        "answer:"
    ),

    # Multi-choice tasks
    "ai2_arc_easy": (
        "Question: {question}\n",
        "Answer: {}",
        "Answer:"
    ),
    "ai2_arc_challenge": (
        "Question: {question}\n",
        "Answer: {}",
        "Answer:"
    ),
    "hellaswag": (
        "Context: {context}\n",
        "Answer: {}",
        "Answer:"
    ),
    "openbookqa": (
        "Context: {context}\n",
        "Answer: {}",
        "Answer:"
    ),
}


def apply_template(dp, dataset) -> dict:
    # Binary
    if dataset == "superglue-wic":
        splits = dp["input"].split(" [SEP] ")
        dp["input"] = TEMPLATES[dataset][0].format(sentence1=splits[0], sentence2=splits[1], word=splits[2])
        dp["output"] = TEMPLATES[dataset][1].format(dp["output"])

    elif dataset == "superglue-boolq":
        splits = dp["input"].split(" [SEP] ")
        dp["input"] = TEMPLATES[dataset][0].format(passage=splits[0], question=splits[1])
        dp["output"] = TEMPLATES[dataset][1].format(dp["output"])

    elif dataset == "superglue-cb":
        splits = dp["input"].split(" [SEP] ")
        dp["input"] = TEMPLATES[dataset][0].format(premise=splits[0], hypothesis=splits[1])
        dp["output"] = TEMPLATES[dataset][1].format(dp["output"])

    elif dataset == "superglue-wsc":
        splits = dp["input"].split(" [SEP] ")
        dp["input"] = TEMPLATES[dataset][0].format(text=splits[0], span2_text=splits[2], span1_text=splits[1])
        dp["output"] = TEMPLATES[dataset][1].format(dp["output"])

    elif dataset == "superglue-rte":
        splits = dp["input"].split(" [SEP] ")
        dp["input"] = TEMPLATES[dataset][0].format(premise=splits[0], hypothesis=splits[1])
        dp["output"] = TEMPLATES[dataset][1].format(dp["output"])

    # Multi-choice
    elif dataset == "piqa":
        dp["input"] = TEMPLATES[dataset][0].format(goal=dp["input"])

    elif dataset == "ai2_arc_easy":
        dp["input"] = TEMPLATES[dataset][0].format(question=dp["input"])

    elif dataset == "ai2_arc_challenge":
        dp["input"] = TEMPLATES[dataset][0].format(question=dp["input"])

    elif dataset == "hellaswag":
        dp["input"] = TEMPLATES[dataset][0].format(context=dp["input"])

    elif dataset == "openbookqa":
        dp["input"] = TEMPLATES[dataset][0].format(context=dp["input"])

    else:
        raise NotImplementedError(dataset)
