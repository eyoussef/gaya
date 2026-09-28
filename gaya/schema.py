CHOICE = "choice"
SCORE = "score"
BOOL = "bool"
QTYPES = (CHOICE, SCORE, BOOL)


def choice(qid, instructions, criteria):
    """criteria: {label: description} — answer space defined at request time."""
    return {"id": qid, "type": CHOICE, "instructions": instructions, "criteria": criteria}


def score(qid, instructions, criteria):
    """criteria: ordered list from lowest to highest — ordinal."""
    return {"id": qid, "type": SCORE, "instructions": instructions,
            "criteria": {str(i): c for i, c in enumerate(criteria)}}


def boolean(qid, instructions):
    """Neutral anchors: option labels carry no yes/no semantics, so the model must read
    the state (laya's noul can follow its own true:/false: labels — issue #156)."""
    return {"id": qid, "type": BOOL, "instructions": instructions,
            "criteria": {"A": "the answer is yes: " + instructions,
                         "B": "the answer is no: " + instructions}}


def _criteria_pairs(q):
    crit = q.get("criteria")
    if isinstance(crit, dict):
        return list(crit.items())
    if isinstance(crit, list):
        return [(str(i), c) for i, c in enumerate(crit)]
    # bool without explicit criteria: neutral anchors
    return [("A", "the answer is yes: " + q["instructions"]),
            ("B", "the answer is no: " + q["instructions"])]


def option_texts(q):
    return [f"{q['instructions']}\n{label}: {desc}" for label, desc in _criteria_pairs(q)]


def option_labels(q):
    return [label for label, _ in _criteria_pairs(q)]


def question_prompt(q):
    lines = [f"Question id: {q['id']} (type: {q['type']})", q["instructions"], "Options:"]
    for lab, d in _criteria_pairs(q):
        lines.append(f"- {lab}: {d}")
    return "\n".join(lines)