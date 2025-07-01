from datetime import timedelta

from process_atoms.mine.declare.enums.mp_constants import TraceState
from process_atoms.mine.declare.models.checker_result import CheckerResult
from process_atoms.mine.declare.parsers.decl_parser import (
    parse_data_cond,
    parse_time_cond,
)

# Defining global and local functions/variables to use within eval() to prevent code injection
glob = {"__builtins__": None}


# mp-not-responded-existence constraint checker
# Description:
def mp_not_responded_existence(trace, done, a, b, rules):
    activation_rules = parse_data_cond(rules["activation"])
    correlation_rules = parse_data_cond(rules["correlation"])
    time_rule = parse_time_cond(rules["time"])

    pendings = []
    num_fulfillments = 0
    num_violations = 0
    num_pendings = 0

    for event in trace:
        if event == a:
            locl = {"A": event}
            if eval(activation_rules, glob, locl):
                pendings.append(event)

    for event in trace:
        if not pendings:
            break

        if event == b:
            for A in reversed(pendings):
                locl = {
                    "A": A,
                    "T": event,
                    "timedelta": timedelta,
                    "abs": abs,
                    "float": float,
                }
                if eval(correlation_rules, glob, locl) and eval(time_rule, glob, locl):
                    pendings.remove(A)
                    num_violations += 1

    if done:
        num_fulfillments = len(pendings)
    else:
        num_pendings = len(pendings)

    num_activations = num_fulfillments + num_violations + num_pendings
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=num_pendings,
        num_activations=num_activations,
        state=state,
    )


# mp-not-response constraint checker
# Description:
def mp_not_response(trace, done, a, b, rules):
    activation_rules = parse_data_cond(rules["activation"])
    correlation_rules = parse_data_cond(rules["correlation"])
    time_rule = parse_time_cond(rules["time"])

    pendings = []
    num_fulfillments = 0
    num_violations = 0
    num_pendings = 0

    for event in trace:
        if event == a:
            locl = {"A": event}
            if eval(activation_rules, glob, locl):
                pendings.append(event)

        if pendings and event == b:
            for A in reversed(pendings):
                locl = {
                    "A": A,
                    "T": event,
                    "timedelta": timedelta,
                    "abs": abs,
                    "float": float,
                }
                if eval(correlation_rules, glob, locl) and eval(time_rule, glob, locl):
                    pendings.remove(A)
                    num_violations += 1

    if done:
        num_fulfillments = len(pendings)
    else:
        num_pendings = len(pendings)

    num_activations = num_fulfillments + num_violations + num_pendings
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=num_pendings,
        num_activations=num_activations,
        state=state,
    )


# mp-not-chain-response constraint checker
# Description:
def mp_not_chain_response(trace, done, a, b, rules):
    activation_rules = parse_data_cond(rules["activation"])
    correlation_rules = parse_data_cond(rules["correlation"])
    time_rule = parse_time_cond(rules["time"])

    num_activations = 0
    num_violations = 0
    num_pendings = 0

    for index, event in enumerate(trace):
        if event == a:
            locl = {"A": event}

            if eval(activation_rules, glob, locl):
                num_activations += 1

                if index < len(trace) - 1:
                    if trace[index + 1] == b:
                        locl = {
                            "A": event,
                            "T": trace[index + 1],
                            "timedelta": timedelta,
                            "abs": abs,
                            "float": float,
                        }
                        if eval(correlation_rules, glob, locl) and eval(
                            time_rule, glob, locl
                        ):
                            num_violations += 1
                else:
                    if not done:
                        num_pendings = 1

    num_fulfillments = num_activations - num_violations - num_pendings
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=num_pendings,
        num_activations=num_activations,
        state=state,
    )


# mp-not-precedence constraint checker
# Description:
def mp_not_precedence(trace, done, a, b, rules):
    activation_rules = parse_data_cond(rules["activation"])
    correlation_rules = parse_data_cond(rules["correlation"])
    time_rule = parse_time_cond(rules["time"])

    num_activations = 0
    num_violations = 0
    Ts = []

    for event in trace:
        if event == a:
            Ts.append(event)

        if event == b:
            locl = {"A": event}

            if eval(activation_rules, glob, locl):
                num_activations += 1

                for T in Ts:
                    locl = {
                        "A": event,
                        "T": T,
                        "timedelta": timedelta,
                        "abs": abs,
                        "float": float,
                    }
                    if eval(correlation_rules, glob, locl) and eval(
                        time_rule, glob, locl
                    ):
                        num_violations += 1
                        break

    num_fulfillments = num_activations - num_violations
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=None,
        num_activations=num_activations,
        state=state,
    )


# mp-not-chain-precedence constraint checker
# Description:
def mp_not_chain_precedence(trace, done, a, b, rules):
    activation_rules = parse_data_cond(rules["activation"])
    correlation_rules = parse_data_cond(rules["correlation"])
    time_rule = parse_time_cond(rules["time"])

    num_activations = 0
    num_violations = 0

    for index, event in enumerate(trace):
        if event == b:
            locl = {"A": event}

            if eval(activation_rules, glob, locl):
                num_activations += 1

                if index != 0 and trace[index - 1] == a:
                    locl = {
                        "A": event,
                        "T": trace[index - 1],
                        "timedelta": timedelta,
                        "abs": abs,
                        "float": float,
                    }
                    if eval(correlation_rules, glob, locl) and eval(
                        time_rule, glob, locl
                    ):
                        num_violations += 1

    num_fulfillments = num_activations - num_violations
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=None,
        num_activations=num_activations,
        state=state,
    )


def mp_not_succession(trace, done, a, b, rules):
    num_fulfillments = 0
    num_violations = 0
    bs = []
    onward = []
    for idx, event in enumerate(trace):
        if event == b:
            bs.append(event)
    for idx, event in enumerate(trace):
        if event == a:
            onward = trace[idx:]
            break
    for event in onward:
        if event == b and len(bs) > 0:
            num_violations += 1
            break
    if num_violations == 0 and len(onward) > 0:
        num_fulfillments = 1

    num_activations = num_fulfillments + num_violations
    vacuous_satisfaction = rules["vacuous_satisfaction"]
    state = None

    if not vacuous_satisfaction and num_activations == 0:
        if done:
            state = TraceState.VIOLATED
        else:
            state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_violations == 0:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_violations > 0:
        state = TraceState.VIOLATED
    elif done and num_violations == 0:
        state = TraceState.SATISFIED

    return CheckerResult(
        num_fulfillments=num_fulfillments,
        num_violations=num_violations,
        num_pendings=None,
        num_activations=num_activations,
        state=state,
    )
