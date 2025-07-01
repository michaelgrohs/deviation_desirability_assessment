from math import ceil

from process_atoms.mine.declare.checkers.choice import mp_choice, mp_exclusive_choice
from process_atoms.mine.declare.checkers.existence import (
    mp_absence,
    mp_end,
    mp_exactly,
    mp_existence,
    mp_init,
)
from process_atoms.mine.declare.checkers.negative_relation import (
    mp_not_chain_precedence,
    mp_not_chain_response,
    mp_not_precedence,
    mp_not_responded_existence,
    mp_not_response,
)
from process_atoms.mine.declare.checkers.relation import (
    mp_alternate_precedence,
    mp_alternate_response,
    mp_chain_precedence,
    mp_chain_response,
    mp_precedence,
    mp_responded_existence,
    mp_response,
)
from process_atoms.mine.declare.enums.mp_constants import Template, TraceState
from process_atoms.mine.declare.models.checker_result import CheckerResult
from process_atoms.mine.declare.models.decl_model import DeclModel
from process_atoms.models.event_log import EventLog


def check_trace_conformance(trace, model, consider_vacuity):
    """
    Check the conformance of a trace with a model.
    TODO currently trace is an activity sequence only (no event)
    """
    rules = {"vacuous_satisfaction": consider_vacuity}

    # Set containing all constraints that raised SyntaxError in checker functions
    error_constraint_set = set()

    trace_results = {}

    for idx, constraint in enumerate(model.constraints):
        constraint_str = model.serialized_constraints[idx]
        rules["activation"] = constraint["condition"][0]

        if constraint["template"].supports_cardinality:
            rules["n"] = constraint["n"]
        if constraint["template"].is_binary:
            rules["correlation"] = constraint["condition"][1]

        rules["time"] = constraint["condition"][
            -1
        ]  # time condition is always at last position

        try:
            if constraint["template"] is Template.EXISTENCE:
                trace_results[constraint_str] = mp_existence(
                    trace, True, constraint["activities"][0], rules
                )

            elif constraint["template"] is Template.ABSENCE:
                trace_results[constraint_str] = mp_absence(
                    trace, True, constraint["activities"][0], rules
                )

            elif constraint["template"] is Template.INIT:
                trace_results[constraint_str] = mp_init(
                    trace, True, constraint["activities"][0], rules
                )

            elif constraint["template"] is Template.END:
                trace_results[constraint_str] = mp_end(
                    trace, True, constraint["activities"][0], rules
                )

            elif constraint["template"] is Template.EXACTLY:
                trace_results[constraint_str] = mp_exactly(
                    trace, True, constraint["activities"][0], rules
                )

            elif constraint["template"] is Template.CHOICE:
                trace_results[constraint_str] = mp_choice(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.EXCLUSIVE_CHOICE:
                trace_results[constraint_str] = mp_exclusive_choice(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.RESPONDED_EXISTENCE:
                trace_results[constraint_str] = mp_responded_existence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.RESPONSE:
                trace_results[constraint_str] = mp_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.ALTERNATE_RESPONSE:
                trace_results[constraint_str] = mp_alternate_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.CHAIN_RESPONSE:
                trace_results[constraint_str] = mp_chain_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.PRECEDENCE:
                trace_results[constraint_str] = mp_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.ALTERNATE_PRECEDENCE:
                trace_results[constraint_str] = mp_alternate_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.CHAIN_PRECEDENCE:
                trace_results[constraint_str] = mp_chain_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
            elif constraint["template"] is Template.SUCCESSION:
                trace_results_response = mp_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                trace_results_precedence = mp_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                res = CheckerResult(
                    num_fulfillments=trace_results_response.num_fulfillments,
                    num_violations=trace_results_response.num_violations,
                    num_pendings=None,
                    num_activations=trace_results_response.num_activations,
                    state=TraceState.VIOLATED
                    if trace_results_response.state == TraceState.VIOLATED
                    or trace_results_precedence.state == TraceState.VIOLATED
                    else TraceState.SATISFIED,
                )
                trace_results[constraint_str] = res

            elif constraint["template"] is Template.ALTERNATE_SUCCESSION:
                trace_results_response = mp_alternate_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                trace_results_precedence = mp_alternate_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                res = CheckerResult(
                    num_fulfillments=trace_results_response.num_fulfillments,
                    num_violations=trace_results_response.num_violations,
                    num_pendings=None,
                    num_activations=trace_results_response.num_activations,
                    state=TraceState.VIOLATED
                    if trace_results_response.state == TraceState.VIOLATED
                    or trace_results_precedence.state == TraceState.VIOLATED
                    else TraceState.SATISFIED,
                )
                trace_results[constraint_str] = res

            elif constraint["template"] is Template.CHAIN_SUCCESSION:
                trace_results_response = mp_chain_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                trace_results_precedence = mp_chain_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
                res = CheckerResult(
                    num_fulfillments=trace_results_response.num_fulfillments,
                    num_violations=trace_results_response.num_violations,
                    num_pendings=None,
                    num_activations=trace_results_response.num_activations,
                    state=TraceState.VIOLATED
                    if trace_results_response.state == TraceState.VIOLATED
                    or trace_results_precedence.state == TraceState.VIOLATED
                    else TraceState.SATISFIED,
                )
                trace_results[constraint_str] = res

            elif constraint["template"] is Template.NOT_RESPONDED_EXISTENCE:
                trace_results[constraint_str] = mp_not_responded_existence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
            elif constraint["template"] is Template.NOT_CO_EXISTENCE:
                trace_results[constraint_str] = mp_not_responded_existence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.NOT_RESPONSE:
                trace_results[constraint_str] = mp_not_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.NOT_CHAIN_RESPONSE:
                trace_results[constraint_str] = mp_not_chain_response(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.NOT_PRECEDENCE:
                trace_results[constraint_str] = mp_not_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )

            elif constraint["template"] is Template.NOT_CHAIN_PRECEDENCE:
                trace_results[constraint_str] = mp_not_chain_precedence(
                    trace,
                    True,
                    constraint["activities"][0],
                    constraint["activities"][1],
                    rules,
                )
        except SyntaxError:
            if constraint_str not in error_constraint_set:
                error_constraint_set.add(constraint_str)
                print(
                    'Condition not properly formatted for constraint "'
                    + constraint_str
                    + '".'
                )

    return trace_results


def discover_constraint(log: EventLog, constraint, consider_vacuity):
    # Fake model composed by a single constraint
    model = DeclModel()
    model.constraints.append(constraint)
    model.set_constraints()
    discovery_res = {}

    for i, trace in enumerate(log):
        if len(trace.events) == 0:
            continue
        trc_res = check_trace_conformance(
            trace.get_activity_sequence(), model, consider_vacuity
        )
        if not trc_res:  # Occurring when constraint data conditions are formatted bad
            break

        constraint_str, checker_res = next(
            iter(trc_res.items())
        )  # trc_res will always have only one element inside
        if checker_res.state == TraceState.SATISFIED:
            new_val = {i: checker_res}
            if constraint_str in discovery_res:
                discovery_res[constraint_str] |= new_val
            else:
                discovery_res[constraint_str] = new_val

    return discovery_res


def query_constraint(log, constraint, consider_vacuity, min_support):
    # Fake model composed by a single constraint
    model = DeclModel()
    model.constraints.append(constraint)

    sat_ctr = 0
    for i, trace in enumerate(log):
        trc_res = check_trace_conformance(
            trace.get_activity_sequence(), model, consider_vacuity
        )
        if not trc_res:  # Occurring when constraint data conditions are formatted bad
            break

        constraint_str, checker_res = next(
            iter(trc_res.items())
        )  # trc_res will always have only one element inside
        if checker_res.state == TraceState.SATISFIED:
            sat_ctr += 1
            # If the constraint is already above the minimum support, return it directly
            if sat_ctr / len(log) >= min_support:
                return constraint_str
        # If there aren't enough more traces to reach the minimum support, return nothing
        if len(log) - (i + 1) < ceil(len(log) * min_support) - sat_ctr:
            return None

    return None
