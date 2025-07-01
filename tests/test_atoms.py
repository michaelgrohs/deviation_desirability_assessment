import os
import time
from zipfile import ZipFile

import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

from process_atoms.constants import BPMN2_NAMESPACE
from process_atoms.match.matcher import (
    build_faiss_index,
    get_embeddings,
    match_activities_based_on_index,
)
from process_atoms.mine.declare.enums.mp_constants import Template
from process_atoms.models.column_types import (
    CaseID,
    Categorical,
    Continuous,
    EventTime,
    EventType,
)
from process_atoms.models.event_log import EventLog, EventLogSchemaTypes
from process_atoms.processatoms import ProcessAtoms
from tests.constants import MODEL_ID, MODEL_JSON

considered_templates = [
    Template.RESPONSE.templ_str,
    Template.PRECEDENCE.templ_str,
    Template.SUCCESSION.templ_str,
    Template.RESPONDED_EXISTENCE.templ_str,
    Template.ALTERNATE_RESPONSE.templ_str,
    Template.ALTERNATE_PRECEDENCE.templ_str,
    Template.CHAIN_RESPONSE.templ_str,
    Template.CHAIN_PRECEDENCE.templ_str,
    Template.NOT_SUCCESSION.templ_str,
    Template.EXCLUSIVE_CHOICE.templ_str,
]


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    for file in os.listdir("./data/"):
        if file.endswith(".zip"):
            with ZipFile("./data/" + file, "r") as zip_ref:
                zip_ref.extractall("./data/")


def test_log_mining():
    api = ProcessAtoms()

    # Mine atoms from the log
    atoms = api.transform_bpmn_to_atoms_with_petri(
        MODEL_ID, MODEL_JSON, considered_templates
    )
    # Check the mined atoms
    assert len(atoms) > 0

    schema = EventLogSchemaTypes(
        # schema for case-level attributes
        cases={
            "Case ID": CaseID,
            "(case) Company": Categorical,
            "(case) Document Type": Categorical,
            "(case) GR-Based Inv. Verif.": Categorical,
            "(case) Goods Receipt": Categorical,
            "(case) Item": Categorical,
            "(case) Item Category": Categorical,
            "(case) Item Type": Categorical,
            "(case) Name": Categorical,
            "(case) Purch. Doc. Category name": Categorical,
            "(case) Purchasing Document": Categorical,
            "(case) Source": Categorical,
            "(case) Spend area text": Categorical,
            "(case) Spend classification text": Categorical,
            "(case) Sub spend area text": Categorical,
            "(case) Vendor": Categorical,
        },
        events={
            "Case ID": CaseID,
            "Activity": EventType,
            "Complete Timestamp": EventTime,
            "Resource": Categorical,
            "Cumulative net worth (EUR)": Continuous,
        },
    )
    LOG_PATH = "data/BPI_Challenge_2019.csv"
    PROCESS = "BPIC_19"
    # read the full log
    log = pd.read_csv(LOG_PATH, parse_dates=["Complete Timestamp"])
    # split into case and event attributes
    df_cases = log[list(schema.cases.keys())].drop_duplicates(subset="Case ID")
    df_events = log[list(schema.events.keys())]

    print(
        df_events[df_events["Activity"] == "Record Service Entry Sheet"][
            "Case ID"
        ].nunique()
    )

    # create event log object
    log = EventLog(df_cases, df_events, schema)
    # Create the Process Atoms API
    api = ProcessAtoms()
    # Mine atoms from the log
    start_time = time.time()
    atoms = api.mine_atoms_from_log(
        PROCESS,
        log,
        considered_templates,
        consider_vacuity=False,
        local=True,
        d4py=False,
        min_support=(0.1 * len(log)) / len(log),
    )
    end_time = time.time()

    delta = end_time - start_time

    print(f"Atom mining took {delta} seconds.")


def test_atom_gen():
    api = ProcessAtoms()
    # Convert the model to a log
    atoms = api.transform_bpmn_to_atoms_with_petri(
        MODEL_ID, MODEL_JSON, considered_templates
    )
    assert len(atoms) > 0


def test_path_gen_exessively():
    model_json_df = pd.read_csv("./data/bestpractices.csv", sep=",")
    model_json_df = model_json_df[model_json_df["Namespace"] == BPMN2_NAMESPACE]
    for _, row in model_json_df.iterrows():
        model_id = row["Model ID"]
        model_json = row["Model JSON"]
        api = ProcessAtoms()
        # Convert the model to a log
        api.transform_bpmn_to_atoms_with_petri(
            model_id, model_json, considered_templates
        )
        break


def test_end_to_end():
    BPMN2_NAMESPACE = "http://b3mn.org/stencilset/bpmn2.0#"
    model_json_df = pd.read_csv("./data/bestpractices.csv", sep=",")
    model_json_df = model_json_df[model_json_df["Namespace"] == BPMN2_NAMESPACE]
    api = ProcessAtoms()
    process_atoms = []
    for _, row in model_json_df.iterrows():
        process_atoms.extend(
            api.transform_bpmn_to_atoms_with_petri(
                row["Model ID"], row["Model JSON"], considered_templates
            )
        )
        break
    aggregated = api.aggregate_atoms(process_atoms)

    schema = EventLogSchemaTypes(
        # schema for case-level attributes
        cases={
            "Case ID": CaseID,
            "(case) Company": Categorical,
            "(case) Document Type": Categorical,
            "(case) GR-Based Inv. Verif.": Categorical,
            "(case) Goods Receipt": Categorical,
            "(case) Item": Categorical,
            "(case) Item Category": Categorical,
            "(case) Item Type": Categorical,
            "(case) Name": Categorical,
            "(case) Purch. Doc. Category name": Categorical,
            "(case) Purchasing Document": Categorical,
            "(case) Source": Categorical,
            "(case) Spend area text": Categorical,
            "(case) Spend classification text": Categorical,
            "(case) Sub spend area text": Categorical,
            "(case) Vendor": Categorical,
        },
        events={
            "Case ID": CaseID,
            "Activity": EventType,
            "Complete Timestamp": EventTime,
            "Resource": Categorical,
            "Cumulative net worth (EUR)": Continuous,
        },
    )
    LOG_PATH = "data/BPI_Challenge_2019.csv"
    PROCESS = "BPIC_19"
    # read the full log
    log = pd.read_csv(LOG_PATH, parse_dates=["Complete Timestamp"])
    # split into case and event attributes
    df_cases = log[list(schema.cases.keys())].drop_duplicates(subset="Case ID")
    df_events = log[list(schema.events.keys())]

    # create event log object
    eventlog = EventLog(df_cases, df_events, schema)
    sent_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")

    atom_activities = list(
        {operand for atom in aggregated for operand in atom.operands}
    )
    log_activities = eventlog.unique_activities()

    def in_matches(log, act):
        return act in matches and log in matches[act]

    activity_embeddings = get_embeddings(sent_model, atom_activities)
    index = build_faiss_index(sent_model, atom_activities, activity_embeddings)
    matches = match_activities_based_on_index(
        index, sent_model, atom_activities, log_activities, k=1
    )
    print(matches)
    fitted_atoms = api.fit_atoms_to_log(
        process=PROCESS,
        event_log=eventlog,
        process_atoms=aggregated,
        matching_function=in_matches,
        instantiate_for_log=True,
        partial_instantiation=False,
    )
    api.check_atom_violations(
        process=PROCESS, event_log=eventlog, process_atoms=fitted_atoms
    )


def test_model_mining_petri():
    api = ProcessAtoms()
    # Convert the model to a log
    atoms = api.transform_bpmn_to_atoms_with_petri(
        MODEL_ID, MODEL_JSON, considered_templates=considered_templates
    )
    assert len(atoms) > 0
