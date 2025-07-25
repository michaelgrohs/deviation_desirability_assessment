import re
from collections import deque

from process_atoms.constants import (
    DATA_OBJECT,
    ELEMENT_CATEGORY,
    ELEMENT_ID,
    GLOSSARY_LINKS,
    LABEL,
)


def camel_to_white(label):
    label = CAMEL_PATTERN_1.sub(r"\1 \2", label)
    return CAMEL_PATTERN_2.sub(r"\1 \2", label)


NON_ALPHANUM = re.compile("[^a-zA-Z]")
CAMEL_PATTERN_1 = re.compile("(.)([A-Z][a-z]+)")
CAMEL_PATTERN_2 = re.compile("([a-z0-9])([A-Z])")


def replace_multiple_substrings(
    text: str, substrings: list[str], replacement: str
) -> str:
    """
    Replaces all occurrences of multiple substrings in the text with the given replacement string.

    :param text: The original text.
    :param substrings: A list of substrings to be replaced.
    :param replacement: The string to replace the substrings with.
    :return: Modified text with replacements.
    """
    for substring in substrings:
        text = text.replace(substring, replacement)
    return text


def sanitize_label(label):
    # handle some special cases
    label = str(label)
    if "&" in label:
        label = label.replace("&", "and")
    label = label.replace("\n", " ").replace("\r", "")
    label = label.replace("(s)", "s")
    label = label.replace("'", "")
    label = re.sub(" +", " ", label)
    label = label.strip()
    # handle camel case
    label = camel_to_white(label)
    # delete unnecessary whitespaces
    label = re.sub("\s{1,}", " ", label)
    return label


def sanitize_label_full(label):
    # handle some special cases
    label = sanitize_label(label)
    # remove non-alphanumeric characters
    label = NON_ALPHANUM.sub("", label)
    return label


def compute_finite_paths_of_tasks(follows, labels, tasks):
    node_paths = compute_finite_paths_with_node_ids(follows, labels)
    task_paths = []
    for path in node_paths:
        task_path = [node for node in path if node in tasks]
        # ensure task_path is not empty and has not been seen before
        if task_path and task_path not in task_paths:
            task_paths.append(task_path)
    return task_paths


def compute_finite_paths_with_node_ids(follows, labels):
    print_info = True

    # Find source and sink shapes (typically events)
    source_shapes = set()
    sink_shapes = set()
    for s in follows.keys():
        # Iterate over all shapes except sequence flows
        irrelevant_shapes = ("SequenceFlow", "DataObject", "Pool", "Lane")
        if not labels[s].startswith(irrelevant_shapes):
            if len(get_postset(labels, follows, s)) == 0:
                sink_shapes.add(s)
            if len(get_preset(labels, follows, s)) == 0:
                source_shapes.add(s)

    # Print source and sink shapes
    if print_info:
        print()
        print("Source and sink shapes:")
        print([labels[s] for s in source_shapes])
        print([labels[s] for s in sink_shapes])

    # Get all finite paths from start to end shapes
    finite_paths = []
    for s1 in source_shapes:
        for s2 in sink_shapes:
            if s1 != s2:
                finite_paths = [
                    *finite_paths,
                    *get_possible_paths(labels, follows, s1, s2, []),
                ]

    # Print all finite paths
    if print_info:
        print()
        print("All finite paths:")
        for p in finite_paths:
            print([labels[s] for s in p])
    return finite_paths


def get_possible_paths(labels, follows, s1, s2, path=[]):
    # Returns all possible paths from s1 to s2 as a list of lists
    postset = get_postset(labels, follows, s1)
    # if target (s2) is in postset, add current and target shape and return
    if s2 in postset:
        path.append(s1)
        path.append(s2)
        return [path]
    # if no shapes in postset, return empty list
    if len(postset) == 0:
        return []
    # Several shapes in postset ...
    else:
        path.append(s1)
        # Determine shapes to be visited (make sure that we don't visit the same shape again and get stuck
        to_be_visited = postset.difference(set(path))
        # If no shape is left, return empty list
        if len(to_be_visited) == 0:
            return []
        else:
            paths = []
            if labels[s1].startswith("ParallelGateway"):
                print("\t" + str([labels[x] for x in to_be_visited]))
            # Recursively traverse
            for s in to_be_visited:
                recursive_paths = get_possible_paths(
                    labels, follows, s, s2, path.copy()
                )
                if len(recursive_paths) > 0:
                    if isinstance(recursive_paths[0], list):
                        for p in recursive_paths:
                            paths.append(p)
                    else:
                        paths.append(recursive_paths)
            return paths


def get_transitive_postset(labels, follows, shape, visited_shapes):
    # Returns all shapes in the postset of a shape. Note that these might
    # include all shapes if the model contains loops.
    # Obtain all shapes  in the postset of considered shape
    transitive_post_set = get_postset(labels, follows, shape)
    # Determine which transitions still need to be visited
    to_be_visited = transitive_post_set.difference(visited_shapes)
    # Update visited shapes
    visited_shapes.update(to_be_visited)
    if len(to_be_visited) == 0:
        return set()
    else:
        # Recursively build transitive postset
        for s in to_be_visited:
            recursive_result = get_transitive_postset(
                labels, follows, s, visited_shapes
            )
            transitive_post_set.update(recursive_result)
            visited_shapes.update(recursive_result)
        return transitive_post_set


def get_postset(labels, follows, shape):
    # Note: The direct postset of a shape typically only contains the arc, not another element.
    # Exceptions are attached events. Both is handled properly.
    postset = set()
    direct_postset = set(follows[shape])
    for s in direct_postset:
        # Ignore message flows
        if labels[s].startswith("MessageFlow"):
            continue
        if not labels[s].startswith("SequenceFlow"):
            postset.add(s)
        else:
            postset.update(follows[s])
    return postset


def get_preset(labels, follows, shape):
    # Note: The direct preset of a shape typically only contains the arc, not another element.
    # Exceptions are attached events. Both is handled properly.
    preset = set()
    for s1 in follows.keys():
        if s1 != shape and shape in follows[s1]:
            if not labels[s1].startswith("MessageFlow"):
                if not labels[s1].startswith("SequenceFlow"):
                    preset.add(s1)
                else:
                    for s2 in follows.keys():
                        if s2 != s1 and s1 in follows[s2]:
                            preset.add(s2)
    return preset


def get_full_postset(labels, follows, shape):
    # Note: The direct postset of a shape typically only contains the arc, not another element.
    # Exceptions are attached events. Both is handled properly.
    postset = set()
    direct_postset = set(follows[shape])
    for s in direct_postset:
        if (
            s in labels
            and not labels[s].startswith("MessageFlow")
            and not labels[s].startswith("SequenceFlow")
            and not labels[s].startswith("Association")
        ):
            postset.add(s)
        else:
            if s in follows:
                postset.update(follows[s])
    return postset


def process_bpmn_shapes(shapes):
    follows = {}
    labels = {}
    tasks = set()

    # Analyze shape list and store all shapes and activities
    # PLEASE NOTE: the code below ignores BPMN sub processes
    for shape in shapes:
        # Save all shapes to dict
        # print(shape['stencil']['id'], shape)

        # If current shape is a pool or a lane, we have to go a level deeper
        if shape["stencil"]["id"] == "Pool" or shape["stencil"]["id"] == "Lane":
            result = process_bpmn_shapes(shape["childShapes"])
            follows.update(result[0])
            labels.update(result[1])
            tasks.update(result[2])

        shapeID = shape["resourceId"]
        outgoingShapes = [s["resourceId"] for s in shape["outgoing"]]
        if shapeID not in follows:
            follows[shapeID] = outgoingShapes

        # Save all tasks and respective labels separately
        if shape["stencil"]["id"] == "Task":
            if not shape["properties"]["name"] == "":
                tasks.add(shape["resourceId"])
                labels[shape["resourceId"]] = (
                    shape["properties"]["name"]
                    .replace("\n", " ")
                    .replace("\r", "")
                    .replace("  ", " ")
                )
            else:
                labels[shape["resourceId"]] = "Task"
        else:
            if "name" in shape["properties"] and not shape["properties"]["name"] == "":
                labels[shape["resourceId"]] = (
                    shape["stencil"]["id"]
                    + " ("
                    + shape["properties"]["name"]
                    .replace("\n", " ")
                    .replace("\r", "")
                    .replace("  ", " ")
                    + ")"
                )
            else:
                labels[shape["resourceId"]] = shape["stencil"]["id"]
    return follows, labels, tasks


def fromJSON(model_dict):
    follows, labels, tasks = process_bpmn_shapes(model_dict["childShapes"])
    return follows, labels, tasks


def is_relevant(shape, labels, irrelevant_shapes):
    label = labels[shape]
    first_word = label.split(" ", 1)[0]
    return first_word not in irrelevant_shapes


def is_choice(shape, labels):
    return (
        labels[shape].startswith("Exclusive")
        or labels[shape].startswith("Inclusive")
        or labels[shape].startswith("Eventbased")
    )


def get_type(shape, labels, irrelevant_shapes):
    label = labels[shape]
    first_word = label.split(" ", 1)[0]
    if (
        first_word.endswith("Event")
        or first_word.endswith("EventCatching")
        or first_word.endswith("EventThrowing")
    ):
        return "Event"
    if first_word.endswith("Gateway"):
        return "Gateway"
    if first_word not in irrelevant_shapes:
        return "Task"
    return first_word


def is_relevant_label(task_name):
    terms = {"Message"}
    if task_name is None:
        return False
    if task_name == "":
        return False
    if task_name.isnumeric():
        return False
    if task_name in terms:
        return False
    if "Gateway" in task_name:
        return False
    if task_name.startswith("EventSubprocess") or task_name.startswith("Subprocess"):
        return False
    return True


def _traverse_and_extract_data_object_relations(follows, labels):
    data_objects = {}
    for s in follows.keys():
        if is_relevant(s, labels, ()):
            postset = get_full_postset(labels, follows, s)
            for elem in postset:
                if elem not in follows.keys():
                    continue
                ty = get_type(elem, labels, ())
                if ty == "Object":
                    if s in data_objects:
                        data_objects[s].append(elem)
                    else:
                        data_objects[s] = list()
                        data_objects[s].append(elem)
    return data_objects


def parse_model_elements(
    model_id: str,
    model_dict: dict,
    parse_parent: bool = False,
    parse_outgoing: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Parses the recursive childShapes and produces a flat list of model elements with the most important attributes
    such as id, category, label, outgoing, and parent elements.
    """

    elements_flat = {}
    follows = {}
    labels = {}
    model_flow, model_labels, t = fromJSON(model_dict)
    for model_label in model_labels:
        labels[str(model_id) + str(model_label)] = model_labels[model_label]
    for model_follow in model_flow:
        follows[str(model_id) + str(model_follow)] = [
            str(model_id) + str(elm) for elm in model_flow[model_follow]
        ]
    stack = deque([model_dict])
    data_object_relations = _traverse_and_extract_data_object_relations(follows, labels)
    while len(stack) > 0:
        element = stack.pop()
        for c in element.get("childShapes", []):
            c["parent"] = element["resourceId"]
            stack.append(c)
        # don't append root as element
        if element["resourceId"] == model_dict["resourceId"]:
            continue
        element_id = str(model_id) + str(element["resourceId"])
        # NOTE: it's possible to add other attributes here, such as the bounds of an element
        record = {
            ELEMENT_ID: element_id,
            ELEMENT_CATEGORY: element["stencil"].get("id")
            if "stencil" in element
            else None,
            LABEL: sanitize_label(element["properties"].get("name")),
            GLOSSARY_LINKS: element["glossaryLinks"]
            if "glossaryLinks" in element
            else {},
            DATA_OBJECT: data_object_relations[element_id]
            if element_id in data_object_relations
            else [],
        }
        if parse_parent:
            record["parent"] = element.get("parent")
        if parse_outgoing:
            record["outgoing"] = [
                v for d in element.get("outgoing", []) for v in d.values()
            ]

        elements_flat[element_id] = record

    return elements_flat, follows, labels
