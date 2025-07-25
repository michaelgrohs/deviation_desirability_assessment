"""
All code in module is based upon Adrian Rebmann's codebase.
"""

import re

NON_ALPHANUM = re.compile("[^a-zA-Z]")
CAMEL_PATTERN_1 = re.compile("(.)([A-Z][a-z]+)")
CAMEL_PATTERN_2 = re.compile("([a-z0-9])([A-Z])")


class Sanitizer:
    def __init__(self, sanitize) -> None:
        self.sanitize = sanitize

    def sanitize_label(self, label):
        # replace [] with ()
        label = label.replace("[", "(").replace("]", ")")
        # reduce whitespaces / linebreaks / tabs to one whitespace
        label = re.sub(r"\s{1,}", " ", label)
        return label

    def sanitize_label_complete(self, label):
        if not self.sanitize:
            return label
        label = str(label)
        if " - " in label:
            label = label.split(" - ")[-1]
        if "&" in label:
            label = label.replace("&", "and")
        label = label.replace("\n", " ").replace("\r", "")
        label = label.replace("(s)", "s")
        label = label.replace("'", "")
        label = re.sub(" +", " ", label)

        label = NON_ALPHANUM.sub(" ", label)
        label = label.strip()
        label = " ".join([part for part in label.split() if len(part) > 1])

        label = self.__camel_to_white(label)

        label = label.lower()

        label = re.sub(r"\s{1,}", " ", label)
        return label

    def __camel_to_white(self, label):
        label = CAMEL_PATTERN_1.sub(r"\1 \2", label)
        return CAMEL_PATTERN_2.sub(r"\1 \2", label)
