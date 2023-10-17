from typing import List, Dict, Union
import re


class PatternGeneralizer:
    def __init__(self, patterns: List[Dict[str, Union[str, str]]]):
        """
        Initialize the PatternGeneralizer with a list of dictionaries.
        Each dictionary must contain:
            - 'column': The column name where the pattern should be applied
            - 'pattern': The regex pattern to match
            - 'replacement': The string to replace the pattern with
        """
        self.patterns = patterns

    def transform_row(self, row):
        for pattern_dict in self.patterns:
            column = pattern_dict['column']
            pattern = pattern_dict['pattern']
            replacement = pattern_dict['replacement']

            if isinstance(row[column], str):
                row[column] = re.sub(pattern, replacement, row[column])

        return row

