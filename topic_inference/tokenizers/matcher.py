from topic_inference.utils import readpickle, writepickle
from copy import deepcopy


def process_match(match):
    """compute the beginning and ending index"""
    end_index, value = match
    original_value = value[0]
    start_index = end_index - len(original_value) + 1
    return (start_index, end_index + 1), value


def is_token(text, interval):
    """check if an interval is not a subword"""
    if interval[0] > 0:
        char_left = text[interval[0] - 1]
        if char_left.isalnum():
            return False
    if interval[1] < len(text) - 1:
        # this fixes cases where a plural is formed with a trailing 's'
        chars_right = text[interval[1]:interval[1] + 2]
        if chars_right[0].lower() == 's' and not chars_right[1].isalnum():
            return True
    if interval[1] < len(text):
        char_right = text[interval[1]]
        if char_right.isalnum():
            return False
    return True


def overlap(intervals):
    """find the subset of intervals that do not overlap or intersect"""
    intervals = sorted(intervals)

    # remove the intervals that are contained in other intervals
    contained = []
    for i in range(0, len(intervals) - 1):
        for j in range(i + 1, len(intervals)):
            if intervals[j][0] >= intervals[i][0] and \
                    intervals[j][1] <= intervals[i][1]:
                contained.append(intervals[j])
            if intervals[i][0] >= intervals[j][0] and intervals[i][1] <= \
                    intervals[j][1]:
                contained.append(intervals[i])

    for o in contained:
        if o in intervals:
            intervals.remove(o)

    # remove intervals that intersect other intervals
    # the rule here is to pick the longest, if the intervals have the same size
    # then pick the first being mentioned
    intersected = []
    for i in range(0, len(intervals) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            len0 = intervals[i][1] - intervals[i][0]
            len1 = intervals[i + 1][1] - intervals[i + 1][0]
            p = intervals[i] if len0 < len1 else intervals[i + 1]
            intersected.append(p)

    for o in intersected:
        if o in intervals:
            intervals.remove(o)

    return intervals


def mask_matches(text, d, mask=' __match__{} '):
    """apply a mask on the captured intervals"""
    for match in list(reversed(list(d.keys()))):
        text = text[:match[0]] + mask.format(
            d[match][1].replace(' ', '_')) + text[match[1] + 1:]
    return text


class Matcher:

    def __init__(self, automaton, mask=' __match__{} ', remove_overlaps=True,
                 remove_subwords=True):
        self.automaton = automaton
        self.mask = mask
        self.remove_overlaps = remove_overlaps
        self.remove_subwords = remove_subwords

    def save(self, filepath):
        if not self.automaton:
            raise TypeError("Can't save automaton. It does not exist.")
        writepickle(self.automaton, filepath)

    def load(self, filepath):
        self.automaton = readpickle(filepath)

    def copy(self):
        return deepcopy(self)

    def get_intervals(self, text, remove_overlaps=None):
        intervals = self.automaton.iter(text)
        intervals = {k: v for k, v in [process_match(o) for o in intervals]}

        # remove the intervals that are subwords
        # the .get method is used to guarantee backwards compatibility
        if self.__dict__.get('remove_subwords', True):
            intervals = {o: intervals[o] for o in intervals.keys()
                         if is_token(text, o)}

        # select the intervals that do not overlap
        remove_overlaps = remove_overlaps or self.remove_overlaps
        if remove_overlaps:
            intervals = {o: intervals[o] for o in overlap(intervals.keys())}
        return intervals

    def transform(self, text):
        intervals = self.get_intervals(text, remove_overlaps=True)
        text = mask_matches(text, intervals, self.mask)
        return text

    def tokenize(self, text, return_values=False):
        """
        Matches a single input text to the automaton patterns.

        :param text: A string potentially containing patterns to be discovered
         by the automaton.
        :type text: str
        :param return_values: If true, returns a list of tuples where each tuple
         contains the token and other data that might have been added to the
          automaton (custom only).
           If false, returns a list of tokens as strings.
        :type return_values: bool
        :return: A list of tokens.
        :rtype: list
        """
        intervals = self.get_intervals(text)
        return [o if return_values else o[0] for o in intervals.values()]
