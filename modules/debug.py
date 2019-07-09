

################################################################################
# IMPORTS
################################################################################


import re


################################################################################
# FUNCTIONS
################################################################################


def get_attributes(o, skip_private=True, max_leading_underscores=None):
    attributes = {}
    if skip_private:
        max_leading_underscores = 0
    pattern = '^' + '_' * (max_leading_underscores + 1) if max_leading_underscores is not None else None
    for name in dir(o):
        if max_leading_underscores is not None and re.search(pattern, name):
            continue
        attributes[name] = getattr(o, name)
    return attributes
