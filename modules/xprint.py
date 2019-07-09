"""Module containing the xprint function"""

################################################################################
# IMPORTS
################################################################################


import os
import sys
from datetime import datetime


################################################################################
# FUNCTIONS
################################################################################


def xprint(*args, **kwargs):
    """Wraps the normal print function and prints to stdout and optionally to a log file

    Takes optional keyword argument prefix, which will not be passed on to print.
    """
    # Compile text
    if 'prefix' in kwargs:
        xprint.prefix = kwargs['prefix']
        del kwargs['prefix']
    kwargs['flush'] = kwargs.get('flush', xprint.flush_by_default)
    sep = kwargs.get('sep', ' ')
    text = ""
    if xprint.prefix:
        text += xprint.prefix_function()
    for idx, arg in enumerate(args):
        text += sep + str(arg) if idx > 0 else str(arg)

    # Print text to stdout
    print(text, **kwargs)

    # Fetch end of line
    end = kwargs['end'] if 'end' in kwargs else '\n'

    # Print text to log file
    if xprint.log_file and not xprint.log_file.closed:
        xprint.log_file.write(text + end)
        xprint.log_file.flush()

    # Prepare next printout
    xprint.prefix = end.rstrip(' \t\r').endswith('\n')


def _create_log_file_from_script_path(script_path, log_file_dir=None):
    script_name = os.path.splitext(os.path.basename(script_path))[0]
    if log_file_dir is None:
        log_file_dir = os.path.abspath(os.path.dirname(script_path))
    file_name = ("logfile_" + script_name + "_"
                 + str(datetime.now().replace(microsecond=0)).replace(' ', '_')
                 .replace(':', ".")
                 + ".txt")
    xprint.log_file = open(os.path.join(log_file_dir, file_name), 'w')


def _indent():
    xprint.indentation += 1


def _unindent():
    if xprint.indentation <= 0:
        raise IndentationError("Attempting to unindent when there is no"
                               " indentation")
    xprint.indentation -= 1


# Initialize xprint "methods"
xprint.create_log_file_from_script_path = _create_log_file_from_script_path
xprint.indent = _indent
xprint.unindent = _unindent

# Initialize xprint state variables
xprint.prefix = True
xprint.prefix_function = lambda: (
        str(datetime.now().replace(microsecond=0)) + ":  " +
        xprint.indentation * xprint.indentation_length * ' '
)
xprint.log_file = None
xprint.flush_by_default = True
xprint.indentation_length = 4
xprint.indentation = 0


def eprint(*args, **kwargs):
    kwargs['file'] = sys.stderr
    xprint(*args, **kwargs)
