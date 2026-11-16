"""Explicit host/container ownership and dependency paths shared by launchers."""
from pathlib import Path
import re
import shlex

FIELDS = ('host-root', 'container-root', 'mount-root', 'container', 'container-id',
          'task-label', 'cann-env', 'fsdp-root', 'moonep-root', 'shmem-root')


def add_arguments(parser):
    for name in FIELDS:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--mount-target', default='/project')
    parser.add_argument('--allow-warning-alarm', action='store_true',
                        help='Explicitly permit idle Warning/Alarm devices; occupancy checks still apply')


def forwarded(args):
    result = []
    for key in (*FIELDS, 'mount-target'):
        result += ['--' + key, getattr(args, key.replace('-', '_'))]
    if args.allow_warning_alarm:
        result += ['--allow-warning-alarm']
    return result


def validate_output_name(name):
    if (not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_/-]*', name)
            or '..' in Path(name).parts):
        raise ValueError('invalid relative output name')


def shell_command(args, argv):
    return 'source ' + shlex.quote(args.cann_env) + ' && exec ' + shlex.join(argv)
