"""Isolated online-frontier source; share the study's device admission lock."""
import argparse
import subprocess
import remote

remote.REMOTE = '/home/z00956592/megaflux/temporal-replica-frontier-20261115'
remote.CONTAINER = 'temporal-replica-frontier-20261115'
parser = argparse.ArgumentParser()
parser.add_argument('action',choices=('deploy','gate'))
parser.add_argument('--tag')
parser.add_argument('--metadata-only',action='store_true')
args,extra = parser.parse_known_args()
if args.action == 'deploy':
    remote.deploy()
    controller = remote.ROOT/'controller.py'
    contents = controller.read_text().replace("(TASK / 'evidence' / 'device-jobs.lock')",
        "pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock')")
    controller.write_text(contents,encoding='utf-8',newline='\n')
    subprocess.run(['scp',str(controller),f'{remote.HOST}:{remote.REMOTE}/stack/tools/controller.py'],check=True)
else:
    if not args.tag or not all(value.isalnum() or value in '-_.' for value in args.tag):
        parser.error('a safe tag is required')
    if any(not all(value.isalnum() or value in '-_./:' for value in item) for item in extra):
        parser.error('unexpected shell metacharacter')
    remote.gate(args.tag,extra,metadata_only=args.metadata_only)
