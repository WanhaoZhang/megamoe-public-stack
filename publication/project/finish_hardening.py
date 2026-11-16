"""After frozen pilots finish, harden duplicate completion and requalify NPU math."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'diagnostics/hardening.json'


def save(state):STATE.write_text(json.dumps(state,indent=2)+'\n')


def main():
    if STATE.exists():raise RuntimeError('Preserve hardening registration')
    state=dict(stage='waiting_for_frozen_pilot',reason='Do not change proof-covered core during registered25-update runs')
    save(state)
    for _ in range(240):
        try:pilot=json.loads((ROOT/'lifetime-b2.json').read_text())
        except json.JSONDecodeError:time.sleep(30);continue
        if pilot['stage'] in ('screening_complete','stopped_at_gate'):break
        time.sleep(30)
    else:raise TimeoutError('Pilot waiting limit; no automatic core mutation')
    if pilot['stage']!='screening_complete':raise RuntimeError('Pilot failed; preserve its core before manual recovery')
    source=ROOT/'stack/tools/step_lifetime.py'
    text=source.read_text(encoding='utf-8')
    old='''        """Retire only transfers whose completion has actually been observed."""
        tickets = [self.tickets[key] for key in identities]'''
    new='''        """Retire only transfers whose completion has actually been observed."""
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate completion identity')
        tickets = [self.tickets[key] for key in identities]'''
    if text.count(old)!=1:raise RuntimeError('Completion contract source changed; do not blind-patch')
    source.write_text(text.replace(old,new),encoding='utf-8',newline='\n')
    tests=ROOT/'stack/tools/test_step_lifetime.py'
    text=tests.read_text(encoding='utf-8')
    marker="\n\nif __name__ == '__main__':"
    addition='''
    def test_completion_batch_rejects_duplicates_atomically(self):
        self.context.contribute(('a',), self.key, 0)
        self.context.submit((('a',),))
        with self.assertRaises(ValueError):
            self.context.complete((('a',), ('a',)))
        self.assertFalse(self.context.tickets[('a',)].complete)
        self.context.complete((('a',),))
        self.context.finish()
'''
    if text.count(marker)!=1:raise RuntimeError('Contract test source changed')
    tests.write_text(text.replace(marker,'\n'+addition+marker),encoding='utf-8',newline='\n')
    state['stage']='cpu_contract';save(state)
    with (ROOT/'diagnostics/hardening-cpu.log').open('wb') as log:
        subprocess.run([sys.executable,str(tests)],check=True,stdout=log,stderr=subprocess.STDOUT)
    with (ROOT/'diagnostics/duplicate-completion-after.json').open('wb') as log:
        subprocess.run([sys.executable,str(ROOT/'implementation/check_duplicate_completion.py')],check=True,
                       stdout=log,stderr=subprocess.STDOUT)
    state['stage']='publishing_guard';save(state)
    with (ROOT/'diagnostics/hardening-publish.log').open('wb') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'publish_b.py'),
            'tools/step_lifetime.py','tools/test_step_lifetime.py'],check=True,stdout=log,stderr=subprocess.STDOUT)
    state['stage']='independent_npu_requalification';save(state)
    with (ROOT/'diagnostics/hardening-npu.log').open('wb') as log:
        subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'run_independent_gate.py'),
            '--tag','lifetime-independent-ffn-v4'],check=True,stdout=log,stderr=subprocess.STDOUT)
    state.update(stage='complete',passed=True,performance_data_source='original frozen pilot, prior v3 certificate retained',
        scope='reject duplicate completion before retirement; six CPU contracts and64forced-replica NPU comparisons')
    save(state)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state=json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(stage='failed',error=str(error));save(state);raise
