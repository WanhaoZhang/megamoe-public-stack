"""Offline task-order diagnostic, never an online scheduling policy."""
import hashlib
import json
from pathlib import Path
from temporal_scheduler import TaskKey,WindowPlan

class DiagnosticScheduleReplay:
    def __init__(self,path,shape):
        payload=Path(path).read_bytes();data=json.loads(payload)
        if data.get('role')!='offline_task_selection_diagnostic' or data.get('version')!=1:
            raise ValueError('Task replay must be explicitly labelled offline diagnostic')
        if data['model_shape']!=shape:raise ValueError('Task replay model shape differs')
        self.sha256=hashlib.sha256(payload).hexdigest()
        self.source_tag=data['source_tag'];self.window=data['window'];self.steps=data['steps']
        self.records={}
        for row in data['groups']:
            identity=(row['step'],row['group'])
            if identity in self.records:raise ValueError('Duplicate replay group')
            for field in ('ready_keys','keys'):
                keys=row[field]
                if not keys or any(len(key)!=3 or any(type(value)!=int or value<0 for value in key) for key in keys):
                    raise ValueError('Invalid replay task identity')
                if len({tuple(key) for key in keys})!=len(keys):raise ValueError('Duplicate replay task')
                if any(key[0]!=row['step'] for key in keys):raise ValueError('Mixed optimizer versions in replay')
            if not {tuple(key) for key in row['keys']}.issubset({tuple(key) for key in row['ready_keys']}):
                raise ValueError('Selected task was not ready in source trace')
            self.records[identity]=row
    def select(self,ready,group):
        actual=[[key.step,key.microbatch,key.layer] for key in ready]
        if not actual:raise ValueError('Replay requires a ready task')
        row=self.records[(ready[0].step,group)]
        if row['ready_keys']!=actual:raise RuntimeError('Offline replay ready frontier differs from source trace')
        lookup={tuple(key):task for key,task in zip(actual,ready)}
        selected=tuple(lookup[tuple(key)] for key in row['keys'])
        return WindowPlan(selected,tuple(key for key in ready if key not in selected),None,None)
