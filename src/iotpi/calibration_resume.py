"""Append-only calibration transactions keyed by (unit type, family, index).

A units.jsonl row commits a whole unit before its raw projections are appended.
Replay reconstructs source pools/counts from saved outputs; no completed agent or
detector call is repeated. A torn projection is repaired by appending missing IDs.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .agents import AgentOutput
from .runlog import read_jsonl


def canonical(value):
    return json.loads(json.dumps(value,sort_keys=True))


def projection_id(key,stream,index):
    return hashlib.sha256(json.dumps([key,stream,index]).encode()).hexdigest()


class CalibrationJournal:
    def __init__(self,run):
        self.raw=run;self.dir=run.dir
        self._lock=(self.dir/'calibration.lock').open('a')
        try:fcntl.flock(self._lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close();run.close();raise RuntimeError('calibration directory already owned by another process')
        self.completed={};self.active=None;self.buffer=[];self.replaying=False
        self.cursors=defaultdict(int);self.saved=defaultdict(list);self.last_generation_id=None
        self.calibration_provenance={};self._generation_count=0;self.replayed_units=0
        self.shortfalls={}
        for row in read_jsonl(self.dir/'units.jsonl'):
            if 'unit_key' not in row or 'records' not in row:
                raise ValueError('legacy units require --resume-from into a new directory')
            key=tuple(row['unit_key'])
            if key in self.completed:raise ValueError('duplicate calibration unit key')
            self.completed[key]=row
            for stream,record in row['records']:
                if stream=='generations':self._generation_count=max(self._generation_count,record['generation_id']+1)
        for row in read_jsonl(self.dir/'source_shortfalls.jsonl'):
            self.shortfalls[(row['unit'],row['family'])]=row
        self.recover_projections()

    def log(self,text):self.raw.log(text)
    def save_json(self,name,value):self.raw.save_json(name,value)

    def recover_projections(self):
        expected=defaultdict(list)
        for key,unit in self.completed.items():
            indexes=defaultdict(int)
            for stream,row in unit['records']:
                uid=projection_id(key,stream,indexes[stream]);indexes[stream]+=1
                expected[stream].append((uid,row))
        for stream,records in expected.items():
            seen={r.get('calibration_record_id') for r in read_jsonl(self.dir/f'{stream}.jsonl')}
            for uid,row in records:
                if uid not in seen:
                    self.raw.record(stream,{**row,'calibration_record_id':uid});seen.add(uid)

    def begin_unit(self,unit,family,index):
        if self.active is not None:
            raise RuntimeError('previous calibration unit has no completion checkpoint')
        self.active=(unit,family,index);self.buffer=[];self.cursors=defaultdict(int)
        self.replaying=self.active in self.completed
        self.saved=defaultdict(list)
        if self.replaying:
            for stream,row in self.completed[self.active]['records']:self.saved[stream].append(row)

    def _consume(self,stream,row=None):
        index=self.cursors[stream]
        if index>=len(self.saved[stream]):raise ValueError(f'checkpoint missing {stream}: {self.active}')
        expected=self.saved[stream][index]
        if row is not None:
            # New checkpoint metadata may be absent from a legacy imported row.
            actual=canonical(row)
            if any(actual.get(k)!=v for k,v in canonical(expected).items() if k!='calibration_record_id'):
                raise ValueError(f'checkpoint replay differs for {stream}: {self.active}')
        self.cursors[stream]+=1
        return expected

    def replay_generation(self,inbox,memory,step,sc,node,role,seed):
        from dataclasses import asdict
        expected=self._consume('generations')
        inputs=[]
        for msg in inbox:
            value=asdict(msg);value['lineage']=sorted(msg.lineage);inputs.append(value)
        signature={'context':sc.context_id,'episode_seed':sc.seed,'node':node,'role':role,
                   'step':step,'generation_seed':seed,'inputs':inputs,'memory':memory}
        if any(canonical(v)!=canonical(expected[k]) for k,v in signature.items()):
            raise ValueError(f'changed calibration inputs or seeds: {self.active}')
        self.last_generation_id=expected['generation_id']
        out=AgentOutput(text=expected['output'],requests=[tuple(x) for x in expected['requests']],
            commands=[tuple(x) for x in expected['commands']],replies=[tuple(x) for x in expected['replies']],
            report=expected['report'],notice=expected['notice'],parse_ok=expected['parse_ok'],
            prompt_tokens=expected['prompt_tokens'],generation_tokens=expected['generation_tokens'],
            seconds=expected['seconds'],generation_meta=expected['generation_meta'])
        out.prompt=expected.get('prompt')
        return out

    def detector_result(self,name,msg):
        rows=self.saved['detector_measurements'];i=self.cursors['detector_measurements']
        if i>=len(rows):raise ValueError('checkpoint missing detector event')
        row=rows[i]
        if (row['detector'],row['kind'],row['text'])!=(name,msg.kind,msg.text):
            raise ValueError('changed calibration detector input/order')
        return row

    def record(self,stream,row):
        if stream=='source_shortfalls':
            key=(row['unit'],row['family'])
            if key in self.shortfalls:
                if canonical(self.shortfalls[key])!=canonical(row):raise ValueError('source capacity changed')
                return
            self.shortfalls[key]=row;self.raw.record(stream,row);return
        if self.active is None:raise RuntimeError(f'{stream} outside a calibration unit')
        if stream=='units':
            if self.replaying:
                stored=self.completed[self.active]['summary']
                actual=canonical(row)
                if any(actual.get(k)!=v for k,v in canonical(stored).items()):
                    raise ValueError(f'changed calibration outcome: {self.active}')
                if any(self.cursors[s]!=len(rows) for s,rows in self.saved.items()):
                    raise ValueError('checkpoint contains unconsumed raw events')
                self.replayed_units+=1
            else:
                checkpoint={**row,'unit_key':list(self.active),'index':self.active[2],
                            'family':self.active[1],'summary':row,'records':self.buffer}
                self.raw.record('units',checkpoint)
                self.completed[self.active]=checkpoint
                # Durable checkpoint exists before any raw projection is written.
                indexes=defaultdict(int)
                for kind,record in self.buffer:
                    uid=projection_id(self.active,kind,indexes[kind]);indexes[kind]+=1
                    self.raw.record(kind,{**record,'calibration_record_id':uid})
            self.active=None;self.replaying=False;self.buffer=[];return
        if self.replaying:self._consume(stream,row)
        else:self.buffer.append((stream,row))

    def close(self):
        self.raw.close();self._lock.close()


class ResumableDetector:
    def __init__(self,name,inner,journal):
        self.name,self.inner,self.journal=name,inner,journal
        self.observations=[]
    def flags(self,msg):
        if self.journal.replaying:
            row=self.journal.detector_result(self.name,msg)
            self.observations=[row['scoring']] if row.get('scoring') else []
            return row['flagged'],row['score']
        result=self.inner.flags(msg);self.observations=self.inner.observations;return result


def import_legacy(source,journal,units_per_cell,families):
    """Read-only migration from the pre-checkpoint format into a new run.

    Only units with a completed units.jsonl row are imported. Orphan detector or
    generation records are retained in the source and excluded from fitted counts.
    """
    source=Path(source)
    if source.resolve()==journal.dir.resolve():raise ValueError('legacy source must remain untouched')
    units=read_jsonl(source/'units.jsonl');generations=read_jsonl(source/'generations.jsonl')
    if any('records' in row for row in units):raise ValueError('use --resume for checkpoint-format runs')
    def signature(prov):
        return tuple(prov.get(k) for k in ('unit','source_id','root_source_id','receiving_context','reuse_index','memory_source_step'))
    mapping={};by_gen={};checkpoints=[];position=0;seen_keys=set()
    for unit in units:
        count=2 if unit['unit']=='benign' else 1
        group=generations[position:position+count];position+=count
        if len(group)!=count or group[0]['output']!=unit['output']:
            raise ValueError('legacy generation/unit completion order mismatch')
        if count==2 and group[1]['output']!=unit['coord_output']:raise ValueError('legacy benign pair mismatch')
        gen=group[0];kind=unit['unit'];family=gen['attack']['family'] if gen['attack'] else None
        offset=families.index(family)*units_per_cell if family else 0
        if kind=='gateway_raw':index=gen['episode_seed']-50_000-offset
        elif kind in ('coordinator_report','coordinator_persist'):index=gen['episode_seed']-60_000-offset
        elif kind=='gateway_notice':index=gen['episode_seed']-70_000-offset
        elif kind=='gateway_persist':
            index=(gen['episode_seed']-50_000-offset if gen['episode_seed']<60_000
                   else units_per_cell+gen['episode_seed']-70_000-offset)
        elif kind=='benign':index=gen['episode_seed']-90_000
        else:raise ValueError('unknown legacy unit')
        key=(kind,family,index)
        if key in seen_keys or not 0<=index<2*units_per_cell:raise ValueError('invalid/duplicate legacy unit key')
        seen_keys.add(key)
        records=[('generations',g) for g in group]
        checkpoint={**unit,'unit_key':list(key),'index':index,'family':family,'summary':unit,'records':records}
        checkpoints.append(checkpoint)
        for g in group:
            if g['calibration_provenance']['unit']!=kind:raise ValueError('legacy unit type mismatch')
            mapping[signature(g['calibration_provenance'])]=checkpoint
            by_gen[g['generation_id']]=checkpoint
    ignored=defaultdict(int)
    for stream in ('agent_measurements','detector_measurements','source_exposures'):
        for row in read_jsonl(source/f'{stream}.jsonl'):
            checkpoint=by_gen.get(row['generation_id']) if stream=='agent_measurements' else mapping.get(signature(row))
            if checkpoint is None:ignored[stream]+=1;continue
            checkpoint['records'].append((stream,row))
    for checkpoint in checkpoints:
        journal.raw.record('units',checkpoint)
        journal.completed[tuple(checkpoint['unit_key'])]=checkpoint
    for row in read_jsonl(source/'source_shortfalls.jsonl'):journal.record('source_shortfalls',row)
    journal._generation_count=max(by_gen,default=-1)+1
    journal.recover_projections()
    return {'source':str(source),'completed_units_imported':len(checkpoints),
            'completed_generations_imported':len(by_gen),'uncommitted_generations_ignored':len(generations)-position,
            'uncommitted_rows_ignored':dict(ignored)}
