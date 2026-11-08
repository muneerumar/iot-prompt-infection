"""Sequential offline Llama calibration/pilot/prediction with immutable evidence.

No Qwen, scale experiment, or paper writing. Run only after the audited design is frozen.
"""
from __future__ import annotations
import argparse,json,math,os,re,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iotpi.runlog import Run,read_jsonl
from iotpi.calibrate import required_sample_sizes
from workflow_support import (DOCS,analyse_pilot,cell,commit,evidence,gate_one,gate_two,gate_three,
                              rel,report,sha,tune_cache)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,row))+' |' for row in rows])

def select_owned_run(candidates, expected_name, known=None):
    """Identify this command's artifact; unrelated result directories are allowed."""
    candidates = ([Path(known)] if known is not None else []) + list(candidates)
    matches = set()
    for directory in candidates:
        meta = directory / 'meta.json'
        if not directory.is_dir() or not meta.exists():
            continue
        try:
            name = json.loads(meta.read_text()).get('name')
        except json.JSONDecodeError:
            continue
        if name == expected_name:
            matches.add(directory)
    if known is not None and Path(known) in matches:
        return Path(known)
    if len(matches) != 1:
        raise RuntimeError(f'cannot identify owned run {expected_name}: {sorted(map(str,matches))}')
    return next(iter(matches))


def owned_run_identity(script,arguments):
    if '--resume' in arguments:
        directory=Path(arguments[arguments.index('--resume')+1])
        return json.loads((directory/'meta.json').read_text())['name'],directory
    name=arguments[arguments.index('--name')+1] if '--name' in arguments else 'quarantine_probe_llama'
    if script in ['run_calibration.py','run_pilot.py']:
        name+='_'+arguments[arguments.index('--model')+1]
    return name,None


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--timing',required=True)
    parser.add_argument('--calibration')
    parser.add_argument('--pilot')
    parser.add_argument('--prediction')
    parser.add_argument('--probe')
    parser.add_argument('--handoff-copy')
    parser.add_argument('--aborted-calibration',action='append',default=[])
    parser.add_argument('--interruption-evidence')
    args=parser.parse_args(argv)
    manifest=json.loads(Path(args.freeze).read_text())
    timing=json.loads(Path(args.timing).read_text())
    workflow=Run(ROOT,'workflow_llama',{'freeze':args.freeze,'timing':args.timing,'controller_pid':os.getpid(),'aborted_calibrations':args.aborted_calibration,'interruption_evidence':args.interruption_evidence})
    paths={k:Path(v) for k,v in vars(args).items() if k in ['calibration','pilot','prediction','probe'] and v}
    gates={'gate1':{'verdict':'not reached'},'gate2':{'verdict':'not reached'},'gate3':{'verdict':'not reached'}}
    issues=[];active=None;active_process=None;started=time.monotonic();failed=False

    def verify():
        for name,digest in manifest['protected_files'].items():
            path=ROOT/name
            if name=='configs/default.yaml':
                current=yaml.safe_load(path.read_text());original=manifest['original_config']
                permitted=json.loads(json.dumps(current));permitted['pilot'].pop('cache_distance',None)
                original=json.loads(json.dumps(original));original['pilot'].pop('cache_distance',None)
                if permitted!=original:raise ValueError('frozen design changed beyond cache-distance exception')
            elif sha(path)!=digest:raise ValueError('frozen input changed: '+name)

    def save_handoff(status):
        inventory=[]
        for directory in sorted((ROOT/'results').iterdir()):
            if directory.is_dir():
                hashes={name:sha(directory/name) for name in ['episodes.jsonl','calibration.json','summary.json','prediction_summary.json'] if (directory/name).exists()}
                inventory.append({'directory':rel(directory),'files':hashes})
        data={'status':status,'controller_pid':os.getpid(),'controller_run':str(workflow.dir),'paths':{k:str(v) for k,v in paths.items()},
              'gates':gates,'issues':issues,'active':active,'aborted_calibrations':args.aborted_calibration,'elapsed_seconds':time.monotonic()-started,'runs':inventory,
              'git_log':subprocess.check_output(['git','log','--oneline'],cwd=ROOT,text=True)}
        source=evidence('handoff_snapshot',data)
        lines=['# Handoff',status,'',f'Controller PID: `{os.getpid()}`. Controller log: `{workflow.dir/"run.log"}`.',
               'Child command output is in the detached launch log; each experiment also has its own run.log.','',
               table(['Gate','Verdict'],[(name,value['verdict']) for name,value in gates.items()]),'',
               'The harness was audited and fixed before the design freeze. Real Llama preflight passed. Actual measurements and all failed/aborted directories are preserved. No Qwen model, scale experiment, or manuscript was started. Original baseline tests included mock scale wiring checks only.','',
               'Open issues:','']
        lines.extend('- '+issue for issue in issues or ['See docs/design_changes.md and the audit report for modeling/locality limitations.'])
        lines+=['','Documented corrections and additions from the supplied design:','',
                '- Predictive intervals now use replicated binary cohorts; the original parameter-mean interval is retained. Parameter draws, simulation count and seeds are unchanged.',
                '- A separate paired benign false-quarantine probe supplies direct evidence for the stated utility gate; the main pilot conditions and counts are unchanged.',
                '- Fitted agent/detector parameters and cache-distance selection exclude held-out-family units. The evaluator-specific cheap-screen nonce cue was removed before freezing.',
                '- V3 uses ProtectAI OR the shared-model judge for tickets; ProtectAI alone for reports/notices. One message is one budget unit, with per-tier time logged. Reworded C3 requests can resemble legitimate work, so downstream containment relies on lineage holds.',
                '- The v2 detector was selected on calibration buildings and training families under the preregistered recall/FPR rule. The LLM judge shares the agents model, so failures may be correlated.',
                '- Original baseline mock artifacts predate complete calibration journals and are preserved as wiring evidence, excluded from research results.',
                '- A controller-only artifact ownership repair was tested after the initial calibration; the completed calibration and original frozen experimental settings were retained.']
        if args.aborted_calibration:
            lines+=['','Aborted calibration directories (preserved, read-only):','']+['- `'+path+'`' for path in args.aborted_calibration]
            lines+=['','Cause: no exception or exit code survived. pmset reports the three caffeinate clients dying at 22:48:36 PKT, with no sleep event in that hour; root cause is unconfirmed. Investigation: `'+str(args.interruption_evidence)+'`. Completed legacy units were imported into the new checkpoint directory. The original v3 design/tag is unchanged; see the FROZEN.md infrastructure erratum.']
        lines+=['','Completed/active phase directories:','']+[f'- {name}: `{directory}`' for name,directory in paths.items()]
        if active:
            lines+=['',f'Active process: `{active}`.']
            lines+=['','Do not start a second copy while that process is alive.']
            if active.get('stage')=='calibration':
                lines+=['','Calibration now has append-only unit-keyed resume. Once its prior process is confirmed dead, resume with `scripts/run_calibration.py --resume '+str(paths.get('calibration'))+'`. Restart the controller with this --calibration directory after an infrastructure interruption; no completed unit/model call is repeated. Legacy aborted sources remain read-only.']
        if 'pilot'in paths:
            lines+=['','Pilot resume command:','```sh',shlex.join([str(ROOT/'.venv/bin/python'),'scripts/run_pilot.py','--resume',str(paths['pilot'])]),'```']
        lines+=['','Read-only regeneration commands:','```sh']
        for name in ['calibration','pilot','probe','prediction']:
            if name in paths:
                lines.append(shlex.join([str(ROOT/'.venv/bin/python'),'scripts/recompute_run.py','--run',str(paths[name])]))
        lines+=['```','','Run inventory (hashes are snapshots for any active files):','']
        for row in inventory:
            lines.append(f'- `{row["directory"]}`')
            for filename,digest in row['files'].items():lines.append(f'  - `{filename}`: `{digest}`')
        lines+=['','Git history since baseline:','```text',data['git_log'].strip(),'```']
        report('HANDOFF.md','\n'.join(lines),[source])
        if args.handoff_copy:
            import shutil
            shutil.copyfile(DOCS/'HANDOFF.md',Path(args.handoff_copy))
        commit('docs: update Llama workflow handoff',[DOCS/'HANDOFF.md',DOCS/'ledger.csv'])

    def run(stage,script,arguments):
        nonlocal active,active_process
        verify()
        before=set((ROOT/'results').iterdir())
        expected_name,known_directory=owned_run_identity(script,arguments)
        command=['caffeinate','-dimsu',sys.executable,'-u',str(ROOT/'scripts'/script),*arguments]
        child=subprocess.Popen(command,cwd=ROOT,start_new_session=True);active_process=child
        active={'stage':stage,'pid':child.pid,'command':command}
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            try:
                owned=select_owned_run(set((ROOT/'results').iterdir())-before, expected_name,known_directory)
            except RuntimeError:
                owned=None
            if owned is not None:
                paths[stage]=owned;active['directory']=str(owned);break
            if child.poll() is not None:break
            time.sleep(.1)
        workflow.record('processes',active);workflow.log(f'start {stage} pid={child.pid}')
        save_handoff('In progress: '+stage)
        return_code=child.wait();active=None;active_process=None
        workflow.record('processes',{'stage':stage,'returncode':return_code,'finished':time.time()})
        directory=select_owned_run(set((ROOT/'results').iterdir())-before, expected_name, paths.get(stage))
        paths[stage]=directory
        if return_code and stage=='pilot' and (directory/'transactions.jsonl').exists():
            # One bounded recovery attempt, in the same directory, after the old process exits.
            command=['caffeinate','-dimsu',sys.executable,'-u',str(ROOT/'scripts/run_pilot.py'),'--resume',str(directory)]
            child=subprocess.Popen(command,cwd=ROOT,start_new_session=True);active_process=child
            active={'stage':'pilot_resume','pid':child.pid,'command':command,'directory':str(directory)}
            workflow.record('processes',active);save_handoff('In progress: recovering interrupted pilot')
            return_code=child.wait();active=None;active_process=None
            workflow.record('processes',{'stage':'pilot_resume','returncode':return_code,'finished':time.time()})
        if return_code:raise RuntimeError(f'{stage} failed with return code {return_code}; preserved {directory}')
        return directory

    try:
        verify();save_handoff('In progress: starting frozen Llama workflow')
        cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
        units=cfg['calibration']['units_per_cell']
        if 'calibration' in paths:
            calibration_dir=paths['calibration']
            if not (calibration_dir/'calibration.json').exists() or not (calibration_dir/'metrics.json').exists():
                calibration_dir=run('calibration','run_calibration.py',['--resume',str(calibration_dir)])
        else:
            calibration_dir=run('calibration','run_calibration.py',['--backend','mlx','--model','llama-4bit','--units',str(units),'--name','cal_llama_'+manifest.get('tag','frozen-pilot-v2').rsplit('-',1)[-1]])
        calibration=json.loads((calibration_dir/'calibration.json').read_text())
        if calibration.get('backend')!='mlx' or calibration.get('model')!='llama-4bit':
            raise ValueError('real Llama workflow cannot use mock or other-model calibration')
        if manifest.get('tag')=='frozen-pilot-v3':
            from iotpi.v3_revision import project_timing
            observations=[r['scoring'] for r in read_jsonl(calibration_dir/'detector_measurements.jsonl') if r['detector']=='screen' and r.get('scoring')]
            refreshed=project_timing(timing,cfg,observations,read_jsonl(calibration_dir/'generations.jsonl'))
            timing_source=evidence('v3_timing_after_calibration',refreshed)
            report('08_v3_calibration_timing.md','# V3 timing after calibration\n\n'+table(['Metric','Hours'],[(k,refreshed[k]) for k in ['estimated_calibration_plus_pilot_hours','estimated_probe_hours','estimated_total_with_probe_hours','estimated_calibration_plus_pilot_uncached_hours','v3_remaining_after_calibration_hours']])+ '\n\n'+refreshed['interpretation'],[timing_source])
            commit('phase: refresh v3 timing from completed calibration',[DOCS/'08_v3_calibration_timing.md',DOCS/'ledger.csv'])
        sizes=required_sample_sizes(calibration)
        if any(n<20 for n in sizes.values()):
            snapshot=evidence('calibration_coverage',{'calibration':str(calibration_dir),'sample_sizes':sizes,
                'source_shortfalls':calibration['counts'].get('source_shortfalls',[]), 'automatic_reruns':False})
            report('03_calibration.md','# Calibration coverage stopped\n\n'+table(['Rate','n'],sizes.items())+
                   '\n\nSource capacity is finite. No doubling or automatic recalibration is permitted.',[calibration_dir/'calibration.json',snapshot])
            raise RuntimeError('insufficient calibration exposures; no automatic rerun')
        try:
            cache=tune_cache(calibration_dir);cache_source=evidence('cache_tuning',cache)
        except Exception as error:
            problem=evidence('cache_tuning_failure',{'calibration':str(calibration_dir),'sample_sizes':sizes,'error':repr(error)})
            body='# Llama calibration completed; cache selection stopped\n\n'+table(['Agent count','k','n','Rate','Source-clustered interval (v2; Wilson for legacy)'],
                 [[k,r['k'],r['n'],cell(r['rate']),str(r['ci95'])] for k,r in calibration['counts']['agent_counts'].items()])
            for name,counts in calibration['counts']['detector_counts'].items():
                body+='\n\nDetector '+name+'\n\n'+table(['Surface/family or kind','k','n','Rate','Source-clustered interval (v2; Wilson for legacy)'],[[k,r['k'],r['n'],cell(r['rate']),str(r['ci95'])] for k,r in counts.items()])
            body+='\n\nCache selection could not satisfy the predefined operating-point rule: '+repr(error)+'. No evaluation data or other parameters were tuned.'
            report('03_calibration.md',body,[calibration_dir/'calibration.json',problem])
            commit('phase: record calibration and cache-selection failure',[DOCS/'03_calibration.md',DOCS/'ledger.csv'])
            raise
        cfg['pilot']['cache_distance']=cache['selected_distance']
        config_path=ROOT/'configs/default.yaml'
        config_text=config_path.read_text()
        if re.search(r'(?m)^  cache_distance:',config_text):
            config_text=re.sub(r'(?m)^  cache_distance:.*$',f'  cache_distance: {cache["selected_distance"]}',config_text,count=1)
        else:
            config_text=config_text.replace('\npilot:\n',f'\npilot:\n  cache_distance: {cache["selected_distance"]}\n',1)
        if yaml.safe_load(config_text)!=cfg:raise ValueError('cache-only config update did not preserve the frozen design')
        config_path.write_text(config_text)
        with (DOCS/'FROZEN.md').open('a') as f:
            f.write('\n\n## Permitted cache-distance addendum\n\nSelected cache_distance: '+str(cache['selected_distance'])+'. Calibration-only source: `'+str(cache_source)+'`. Updated config SHA256: `'+sha(ROOT/'configs/default.yaml')+'`. No other design values changed.\n')
        report('FROZEN.md',(DOCS/'FROZEN.md').read_text(),[Path(args.freeze),cache_source])
        cal_rows=[]
        for key,row in calibration['counts']['agent_counts'].items():
            cal_rows.append([key,row['k'],row['n'],cell(row['rate']),str(row['ci95'])])
        body='# Llama calibration\n\nTraining-family parameters alone drive risk allocation and prediction; full family tables are descriptive.\n\n'+table(['Role/key/family','k','n','Rate','Source-clustered interval (v2; Wilson for legacy)'],cal_rows)
        for name,counts in calibration['counts']['detector_counts'].items():
            body+='\n\nDetector '+name+'\n\n'+table(['Surface/family or benign kind','k','n','Rate','Source-clustered interval (v2; Wilson for legacy)'],[[k,r['k'],r['n'],cell(r['rate']),str(r['ci95'])] for k,r in counts.items()])
        for name,counts in calibration['counts'].get('training_detector_counts',{}).items():
            body+='\n\nTraining-only detector parameters: '+name+'\n\n'+table(['Surface/family or benign kind','k','n','Rate','Source-clustered interval (v2; Wilson for legacy)'],[[k,r['k'],r['n'],cell(r['rate']),str(r['ci95'])] for k,r in counts.items()])
        body+='\n\nSource independence:\n\n'+table(['Role/key/family','Receiving units','Immediate sources','Original carrier lineages'],[[key,row['n'],row.get('n_sources','legacy'),row.get('n_source_clusters','legacy')] for key,row in calibration['counts']['agent_counts'].items()])
        body+='\n\nUsed-rate coverage:\n\n'+table(['Rate','pooled n'],sizes.items())
        body+='\n\nCache distance and benign distribution:\n\n'+table(['Metric','Value'],[(k,cell(cache[k])) for k in ['selected_distance','benign_count','fingerprint_count','false_matches','false_match_rate']])
        body+='\n\nDistances:\n\n'+table(['Distance','Benign messages'],cache['distance_histogram'].items())
        report('03_calibration.md',body,[calibration_dir/'calibration.json',cache_source])
        commit('phase: complete Llama calibration and permitted cache tuning',[ROOT/'configs/default.yaml',DOCS/'FROZEN.md',DOCS/'03_calibration.md',DOCS/'ledger.csv'])
        save_handoff('In progress: calibration complete')

        if 'pilot'in paths:
            directory=paths['pilot']
            if not (directory/'summary.json').exists():
                # Preserve and resume the existing run; never create a second pilot copy.
                verify();command=['caffeinate','-dimsu',sys.executable,'-u','scripts/run_pilot.py','--resume',str(directory)]
                child=subprocess.Popen(command,cwd=ROOT,start_new_session=True);active_process=child;active={'stage':'pilot_resume','pid':child.pid,'command':command}
                save_handoff('In progress: resuming pilot');code=child.wait();active=None;active_process=None
                if code:raise RuntimeError('pilot resume failed')
        else:
            directory=run('pilot','run_pilot.py',['--backend','mlx','--model','llama-4bit','--risk-rates',str(calibration_dir/'calibration.json'),'--keep-prompts','--name','pilot_llama_'+manifest.get('tag','frozen-pilot-v2').rsplit('-',1)[-1]])
        validation=analyse_pilot(directory);validation_source=evidence('pilot_validation',validation)
        pilot_config=yaml.safe_load((directory/'config.yaml').read_text())
        resolved=pilot_config['resolved']
        if resolved['backend']!='mlx' or resolved['model_key']!='llama-4bit' or resolved.get('split','evaluation')!='evaluation' or resolved.get('limit') is not None:
            raise ValueError('main pilot is not a complete real Llama evaluation run')
        if any(pilot_config[key]!=cfg[key] for key in ['world','pilot','split','generation','detectors']):
            raise ValueError('pilot configuration differs from the frozen design')
        if resolved.get('risk_rates_sha256')!=sha(calibration_dir/'calibration.json'):
            raise ValueError('pilot risk parameters do not match this calibration')
        if validation['rows']!=validation['expected_rows'] or validation['unique_rows']!=validation['rows'] or validation['trace_rows']!=validation['unique_traces'] or validation['budget_violations'] or validation['credit_violations'] or validation['missing_plan_indices']:
            raise RuntimeError('pilot row/budget validation failed; all data preserved')
        if validation['adaptive_query_max']>cfg['pilot']['adaptive']['queries']:raise RuntimeError('adaptive query cap exceeded')
        if not all(p['identical_scenario'] and p['identical_generation_seeds'] for p in validation['pairing']):raise RuntimeError('pairing validation failed')
        summary=json.loads((directory/'summary.json').read_text());gates['gate1']=gate_one(summary)
        for name in ['no_guard','policy_only']:
            if summary['conditions'][name]['harm'][0]<.1:issues.append(name+' harm is near zero; the main comparison is weak for this reference. No tuning was performed on evaluation data.')
        if 'probe'not in paths:
            try:run('probe','run_quarantine_probe.py',['--pilot',str(directory)])
            except Exception as error:issues.append('Supplemental quarantine probe did not complete: '+repr(error))
        if 'probe'in paths and (paths['probe']/'summary.json').exists():gates['gate3']=gate_three(paths['probe'])
        pilot_analysis=evidence('pilot_gates',{'gate1':gates['gate1'],'gate3':gates['gate3'],'validation':validation})
        body='# Llama pilot\n\nGenerated summary, pasted unchanged:\n\n'+(directory/'summary.md').read_text()
        if manifest.get('tag')=='frozen-pilot-v3':
            body+='\n\nOn pre-pilot calibration evidence, agent-reworded attacks are largely undetectable by either screened setup: both use ProtectAI alone on reports and notices. Attack-derived C3 requests look like legitimate requests, and the judge also flags legitimate device requests, so it is not used on reports. Downstream containment therefore relies on lineage holds. The pilot measures whether those holds succeed; calibration alone does not establish their effectiveness. The judge shares the agents model, so failures may be correlated.'
            design=manifest['detector_design']['data']
            body+='\n\nTraining/calibration evidence: '+table(['Metric','k','n'],[['Ticket OR detections',design['raw_tickets']['either'],design['raw_tickets']['n']],['ProtectAI ticket detections',design['raw_tickets']['protectai'],design['raw_tickets']['n']],['Rewritten detections under v3',design['summary']['rewritten']['k'],design['summary']['rewritten']['n']],['Judge benign-report flags',design['benign_reports']['judge'],design['benign_reports']['n']]])
        body+='\n\nHarm by goal:\n\n'+table(['Condition','Goal','Mean and context interval'],[[name,goal,str(values)] for name,s in summary['conditions'].items() for goal,values in s.get('harm_by_goal',{}).items()])
        body+='\n\nHarm by split:\n\n'+table(['Condition','Split','Mean and context interval'],[[name,split,str(s.get('harm_'+split))] for name,s in summary['conditions'].items() for split in ['train','heldout','adaptive']])
        body+='\n\nGate verdicts:\n\n'+table(['Gate','Verdict'],[(key,gates[key]['verdict']) for key in ['gate1','gate3']])
        body+='\n\nValidation:\n\n'+table(['Metric','Value'],[(k,cell(validation[k])) for k in ['rows','expected_rows','unique_rows','trace_rows','unique_traces','parse_failures','generation_calls','parse_failure_rate','budget_violations','credit_violations','adaptive_query_max']])
        if 'probe'in paths and (paths['probe']/'summary.json').exists():
            body+='\n\nControlled false-quarantine probe:\n\n'+json.dumps(gates['gate3'],indent=2)+'\n\n'+(paths['probe']/'summary.md').read_text()
        report('04_pilot.md',body,[directory/'summary.json',validation_source,pilot_analysis,Path(args.freeze)])
        commit('phase: complete Llama pilot and report containment gates',[DOCS/'04_pilot.md',DOCS/'ledger.csv'])
        save_handoff('In progress: pilot complete; prediction follows regardless of gate verdicts')

        if 'prediction'not in paths:
            run('prediction','predict_multihop.py',['--calibration',str(calibration_dir/'calibration.json'),'--pilot',str(directory),'--name','pred_llama'])
        prediction=paths['prediction'];pred_summary=json.loads((prediction/'prediction_summary.json').read_text())
        gates['gate2']=gate_two(pred_summary);pred_source=evidence('prediction_gate',gates['gate2'])
        body='# Llama prediction check\n\n'+(prediction/'prediction_summary.md').read_text()
        body+='\n\nGate: '+gates['gate2']['verdict']+'.\n\n'+gates['gate2']['interpretation']
        body+='\n\nCompare the unscreened references separately with TrustGate in the table. The simulator cheap heuristic scores synthetic text rather than actual LLM rewrites. Carry and target outcomes are also sampled from separate marginal rates; target actions themselves count as carry in the evaluator. These abstractions can bias the prediction and are not fitted away using evaluation data.'
        body+='\n\nThe corrected interval includes replicated binary episode outcomes as documented before evaluation; the original parameter-mean interval remains separately available. A predictive interval does not by itself certify security or establish deployment validity. Gate failure leaves the simulator illustrative.'
        report('05_prediction.md',body,[prediction/'prediction_summary.json',pred_source])
        commit('phase: complete Llama prediction check',[DOCS/'05_prediction.md',DOCS/'ledger.csv'])
        recomputed=[]
        for name in ['calibration','pilot','probe','prediction']:
            if name in paths and any((paths[name]/f).exists() for f in ['calibration.json','summary.json','prediction_summary.json']):
                command=[sys.executable,str(ROOT/'scripts/recompute_run.py'),'--run',str(paths[name])]
                output=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
                recomputed.append({'stage':name,'command':command,'returncode':output.returncode,'stdout':output.stdout,'stderr':output.stderr})
                if output.returncode:issues.append('Read-only regeneration mismatch: '+name)
        recompute_source=evidence('regeneration_checks',{'checks':recomputed})
        workflow.log('regeneration checks -> '+str(recompute_source))
        if gates['gate3']['verdict']=='not reached' or any(r['returncode'] for r in recomputed):
            failed=True
            save_handoff('Finished with unresolved checks; review open issues before treating the workflow as complete')
        else:
            save_handoff('Complete: Llama workflow finished; gate verdicts are reported without retuning')
    except BaseException as error:
        if active_process is not None and active_process.poll() is None:
            os.killpg(active_process.pid,signal.SIGTERM)
            try:active_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active_process.pid,signal.SIGKILL);active_process.wait()
            active=None
        failed=True;issues.append(repr(error));workflow.record('errors',{'error':repr(error),'traceback':traceback.format_exc()})
        workflow.log('stopped: '+repr(error));save_handoff('Stopped with preserved evidence; remaining gates were not reached')
    finally:
        workflow.save_json('metrics.json',{'failed':failed,'gates':gates,'paths':{k:str(v) for k,v in paths.items()},'issues':issues})
        workflow.close()
    return 1 if failed else 0

if __name__=='__main__':sys.exit(main())
