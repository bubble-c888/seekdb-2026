"""Run the repository's established FTS mysqltest cases and expected results."""
import importlib.util,json,os,shutil
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('seekdb_mysqltest',ROOT/'.github/script/seekdb/mysqltest_for_seekdb.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
client=next(iter((ROOT/'deps/3rd').glob('**/bin/obclient')),None)
mysqltest=next(iter((ROOT/'deps/3rd').glob('**/bin/mysqltest')),None)
assert client is not None and mysqltest is not None,'The repository test-utils client and mysqltest are required'
args=SimpleNamespace(seekdb=ROOT/'build_release/src/observer/seekdb',obclient=client.resolve(),mysqltest=mysqltest.resolve(),base_dir=Path(os.environ['RUNNER_TEMP'])/'seekdb-fts-regression',work_dir=ROOT/'fts-results/regression',host='127.0.0.1',port=2881)
args.work_dir.mkdir(parents=True,exist_ok=True)
cases=[c for c in m.discover_cases(ROOT) if c.name.startswith('fts_index.')]
assert cases,'No FTS cases collected'
print('Collected FTS cases:',len(cases),flush=True)
sdb=ROOT/'.github/script/seekdb/sdb.py';deploy=ROOT/'tools/deploy'
results=[]
try:
 m.prepare_instance(args,ROOT,sdb,deploy)
 temp=args.work_dir/'tmp';logs=args.work_dir/'logs';temp.mkdir(exist_ok=True);logs.mkdir(exist_ok=True)
 for case in cases:
  code,output=m.run_case(args,deploy,case,temp,logs)
  (logs/(case.name+'.log')).write_text(output,encoding='utf-8')
  results.append({'case':case.name,'returncode':code})
  print(case.name,'PASS' if code==0 else 'FAIL',flush=True)
  if code:print(output[-6000:],flush=True)
 (args.work_dir/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 assert all(r['returncode']==0 for r in results), 'FTS correctness regression failed'
finally:
 m.destroy_instance(sdb,args.base_dir,ROOT,check=False)
