import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'protocol_check'))
from _assertions import assert_response

config=json.loads((ROOT/'protocol_check/config.json').read_text())
manifest=json.loads((ROOT/'mysqltest/manifest.json').read_text())
cases=manifest['cases'] if isinstance(manifest,dict) else manifest
results=[]
for case in cases:
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        def start():
            process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'main.py'),'--data-dir',directory],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
            assert json.loads(process.stdout.readline())=={'type':'ready'}
            return process
        process=start();ident=0;index=0;mode='exact'
        def request(op,**kw):
            global ident
            ident+=1
            process.stdin.write(json.dumps({'id':ident,'op':op,**kw})+'\n');process.stdin.flush()
            result=json.loads(process.stdout.readline());assert result['id']==ident
            return result
        try:
            expected=json.loads((ROOT/'mysqltest'/case['result']).read_text(encoding='utf-8'))
            for sql in (ROOT/'mysqltest'/case['test']).read_text(encoding='utf-8').splitlines():
                sql=sql.strip()
                if not sql or sql.startswith('#'):continue
                if sql=='--restart':
                    assert request('shutdown')['ok'];process.wait(timeout=10);process=start()
                elif sql.startswith('--mode '):mode=sql.split()[1]
                else:
                    response=request('execute',sql=sql,search_mode=mode)
                    assert_response(response,expected[index],config);index+=1
            assert index==len(expected)
            assert request('shutdown')['ok'];process.wait(timeout=10)
            print(case['id'],'PASS',index,flush=True);results.append({'case':case['id'],'passed':True,'statements':index})
        except Exception as error:
            print(case['id'],'FAIL',repr(error),'SQL',sql,'response',locals().get('response'),flush=True)
            results.append({'case':case['id'],'passed':False,'error':str(error)})
        finally:
            if process.poll()is None:process.kill();process.wait()
output=ROOT/'local-results/windows-public.json';output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(results,indent=2),encoding='utf-8')
assert all(case['passed'] for case in results)
