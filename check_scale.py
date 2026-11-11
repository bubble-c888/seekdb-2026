"""Exercise a live logical table larger than the unchanged 2-GiB container limit."""
import argparse,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'protocol_check'))
from _transport import Session

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);args=parser.parse_args()
    config=json.loads((ROOT/'protocol_check/config.json').read_text())
    with tempfile.TemporaryDirectory(prefix='scale-check-') as directory:
        session=Session(config,Path(directory)/'stderr.log',image=args.image)
        try:
            session.start()
            def request(label,sql,verbose=True):
                print('START '+label,flush=True) if verbose else None
                begin=time.perf_counter();r=session.request(sql=sql)
                assert r['ok'],r
                if verbose:print(json.dumps({'operation':label,'seconds':round(time.perf_counter()-begin,3)}),flush=True)
                return r
            request('schema','CREATE TABLE large (id BIGINT PRIMARY KEY, g BIGINT, text VARCHAR(131072), v VECTOR(4))')
            count=17000
            payload='abcdefgh01234567'*8192
            begin=time.perf_counter()
            for start in range(0,count,20):
                stop=min(count,start+20)
                values=','.join("(%d,%d,'%s','[%d,0,0,0]')"%(i,i%3,payload,i) for i in range(start,stop))
                request('bulk_insert','INSERT INTO large VALUES '+values,False)
                if stop%1000==0:print(json.dumps({'inserted':stop,'elapsed':round(time.perf_counter()-begin,3)}),flush=True)
            print(json.dumps({'logical_payload_bytes':count*len(payload),'container_memory_limit_bytes':2**31}),flush=True)
            rows=request('filtered_scan','SELECT id FROM large WHERE g = 2 ORDER BY id DESC LIMIT 3')
            assert rows['rows']==[[i] for i in range(count-1,-1,-1) if i%3==2][:3]
            request('index','CREATE VECTOR INDEX large_v ON large(v)')
            request('full_table_update','UPDATE large SET g = 4')
            request('explicit_begin','BEGIN')
            request('explicit_update','UPDATE large SET g = 5')
            request('explicit_commit','COMMIT')
            print('START crash_restart',flush=True);session.crash();session.start();print('PASS crash_restart',flush=True)
            rows=request('post_recovery','SELECT id FROM large WHERE g = 5 ORDER BY id DESC LIMIT 3')
            assert rows['rows']==[[count-1],[count-2],[count-3]]
            request('full_table_delete','DELETE FROM large')
            assert request('empty','SELECT id FROM large')['rows']==[]
            session.shutdown();print('PASS larger-than-memory scenario',flush=True)
        finally:session.close()
if __name__=='__main__':main()
