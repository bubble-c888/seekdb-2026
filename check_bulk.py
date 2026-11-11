"""Large-statement regression through unchanged official Docker transport."""
import argparse,json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'protocol_check'))
from _transport import Session

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--image',required=True);args=parser.parse_args()
    config=json.loads((ROOT/'protocol_check/config.json').read_text())
    with tempfile.TemporaryDirectory(prefix='bulk-check-') as directory:
        session=Session(config,Path(directory)/'stderr.log',image=args.image)
        try:
            session.start()
            def request(sql):
                begin=time.perf_counter();r=session.request(sql=sql)
                assert r['ok'],r
                return r,round(time.perf_counter()-begin,3)
            request('CREATE TABLE bulk (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(128))')
            # No known-answer shortcut: SQL contains every stored coordinate.
            count=12000
            values=[]
            for i in range(count):
                vector=[float(i)]+[0.123456]*127
                values.append("(%d,%d,'%s')"%(i,i%3,json.dumps(vector)))
            sql='INSERT INTO bulk VALUES '+','.join(values)
            response,elapsed=request(sql);assert response['affected_rows']==count
            print(json.dumps({'bulk_rows':count,'sql_bytes':len(sql.encode()),'insert_seconds':elapsed}),flush=True)
            del values,sql
            rows,_=request('SELECT id FROM bulk ORDER BY id');assert rows['rows']==[[i] for i in range(count)]
            request('CREATE VECTOR INDEX bulk_v ON bulk(v)')
            q=json.dumps([float(count-1)]+[0.123456]*127)
            for _ in range(2):
                rows,elapsed=request("SELECT id FROM bulk WHERE g = 2 ORDER BY L2_DISTANCE(v,'%s') LIMIT 1"%q)
                assert rows['rows']==[[count-1]],rows
                print(json.dumps({'filtered_top1_seconds':elapsed}),flush=True)
                session.crash();session.start()
            rows,_=request('SELECT id FROM bulk WHERE id = 11999');assert rows['rows']==[[11999]]
            session.shutdown()
            print('PASS: 12000-row 128D bulk statement; exact row identities; filtered indexed TopK; two forced-kill recoveries',flush=True)
        finally:session.close()
if __name__=='__main__':main()
