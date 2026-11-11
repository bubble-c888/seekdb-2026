import json, math, random, struct, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).parent

def start(path):
    p=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'main.py'),'--data-dir',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
    assert json.loads(p.stdout.readline())=={'type':'ready'}
    return p

def sql(p,text,ok=True):
    p.stdin.write(json.dumps({'id':text,'op':'execute','sql':text})+'\n');p.stdin.flush()
    r=json.loads(p.stdout.readline());assert r['ok']==ok,r
    return r.get('rows',r)

def kill(p):
    p.kill();p.wait(timeout=10)

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]

with tempfile.TemporaryDirectory(dir=ROOT) as folder:
    p=start(folder)
    sql(p,'CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT, v VECTOR(4))')
    rng=random.Random(7624)
    rows=[(i,i%5,[f32(rng.uniform(-10,10)) for _ in range(4)]) for i in range(200)]
    values=','.join('(%s,%s,\'%s\')'%(i,g,json.dumps(v)) for i,g,v in rows)
    sql(p,'INSERT INTO t VALUES '+values)
    queries=[]
    for _ in range(35):
        q=[f32(rng.uniform(-10,10)) for _ in range(4)];g=rng.randrange(5);k=rng.randrange(1,20)
        expected=sorted((math.sqrt(math.fsum((a-b)**2 for a,b in zip(v,q))),i) for i,j,v in rows if j==g)[:k]
        statement="SELECT id FROM t WHERE g = %s ORDER BY L2_DISTANCE(v,'%s') LIMIT %s"%(g,json.dumps(q),k)
        assert sql(p,statement)==[[i] for _,i in expected]
        queries.append((statement,[[i] for _,i in expected]))
    sql(p,'CREATE VECTOR INDEX vi ON t(v)')
    for statement,expected in queries:assert sql(p,statement)==expected
    sql(p,'BEGIN');sql(p,"INSERT INTO t VALUES (999,0,'[0,0,0,0]')")
    assert sql(p,'SELECT id FROM t WHERE id = 999')==[[999]]
    kill(p)
    for _ in range(3):
        p=start(folder)
        assert sql(p,'SELECT id FROM t WHERE id = 999')==[]
        assert len(sql(p,'SELECT id FROM t'))==200
        for statement,expected in queries:assert sql(p,statement)==expected
        kill(p)
    p=start(folder)
    sql(p,'BEGIN');sql(p,'CREATE TABLE transient (id BIGINT PRIMARY KEY)');sql(p,'ROLLBACK')
    sql(p,'SELECT * FROM transient',False)
    sql(p,"INSERT INTO t VALUES (998,0,'[0,0,0,0]'),(1,0,'[0,0,0,0]')",False)
    assert sql(p,'SELECT id FROM t WHERE id = 998')==[]
    sql(p,'BEGIN');sql(p,'DELETE FROM t WHERE id = 0');sql(p,'COMMIT');kill(p)
    p=start(folder);assert sql(p,'SELECT id FROM t WHERE id = 0')==[];kill(p)
print('PASS: 175 independent filtered TopK checks; 5 force-kill recoveries; statement atomicity; transactional DDL rollback; committed delete durability')
