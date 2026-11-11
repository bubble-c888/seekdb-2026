"""Independent transaction/patch checks plus process death during redo replay."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
from main import Database
ROOT=Path(__file__).resolve().parent

def child(directory,mode):
    original=os.replace
    hits=0
    def interrupt(source,target):
        nonlocal hits
        result=original(source,target)
        if str(target).endswith('.patch'):
            hits+=1
            if hits==5:os._exit(77)
        return result
    if mode=='recover':os.replace=interrupt
    db=Database(directory)
    if mode=='commit':
        os.replace=interrupt
        db.execute('UPDATE t SET g = 4')
    raise AssertionError('injection did not execute')

def main():
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        db=Database(directory)
        db.execute('CREATE TABLE t (id BIGINT PRIMARY KEY, g BIGINT, text VARCHAR(200000), v VECTOR(2))')
        text='preserved payload '*7000
        values=','.join("(%d,0,'%s','[%d,0]')"%(i,text,i) for i in range(40))
        db.execute('INSERT INTO t VALUES '+values)
        db.execute('CREATE VECTOR INDEX vi ON t(v)')
        db.execute('BEGIN');db.execute('UPDATE t SET g = 2');db.execute("UPDATE t SET v = '[3,4]' WHERE id = 1")
        assert db.execute('SELECT g,v FROM t WHERE id = 1')['rows']==[[2,[3.,4.]]]
        db.execute('ROLLBACK');assert db.execute('SELECT g,v FROM t WHERE id = 1')['rows']==[[0,[1.,0.]]]
        for mode in ('commit','recover','recover'):
            r=subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--child',directory,mode])
            assert r.returncode==77,(mode,r.returncode)
        db=Database(directory)
        rows=db.execute('SELECT id,g,text FROM t ORDER BY id')['rows']
        assert rows==[[i,4,text] for i in range(40)]
        db.execute('BEGIN');db.execute('DELETE FROM t WHERE id = 1');db.execute("INSERT INTO t VALUES (1,8,'replacement','[8,0]')");db.execute('UPDATE t SET g = 9 WHERE id = 1');db.execute('COMMIT')
        db=Database(directory);assert db.execute('SELECT g,text,v FROM t WHERE id = 1')['rows']==[[9,'replacement',[8.,0.]]]
        db.execute('UPDATE t SET g = 10 WHERE id = 1');db.execute('DELETE FROM t WHERE id = 1');db.execute("INSERT INTO t VALUES (1,11,'fresh','[11,0]')")
        db=Database(directory);assert db.execute('SELECT g,text,v FROM t WHERE id = 1')['rows']==[[11,'fresh',[11.,0.]]]
        print('PASS: patch transaction rollback; insert-update and delete-reinsert semantics; interrupted COMMIT materialization; two interrupted recoveries; final complete recovery; large text preserved',flush=True)
if __name__=='__main__':
    if len(sys.argv)>1:child(sys.argv[2],sys.argv[3])
    else:main()
