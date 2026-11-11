import argparse
import functools
import heapq
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid


class DBError(Exception):
    def __init__(self, code):
        self.code = code


def require(condition, code='syntax_error'):
    if not condition:
        raise DBError(code)


class Parser:
    pattern = re.compile(r"\s*(?:('(?:''|[^'])*')|([A-Za-z_][A-Za-z_0-9]*)|(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(<=|>=|<>|!=|[(),;*=<>\[\]]))")

    def __init__(self, sql):
        self.tokens = []
        pos = 0
        # Do not copy the remaining SQL per token: a bulk INSERT may be MBs.
        while pos < len(sql):
            match = self.pattern.match(sql, pos)
            if match is None and sql[pos:].isspace():
                break
            require(match is not None)
            self.tokens.append(next(x for x in match.groups() if x is not None))
            pos = match.end()
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos].upper() if self.pos < len(self.tokens) else ''

    def take(self, expected=None):
        require(self.pos < len(self.tokens))
        value = self.tokens[self.pos]
        self.pos += 1
        require(expected is None or value.upper() == expected)
        return value

    def accept(self, token):
        if self.peek() == token:
            self.pos += 1
            return True
        return False

    def name(self):
        value = self.take()
        require(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', value))
        return value.lower()

    def literal(self):
        token = self.take()
        if token.startswith("'"):
            return token[1:-1].replace("''", "'")
        if token == '[':
            result = []
            if not self.accept(']'):
                while True:
                    result.append(self.literal())
                    if self.accept(']'):
                        break
                    self.take(',')
            return result
        require(re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', token), 'type_error')
        return float(token) if any(c in token.lower() for c in '.e') else int(token)

    def end(self):
        self.accept(';')
        require(not self.peek(), 'unsupported_sql')


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class Database:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = self.root / 'commits'
        self.logs.mkdir(exist_ok=True)
        self.base = self.root / 'rows'
        self.base.mkdir(exist_ok=True)
        self.catalog = {}
        self.indexes = {}
        self.dirty = set()
        self.tx = None
        self.stages = []
        # A checkpoint is published only after materialization is complete.
        # Recovery never rewrites transactions whose applied state is known.
        self.applied_sequence = 0
        checkpoint = self.root / 'checkpoint.json'
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text(encoding='utf-8'))
            self.catalog = state['catalog']
            self.indexes = state['indexes']
            self.applied_sequence = state['sequence']
        # Published commit files are durable redo records. Unpublished .tmp files
        # and transaction workspaces never become visible after process death.
        for log in sorted(self.logs.glob('*.jsonl')):
            if int(log.stem) > self.applied_sequence:
                self.apply(log)
                self.applied_sequence = int(log.stem)
        self.workspace = self.root / 'workspace'
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir()
        self.sequence = max((int(p.stem) for p in self.logs.glob('*.jsonl')), default=self.applied_sequence)
        self.write_checkpoint()
        for table in self.indexes:
            self.build_index(table)

    def write_checkpoint(self):
        target = self.root / 'checkpoint.json'
        temp = target.with_suffix('.tmp')
        state = {'sequence': self.applied_sequence, 'catalog': self.catalog, 'indexes': self.indexes}
        with temp.open('w', encoding='utf-8') as file:
            file.write(encode(state))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
        self.sync_dir(self.root)

    def apply(self, log):
        with log.open(encoding='utf-8') as file:
            for line in file:
                op = json.loads(line)
                if op[0] == 'index':
                    self.indexes[op[1]] = op[2]
                    self.dirty.add(op[1])
                elif op[0] == 'schema':
                    self.catalog[op[1]] = op[2]
                    (self.base / op[1]).mkdir(exist_ok=True)
                else:
                    self.dirty.add(op[1])
                    target = self.base / op[1] / (str(op[2]) + '.json')
                    patch = target.with_suffix('.patch')
                    if op[0] == 'patch':
                        changes = json.loads(patch.read_text(encoding='utf-8')) if patch.exists() else {}
                        changes.update(op[3])
                        temp = patch.with_suffix('.patch-tmp')
                        temp.write_text(encode(changes), encoding='utf-8')
                        os.replace(temp, patch)
                    elif op[3] is None:
                        target.unlink(missing_ok=True)
                        patch.unlink(missing_ok=True)
                    else:
                        temp = target.with_suffix('.tmp')
                        temp.write_text(encode(op[3]), encoding='utf-8')
                        os.replace(temp, target)
                        patch.unlink(missing_ok=True)

    def build_index(self, table):
        """External-merge a coordinate projection; distance provides an exact bound."""
        column = self.indexes[table]
        schema = self.catalog[table]
        position = next(i for i,c in enumerate(schema) if c[0] == column)
        runs = []
        chunk = []
        folder = self.root / 'indexes'
        folder.mkdir(exist_ok=True)
        def write_run():
            path = folder / (uuid.uuid4().hex + '.run')
            with path.open('wb') as file:
                for coordinate, key in sorted(chunk):
                    file.write(struct.pack('<fq', coordinate, key))
            runs.append(path)
            chunk.clear()
        for key,row in self.rows(table):
            chunk.append((row[position][0],key))
            if len(chunk) >= 4096:
                write_run()
        if chunk:
            write_run()
        def records(path):
            with path.open('rb') as file:
                while raw := file.read(12):
                    yield struct.unpack('<fq',raw)
        # Bound open file descriptors even when the dataset creates many runs.
        while len(runs) > 64:
            merged = []
            for begin in range(0, len(runs), 64):
                group = runs[begin:begin + 64]
                output = folder / (uuid.uuid4().hex + '.run')
                with output.open('wb') as file:
                    for record in heapq.merge(*(records(path) for path in group)):
                        file.write(struct.pack('<fq', *record))
                merged.append(output)
                for path in group:
                    path.unlink()
            runs = merged
        target = folder / (table + '.idx')
        temp = target.with_suffix('.tmp')
        try:
            with temp.open('wb') as file:
                for record in heapq.merge(*(records(path) for path in runs)):
                    file.write(struct.pack('<fq',*record))
            os.replace(temp,target)
        finally:
            for path in runs:
                path.unlink(missing_ok=True)
        self.dirty.discard(table)

    def external_sorted(self, rows, key):
        """Sort bounded runs on disk; never retain an entire table for ordering."""
        paths = []
        chunk = []
        def write_run(records):
            path = self.workspace / (uuid.uuid4().hex + '.sort')
            paths.append(path)
            with path.open('w', encoding='utf-8') as file:
                for record in records:
                    file.write(encode(record) + '\n')
            return path
        def records(path):
            with path.open(encoding='utf-8') as file:
                for line in file:
                    yield json.loads(line)
        try:
            runs = []
            for row in rows:
                chunk.append(row)
                if len(chunk) >= 4096:
                    runs.append(write_run(sorted(chunk, key=key)))
                    chunk.clear()
            if chunk:
                runs.append(write_run(sorted(chunk, key=key)))
                chunk.clear()
            while len(runs) > 64:
                merged = []
                for begin in range(0, len(runs), 64):
                    group = runs[begin:begin + 64]
                    merged.append(write_run(heapq.merge(*(records(path) for path in group), key=key)))
                    for path in group:
                        path.unlink()
                runs = merged
            yield from heapq.merge(*(records(path) for path in runs), key=key)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

    def indexed_rows(self, table, vector, predicate, limit):
        if table in self.dirty:
            self.build_index(table)
        path = self.root / 'indexes' / (table + '.idx')
        if not path.stat().st_size or not limit:
            return []
        column = next(i for i,c in enumerate(self.catalog[table]) if c[0] == self.indexes[table])
        heap = []
        with path.open('rb') as file, mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ) as index:
            count = len(index)//12
            low, high = 0,count
            while low < high:
                middle=(low+high)//2
                if struct.unpack_from('<f',index,middle*12)[0] < vector[0]:low=middle+1
                else:high=middle
            left,right=low-1,low
            while left>=0 or right<count:
                dl=abs(struct.unpack_from('<f',index,left*12)[0]-vector[0]) if left>=0 else math.inf
                dr=abs(struct.unpack_from('<f',index,right*12)[0]-vector[0]) if right<count else math.inf
                bound=min(dl,dr)
                if len(heap)==limit and bound > -heap[0][0]:break
                offset=left if dl<=dr else right
                if dl<=dr:left-=1
                else:right+=1
                _,key=struct.unpack_from('<fq',index,offset*12)
                row=self.get(table,key)
                if row is None or not predicate(row):continue
                distance=math.sqrt(math.fsum((a-b)**2 for a,b in zip(row[column],vector)))
                item=(-distance,-key,key,row)
                if len(heap)<limit:heapq.heappush(heap,item)
                elif item[:2]>heap[0][:2]:heapq.heapreplace(heap,item)
        return [(item[2],item[3]) for item in heap]

    def sync_dir(self, directory):
        if os.name != 'nt':
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def commit(self):
        self.sequence += 1
        temp = self.logs / f'{self.sequence:020d}.tmp'
        target = temp.with_suffix('.jsonl')
        with temp.open('w', encoding='utf-8', newline='\n') as file:
            for stage in self.stages:
                with stage.open(encoding='utf-8') as source:
                    shutil.copyfileobj(source, file, 65536)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
        self.sync_dir(self.logs)
        self.apply(target)
        self.applied_sequence = self.sequence
        self.write_checkpoint()
        self.clear()

    def clear(self, rollback=False):
        if rollback and self.tx is not None:
            self.catalog,self.indexes=self.tx
        self.stages = []
        self.tx = None
        shutil.rmtree(self.workspace)
        self.workspace.mkdir()

    def schema(self, name):
        require(name in self.catalog, 'unknown_table')
        return self.catalog[name]

    def overlay(self, table):
        # SQLite and other database/storage engines are deliberately not used.
        # Transaction overrides live on disk, with a bounded one-row scan buffer.
        directory = self.workspace / table
        directory.mkdir(exist_ok=True)
        return directory

    def get(self, table, key):
        overlay = self.overlay(table) / (str(key) + '.json')
        staged = json.loads(overlay.read_text(encoding='utf-8')) if overlay.exists() else {}
        if not isinstance(staged, dict):
            return staged
        path = self.base / table / (str(key) + '.json')
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding='utf-8'))
        patch = path.with_suffix('.patch')
        if patch.exists():
            for index, value in json.loads(patch.read_text(encoding='utf-8')).items():
                row[int(index)] = value
        for index, value in staged.get('patch', {}).items():
            row[int(index)] = value
        return row

    def rows(self, table):
        overlay = self.overlay(table)
        for item in os.scandir(self.base / table):
            if item.name.endswith('.json'):
                key = int(Path(item.name).stem)
                row = self.get(table, key)
                if row is not None:
                    yield key, row
        for item in os.scandir(overlay):
            if item.name.endswith('.json') and not (self.base / table / item.name).exists():
                row = json.loads(Path(item.path).read_text(encoding='utf-8'))
                if row is not None:
                    yield int(Path(item.name).stem), row

    def validate(self, value, column):
        _, kind, size, _ = column
        if kind == 'BIGINT':
            require(type(value) is int and -(2**63) <= value < 2**63, 'type_error')
        elif kind == 'VARCHAR':
            require(isinstance(value, str) and len(value) <= size, 'type_error')
        else:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    raise DBError('invalid_vector')
            require(isinstance(value, list) and len(value) == size, 'invalid_vector')
            converted = []
            for number in value:
                require(type(number) in (float, int), 'invalid_vector')
                try:
                    number = struct.unpack('<f', struct.pack('<f', number))[0]
                except (OverflowError, struct.error):
                    raise DBError('invalid_vector')
                require(math.isfinite(number), 'invalid_vector')
                converted.append(number)
            value = converted
        return value

    def expression(self, parser, columns):
        if parser.accept('L2_DISTANCE'):
            parser.take('(')
            column = parser.name()
            require(column in columns, 'unknown_column')
            index = columns.index(column)
            require(self.active_schema[index][1] == 'VECTOR', 'type_error')
            parser.take(',')
            vector = self.validate(parser.literal(), self.active_schema[index])
            parser.take(')')
            function=lambda row: math.sqrt(math.fsum((a-b)**2 for a,b in zip(row[index], vector)))
            function.vector=vector
            function.column=column
            return function, 'L2_DISTANCE'
        name = parser.name()
        require(name in columns, 'unknown_column')
        index = columns.index(name)
        return (lambda row: row[index]), name

    def where(self, parser, columns):
        conditions = []
        if parser.accept('WHERE'):
            while True:
                name = parser.name()
                require(name in columns, 'unknown_column')
                index = columns.index(name)
                operator = parser.take()
                require(operator in ('=', '<', '<=', '>', '>='), 'unsupported_sql')
                value = self.validate(parser.literal(), self.active_schema[index])
                require(self.active_schema[index][1] != 'VECTOR', 'unsupported_sql')
                conditions.append((index, operator, value))
                if not parser.accept('AND'):
                    break
        operations = {'=': lambda a,b:a==b, '<':lambda a,b:a<b, '<=':lambda a,b:a<=b, '>':lambda a,b:a>b, '>=':lambda a,b:a>=b}
        return lambda row: all(operations[op](row[index],value) for index,op,value in conditions)

    def execute(self, sql, mode='exact'):
        require(mode in ('exact', 'ann'), 'unsupported_mode')
        parser = Parser(sql)
        command = parser.take().upper()
        if command in ('BEGIN','COMMIT','ROLLBACK'):
            parser.end()
            if command == 'BEGIN':
                require(self.tx is None, 'transaction_active')
                self.tx = (self.catalog.copy(),self.indexes.copy())
            else:
                require(self.tx is not None, 'no_transaction')
                self.commit() if command == 'COMMIT' else self.clear(rollback=True)
            return {}
        if command == 'CREATE':
            if parser.accept('VECTOR'):
                parser.take('INDEX'); parser.name(); parser.take('ON'); table=parser.name()
                schema=self.schema(table); parser.take('('); name=parser.name(); parser.take(')'); parser.end()
                require(any(c[0]==name and c[1]=='VECTOR' for c in schema), 'unknown_column')
                require(table not in self.indexes,'index_exists')
                stage=self.workspace/(uuid.uuid4().hex+'.stage')
                stage.write_text(encode(['index',table,name])+'\n',encoding='utf-8')
                self.stages.append(stage)
                if self.tx is None:
                    self.commit();self.build_index(table)
                else:self.indexes[table]=name
                return {}
            parser.take('TABLE'); table=parser.name(); parser.take('(')
            require(table not in self.catalog, 'table_exists')
            schema=[]
            while True:
                name=parser.name(); kind=parser.take().upper(); require(kind in ('BIGINT','VARCHAR','VECTOR'), 'unsupported_type')
                size=0
                if kind!='BIGINT':
                    parser.take('(');size=parser.literal();parser.take(')');require(type(size)is int and size>0,'type_error')
                primary=False
                if parser.accept('PRIMARY'):
                    parser.take('KEY');primary=True;require(kind=='BIGINT','type_error')
                require(name not in [c[0] for c in schema], 'duplicate_column')
                schema.append([name,kind,size,primary])
                if parser.accept(')'):break
                parser.take(',')
            parser.end();require(sum(c[3] for c in schema)==1 and sum(c[1]=='VECTOR' for c in schema)<=1,'invalid_schema')
            stage=self.workspace/(uuid.uuid4().hex+'.stage');stage.write_text(encode(['schema',table,schema])+'\n',encoding='utf-8')
            self.stages.append(stage)
            if self.tx is None:self.commit()
            else:
                self.catalog[table]=schema
                (self.base/table).mkdir(exist_ok=True)
            return {}
        if command == 'SELECT':
            # Parse projection before FROM, locating the table without interpreting
            # words inside quoted literals as SQL clauses.
            start=parser.pos
            from_pos=next((i for i in range(start,len(parser.tokens)) if parser.tokens[i].upper()=='FROM'),None)
            require(from_pos is not None)
            table=parser.tokens[from_pos+1].lower();schema=self.schema(table);self.active_schema=schema
            columns=[c[0] for c in schema]; projections=[];names=[]
            if parser.accept('*'):
                projections=[lambda row,i=i:row[i] for i in range(len(columns))];names=columns.copy()
            else:
                while True:
                    expression,name=self.expression(parser,columns)
                    if parser.accept('AS'):name=parser.name()
                    projections.append(expression);names.append(name)
                    if not parser.accept(','):break
            parser.take('FROM');require(parser.name()==table)
            predicate=self.where(parser,columns);order=None;descending=False;limit=None
            if parser.accept('ORDER'):
                parser.take('BY')
                if parser.peek().lower() in names:
                    index=names.index(parser.take().lower());order=projections[index]
                else:order,_=self.expression(parser,columns)
                if parser.accept('DESC'):descending=True
                else:parser.accept('ASC')
            if parser.accept('LIMIT'):
                limit=parser.literal();require(type(limit)is int and limit>=0,'type_error')
            parser.end()
            rows=((key,row) for key,row in self.rows(table) if predicate(row))
            if (order is not None and not descending and limit is not None and self.tx is None
                and table in self.indexes and getattr(order,'column',None)==self.indexes[table]):
                rows=self.indexed_rows(table,order.vector,predicate,limit)
            if order is not None:
                def compare(a,b):
                    left,right=order(a[1]),order(b[1]);result=(left>right)-(left<right)
                    return (-result if descending else result) or ((a[0]>b[0])-(a[0]<b[0]))
                key=functools.cmp_to_key(compare)
                rows=heapq.nsmallest(limit,rows,key=key) if limit is not None else self.external_sorted(rows,key=key)
            elif limit is not None:
                import itertools
                rows=itertools.islice(rows,limit)
            return {'columns':names,'rows':[[function(row) for function in projections] for _,row in rows]}
        require(command in ('INSERT','UPDATE','DELETE'), 'unsupported_sql')
        if command=='INSERT':parser.take('INTO')
        if command=='DELETE':parser.take('FROM')
        table=parser.name();schema=self.schema(table);self.active_schema=schema;columns=[c[0] for c in schema]
        primary=next(i for i,c in enumerate(schema) if c[3])
        stage=self.workspace/(uuid.uuid4().hex+'.stage');count=0
        try:
            with stage.open('w',encoding='utf-8') as file:
                if command=='INSERT':
                    selected=columns
                    if parser.accept('('):
                        selected=[]
                        while True:
                            selected.append(parser.name())
                            if parser.accept(')'):break
                            parser.take(',')
                        require(len(selected)==len(set(selected)) and set(selected)==set(columns),'unknown_column')
                    parser.take('VALUES');seen=set()
                    while True:
                        parser.take('(');values=[]
                        while True:
                            values.append(parser.literal())
                            if parser.accept(')'):break
                            parser.take(',')
                        require(len(values)==len(selected),'column_count')
                        row=[self.validate(values[selected.index(c[0])],c) for c in schema];key=row[primary]
                        require(key not in seen and self.get(table,key)is None,'duplicate_key');seen.add(key)
                        file.write(encode(['row',table,key,row])+'\n');count+=1
                        if not parser.accept(','):break
                else:
                    changes={}
                    if command=='UPDATE':
                        parser.take('SET')
                        while True:
                            name=parser.name();require(name in columns,'unknown_column');index=columns.index(name)
                            require(index!=primary,'unsupported_sql');parser.take('=');changes[index]=self.validate(parser.literal(),schema[index])
                            if not parser.accept(','):break
                    predicate=self.where(parser,columns)
                    for key,row in self.rows(table):
                        if predicate(row):
                            if command=='UPDATE':
                                operation=['patch',table,key,changes]
                            else:
                                operation=['row',table,key,None]
                            file.write(encode(operation)+'\n');count+=1
                parser.end()
            # Autocommit publishes the validated redo directly. Only explicit
            # transactions need a second on-disk copy for read-your-own-writes.
            if self.tx is not None:
                with stage.open(encoding='utf-8') as file:
                    for line in file:
                        kind,_,key,row=json.loads(line)
                        target = self.overlay(table)/(str(key)+'.json')
                        if kind == 'patch':
                            previous = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'patch': {}}
                            if isinstance(previous, list):
                                for index,value in row.items():previous[int(index)] = value
                                row = previous
                            else:
                                previous['patch'].update(row)
                                row = previous
                        target.write_text(encode(row),encoding='utf-8')
            self.stages.append(stage)
            if self.tx is None:self.commit()
            return {'affected_rows':count}
        except Exception:
            stage.unlink(missing_ok=True)
            raise


def main():
    args=argparse.ArgumentParser();args.add_argument('--data-dir',default='/data');options=args.parse_args()
    db=Database(options.data_dir)
    print('{"type":"ready"}',flush=True)
    for line in sys.stdin:
        ident=None
        try:
            request=json.loads(line);ident=request.get('id')
            if request.get('op')=='shutdown':
                db.clear(rollback=True);print(encode({'id':ident,'ok':True}),flush=True);return
            require(request.get('op')=='execute','unsupported_operation')
            result=db.execute(request['sql'],request.get('search_mode','exact'))
            response={'id':ident,'ok':True,**result}
        except DBError as error:
            response={'id':ident,'ok':False,'error':{'code':error.code,'message':error.code}}
        except Exception as error:
            print(repr(error),file=sys.stderr,flush=True)
            response={'id':ident,'ok':False,'error':{'code':'execution_error','message':str(error)}}
        print(encode(response),flush=True)
    db.clear(rollback=True)


if __name__=='__main__':main()
