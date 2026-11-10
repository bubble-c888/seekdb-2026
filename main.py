import json
import os
import sys

from engine import Engine, SQLError


def emit(response):
    print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    engine = Engine(os.environ.get("VECTORDB_DATA_DIR", "/data"))
    emit({"type": "ready"})
    for line in sys.stdin:
        request = json.loads(line)
        ident = request["id"]
        operation = request["op"]
        if operation == "shutdown":
            engine.close()
            emit({"id": ident, "ok": True})
            return
        try:
            if operation != "execute":
                raise SQLError("unsupported_op", operation)
            result = engine.execute(request["sql"], request.get("search_mode", "exact"))
            emit({"id": ident, "ok": True, **result})
        except SQLError as error:
            emit({"id": ident, "ok": False, "error": {"code": error.code, "message": str(error)}})
    engine.close()


if __name__ == "__main__":
    main()
