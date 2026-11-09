"""Server loop: wires the database, the executor and the protocol together."""

from __future__ import annotations

import signal
import sys
from typing import Optional, TextIO

from .engine.database import Database
from .engine.executor import Executor
from .protocol import Protocol


class Server:
    """Runs the database until the client closes the connection."""

    def __init__(
        self,
        database: Database,
        stdin: Optional[TextIO] = None,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
        ready_target: Optional[str] = None,
    ):
        self.database = database
        self.executor = Executor(database)
        self.protocol = Protocol(
            executor=self.executor,
            stdin=stdin if stdin is not None else sys.stdin,
            stdout=stdout if stdout is not None else sys.stdout,
            stderr=stderr if stderr is not None else sys.stderr,
            ready_target=ready_target,
        )
        self._stopping = False

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # pragma: no cover - signal path
            self._stopping = True
            raise KeyboardInterrupt

        for name in ("SIGTERM", "SIGINT"):
            number = getattr(signal, name, None)
            if number is not None:
                try:
                    signal.signal(number, handler)
                except (ValueError, OSError):  # pragma: no cover
                    pass

    def run(self) -> int:
        self._install_signal_handlers()
        status = 0
        try:
            status = self.protocol.run()
        except KeyboardInterrupt:
            status = 0
        finally:
            self.shutdown()
        return status

    def shutdown(self) -> None:
        """Roll back an open transaction and flush everything to disk."""
        try:
            if self.database.in_transaction:
                self.database.rollback()
            self.database.close()
        except Exception as error:  # pragma: no cover - defensive
            print(f"minivectordb: error during shutdown: {error}", file=sys.stderr)
