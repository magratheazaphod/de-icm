"""SQLite storage for the web app: tournaments, saved spots and player notes.

The database lives at ``$DEICM_DB`` (on Fly, a volume at /data, streamed to a
private bucket by Litestream). With ``DEICM_DB`` unset the app runs stateless
and the storage routes answer 503.

Records are keyed by UUIDs the phone generates, and every write is an
idempotent upsert, so a save queued offline can be replayed any number of
times. Player notes are about real people: they live only in this database
and its backups, never in the repository.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

# Each entry migrates the schema from user_version i to i + 1.
MIGRATIONS = [
    """
    CREATE TABLE tournaments (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        venue TEXT NOT NULL DEFAULT '',
        start_date TEXT NOT NULL DEFAULT '',
        buy_in REAL,
        entrants INTEGER,
        paid_pct REAL,
        notes TEXT NOT NULL DEFAULT '',
        live TEXT,
        client_ts REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE scenarios (
        id TEXT PRIMARY KEY,
        tournament_id TEXT REFERENCES tournaments(id) ON DELETE SET NULL,
        title TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        input TEXT NOT NULL,
        client_ts REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX scenarios_tournament ON scenarios(tournament_id, created_at);
    CREATE TABLE players (
        id TEXT PRIMARY KEY,
        nickname TEXT NOT NULL UNIQUE COLLATE NOCASE,
        description TEXT NOT NULL DEFAULT '',
        archetype TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        fold_to_cbet REAL,
        raise_vs_cbet REAL,
        re_jam_rate REAL,
        fold_to_iso REAL,
        client_ts REAL NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE observations (
        id TEXT PRIMARY KEY,
        player_id TEXT NOT NULL REFERENCES players(id) ON DELETE CASCADE,
        tournament_id TEXT REFERENCES tournaments(id) ON DELETE SET NULL,
        text TEXT NOT NULL,
        seen_at REAL NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE INDEX observations_player ON observations(player_id, seen_at)
    """,
]

TABLES = {
    "tournaments": ("name", "venue", "start_date", "buy_in", "entrants", "paid_pct", "notes", "live"),
    "scenarios": ("tournament_id", "title", "note", "input"),
    "players": (
        "nickname", "description", "archetype", "notes",
        "fold_to_cbet", "raise_vs_cbet", "re_jam_rate", "fold_to_iso",
    ),
}
ORDER = {
    "tournaments": "start_date DESC, created_at DESC",
    "scenarios": "created_at DESC",
    "players": "nickname COLLATE NOCASE",
    "observations": "seen_at DESC",
}
JSON_FIELDS = {"input", "live"}


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by FastAPI's worker threads, serialized by a lock:
        # traffic is one person between hands, so contention is not a concern.
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            # WAL is what Litestream replicates; it also lets readers run during a write.
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA busy_timeout = 5000")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            for i, sql in enumerate(MIGRATIONS[version:], start=version):
                # Statement by statement inside one transaction: executescript would commit early.
                self._db.execute("BEGIN")
                for statement in (s.strip() for s in sql.split(";")):
                    if statement:
                        self._db.execute(statement)
                self._db.execute(f"PRAGMA user_version = {i + 1}")
                self._db.execute("COMMIT")

    def ping(self) -> None:
        with self._lock:
            self._db.execute("SELECT 1").fetchone()

    @staticmethod
    def _out(table: str, row: sqlite3.Row) -> dict:
        d = dict(row)
        for f in JSON_FIELDS & d.keys():
            if d[f] is not None:
                d[f] = json.loads(d[f])
        return d

    def list(self, table: str, **where) -> list[dict]:
        clauses, args = [], []
        for col, val in where.items():
            if val is None:
                clauses.append(f"{col} IS NULL")
            else:
                clauses.append(f"{col} = ?")
                args.append(val)
        sql = f"SELECT * FROM {table}"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._db.execute(f"{sql} ORDER BY {ORDER[table]}", args).fetchall()
        return [self._out(table, r) for r in rows]

    def put(self, table: str, row_id: str, data: dict, client_ts: float) -> dict:
        """Create or fully replace a record, last write wins by the phone's clock.

        A write whose ``client_ts`` is older than the stored one is ignored, so a
        retry queued earlier in the level cannot undo a newer edit. Fields not
        sent take their defaults. Returns the stored record either way.
        """
        fields = TABLES[table]
        values = {f: data.get(f) for f in fields}
        for f in JSON_FIELDS & values.keys():
            if values[f] is not None:
                values[f] = json.dumps(values[f])
        now = time.time()
        cols = ["id", *fields, "client_ts", "created_at", "updated_at"]
        updates = ", ".join(f"{f} = excluded.{f}" for f in (*fields, "client_ts", "updated_at"))
        # Columns with a NOT NULL text default get '' rather than an explicit NULL.
        text_defaults = {"venue", "start_date", "notes", "note", "description", "archetype"}
        vals = [row_id, *(("" if values[f] is None and f in text_defaults else values[f]) for f in fields),
                client_ts, now, now]
        with self._lock:
            self._db.execute(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
                f" ON CONFLICT(id) DO UPDATE SET {updates} WHERE excluded.client_ts >= {table}.client_ts",
                vals,
            )
            row = self._db.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
        return self._out(table, row)

    def observe(self, obs_id: str, player_id: str, text: str, seen_at: float,
                tournament_id: str | None = None) -> dict:
        """Append one observation of a player. Replaying the same id is a no-op."""
        with self._lock:
            self._db.execute(
                "INSERT INTO observations (id, player_id, tournament_id, text, seen_at, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
                (obs_id, player_id, tournament_id, text, seen_at, time.time()),
            )
            row = self._db.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        return dict(row)

    def delete(self, table: str, row_id: str) -> None:
        """Idempotent: deleting a missing record is not an error, so replays are safe."""
        with self._lock:
            self._db.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))

    def tournaments(self) -> list[dict]:
        with self._lock:
            counts = dict(
                self._db.execute(
                    "SELECT tournament_id, COUNT(*) FROM scenarios WHERE tournament_id IS NOT NULL GROUP BY 1"
                ).fetchall()
            )
        return [{**t, "n_scenarios": counts.get(t["id"], 0)} for t in self.list("tournaments")]

    def export(self) -> dict:
        return {
            "schema": len(MIGRATIONS),
            "exported_at": time.time(),
            **{t: self.list(t) for t in (*TABLES, "observations")},
        }
