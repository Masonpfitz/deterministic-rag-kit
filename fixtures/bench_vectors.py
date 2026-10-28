#!/usr/bin/env python3
"""Why the kit stores vectors as a numpy file and not in sqlite-vec.

Builds N random unit vectors (1024 dims, like bge-m3) and times a top-30 nearest
neighbour query with (a) numpy brute force over a memory-mapped float32 file and
(b) the sqlite-vec extension, if it can be loaded. Run it from a venv with numpy and,
optionally, sqlite-vec:   python fixtures/bench_vectors.py 30000 75000
Chunks are roughly 2.5 per note, so 30000 chunks is about 10,000 notes.
"""
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def timeit(fn, n=15):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return ts[len(ts) // 2], ts[int(len(ts) * 0.95) - 1]


def main(sizes):
    rng = np.random.default_rng(0)
    q = rng.standard_normal(1024).astype(np.float32)
    q /= np.linalg.norm(q)
    try:
        import sqlite_vec
        con = sqlite3.connect(":memory:")
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        have_vec = True
    except Exception as exc:  # noqa: BLE001
        have_vec = False
        print("sqlite-vec unavailable here:", exc)
    for n in sizes:
        mat = rng.standard_normal((n, 1024)).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v.npy"
            t = time.perf_counter()
            np.save(path, mat)
            write_s = time.perf_counter() - t
            mm = np.load(path, mmap_mode="r")

            def np_query():
                s = np.asarray(mm @ q)
                top = np.argpartition(-s, 29)[:30]
                return top[np.argsort(-s[top])]
            p50, p95 = timeit(np_query)
            print(f"{n:>7} vectors  numpy mmap: p50 {p50:6.1f} ms  p95 {p95:6.1f} ms   (file {path.stat().st_size/1e6:.0f} MB, write {write_s:.2f}s)")
            if have_vec:
                db = sqlite3.connect(":memory:")
                db.enable_load_extension(True)
                sqlite_vec.load(db)
                db.execute("create virtual table v using vec0(e float[1024])")
                t = time.perf_counter()
                with db:
                    db.executemany("insert into v(rowid, e) values (?, ?)",
                                   ((i, mat[i].tobytes()) for i in range(n)))
                ins = time.perf_counter() - t
                qb = q.tobytes()

                def vec_query():
                    return db.execute("select rowid, distance from v where e match ? and k = 30", (qb,)).fetchall()
                p50, p95 = timeit(vec_query)
                print(f"{n:>7} vectors  sqlite-vec: p50 {p50:6.1f} ms  p95 {p95:6.1f} ms   (insert {ins:.1f}s)")


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [30000, 75000])
