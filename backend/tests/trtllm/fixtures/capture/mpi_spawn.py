"""Can the image's MPI spawn worker processes the way TensorRT-LLM's MpiPoolSession does (mpi4py MPIPoolExecutor), with
the container restrictions of the console's node? Two workers, a small and a large (CMA-sized) payload. No model, no GPU."""
import os
import sys
import time

from mpi4py.futures import MPIPoolExecutor


def work(payload):
    from mpi4py import MPI
    return (MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size(), len(payload), os.getpid())


if __name__ == "__main__":
    t = time.monotonic()
    with MPIPoolExecutor(max_workers=2, path=sys.path) as ex:
        small = list(ex.map(work, [b"x" * 100] * 4))
        big = list(ex.map(work, [b"y" * (16 * 1024 * 1024)] * 2))
    print("small:", sorted({(r, s) for r, s, _, _ in small}), "big payload ok:", [n for _, _, n, _ in big],
          "seconds:", round(time.monotonic() - t, 1), flush=True)
