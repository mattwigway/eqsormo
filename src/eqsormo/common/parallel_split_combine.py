"""
A class to make implementing parallelism easier using a common pattern found in eqsormo.

Designed to be subclassed, define a worker (queue_item) and consumer (worker_return) method.
"""

import multiprocessing
import threading
import queue

class ParallelSplitCombine(object):
    def run (self, tasks, nthreads=None):
        if nthreads is None:
            nthreads = multiprocessing.cpu_count()
        
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_threads = threading.Event()

        for i in range(nthreads):
            threading.Thread(target=self._worker).start()

        threading.Thread(target=self._consumer).start()

        for task in tasks:
            self.task_queue.put(task)

        # wait for all tasks to be consumed to ensure they're in results, then wait for
        # results to be processed
        self.task_queue.join()
        self.result_queue.join()

        stop_threads.set()

    def _worker (self):
        while not self.stop_threads.is_set():
            task = self.task_queue.get(timeout=10)
        except queue.Empty:
            continue  # check for task completiong
        else:
            self.result_queue.put(self.worker(task))
            self.task_queue.task_done()

    def _consumer (self):
        while not self.stop_threads.is_set():
            try:
                res = self.result_queue.get(timeout=10)
            except queue.Empty:
                continue
            else:
                self.consumer(res)
                self.result_queue.task_done()

