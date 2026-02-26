# src/utilities/task_queue.py

"""Task queue management with priorities, workers, and async processing."""

import queue
import threading
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures


class TaskStatus(Enum):
    """Possible states of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Represents a unit of work to be processed."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error": self.error
        }


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class Worker(threading.Thread):
    """Worker thread that processes tasks from a queue."""
    
    def __init__(self, name: str, task_queue: 'TaskQueue'):
        super().__init__(name=name, daemon=True)
        self.task_queue = task_queue
        self.running = True
        self.current_task: Optional[Task] = None
        self.tasks_processed = 0
    
    def run(self):
        while self.running:
            try:
                task = self.task_queue.get_task()
                if task is None:
                    time.sleep(0.01)
                    continue
                
                self.current_task = task
                self._process_task(task)
                self.tasks_processed += 1
            except Exception as e:
                print(f"Worker {self.name} error: {e}")
    
    def _process_task(self, task: Task):
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        try:
            if task.timeout:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(task.func, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                self.task_queue.requeue_task(task)
            else:
                task.status = TaskStatus.FAILED
        finally:
            task.completed_at = datetime.now()
            self.current_task = None
            self.task_queue.task_completed(task)
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False


class TaskQueue:
    """Priority-based task queue with worker threads."""
    
    def __init__(self, num_workers: int = 4, max_queue_size: int = 1000):
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.workers: List[Worker] = []
        self.tasks: Dict[str, Task] = {}
        
        self.queues = {
            TaskPriority.LOW: queue.Queue(),
            TaskPriority.NORMAL: queue.Queue(),
            TaskPriority.HIGH: queue.Queue(),
            TaskPriority.CRITICAL: queue.Queue()
        }
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._task_completed = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        for i in range(self.num_workers):
            worker = Worker(f"Worker-{i+1}", self)
            worker.start()
            self.workers.append(worker)
    
    def add_task(self, func: Callable, *args, 
                 priority: TaskPriority = TaskPriority.NORMAL,
                 task_id: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: Optional[float] = None,
                 **kwargs) -> str:
        """Add a task to the queue."""
        if self.get_queue_size() >= self.max_queue_size:
            raise queue.Full("Task queue is full")
        
        task = Task(
            id=task_id or str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )
        
        with self._lock:
            self.tasks[task.id] = task
            self.queues[priority].put(task)
        
        return task.id
    
    def get_task(self) -> Optional[Task]:
        """Get the next task from the queue based on priority."""
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            try:
                task = self.queues[priority].get_nowait()
                return task
            except queue.Empty:
                continue
        return None
    
    def requeue_task(self, task: Task):
        """Requeue a task for retry."""
        if task.retry_count <= task.max_retries:
            self.queues[task.priority].put(task)
    
    def task_completed(self, task: Task):
        """Mark a task as completed."""
        self._task_completed.set()
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get the status of a task."""
        task = self.tasks.get(task_id)
        return task.to_dict() if task else None
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get the result of a completed task."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            execution_time = None
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
            
            return TaskResult(
                task_id=task.id,
                success=task.status == TaskStatus.COMPLETED,
                result=task.result,
                error=task.error,
                execution_time=execution_time
            )
        
        return None
    
    def wait_for_task(self, task_id: str, timeout: float = 5.0) -> Optional[TaskResult]:
        """Wait for a task to complete with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_task_result(task_id)
            if result is not None:
                return result
            self._task_completed.wait(0.05)
            self._task_completed.clear()
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        with self._lock:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                return True
            elif task.status == TaskStatus.RUNNING:
                return False
            else:
                return False
    
    def get_queue_size(self) -> int:
        """Get the total number of tasks in the queue."""
        total = 0
        for q in self.queues.values():
            total += q.qsize()
        return total
    
    def get_active_tasks(self) -> List[Dict]:
        """Get information about currently running tasks."""
        active = []
        for worker in self.workers:
            if worker.current_task and worker.current_task.status == TaskStatus.RUNNING:
                active.append(worker.current_task.to_dict())
        return active
    
    def shutdown(self, wait: bool = True):
        """Shutdown the task queue and workers."""
        self._stop_event.set()
        
        if wait:
            timeout = 5
            start_time = time.time()
            while self.get_queue_size() > 0 and time.time() - start_time < timeout:
                time.sleep(0.1)
        
        for worker in self.workers:
            worker.stop()
        
        for worker in self.workers:
            worker.join(timeout=2)


class QueueManager:
    """Manages multiple named task queues."""
    
    def __init__(self):
        self.queues: Dict[str, TaskQueue] = {}
        self._lock = threading.Lock()
    
    def create_queue(self, name: str, num_workers: int = 4) -> TaskQueue:
        """Create a new named queue."""
        with self._lock:
            if name in self.queues:
                raise ValueError(f"Queue '{name}' already exists")
            
            queue = TaskQueue(num_workers=num_workers)
            self.queues[name] = queue
            return queue
    
    def get_queue(self, name: str) -> Optional[TaskQueue]:
        """Get a named queue."""
        return self.queues.get(name)
    
    def shutdown_all(self, wait: bool = True):
        """Shutdown all managed queues."""
        for queue in self.queues.values():
            queue.shutdown(wait=wait)