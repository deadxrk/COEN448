"""Deterministic tests for an async task scheduler.

Set SCHEDULER_CLASS to the import path of your scheduler class if it differs
from the default `scheduler.TaskScheduler`.
"""
from __future__ import annotations

import importlib
import inspect
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional
from unittest import mock

import pytest


DEFAULT_SCHEDULER_CLASS = "scheduler.TaskScheduler"


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._now = start

    def time(self) -> float:
        with self._lock:
            return self._now

    def monotonic(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("Cannot move time backwards")
        with self._lock:
            self._now += seconds

    def sleep(self, seconds: float) -> None:
        # Non-blocking sleep for deterministic tests.
        self.advance(seconds)


class SchedulerAdapter:
    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler

    def start(self) -> None:
        if hasattr(self.scheduler, "start"):
            self.scheduler.start()

    def shutdown(self) -> None:
        for method_name in ("shutdown", "stop", "close"):
            if hasattr(self.scheduler, method_name):
                getattr(self.scheduler, method_name)()
                return

    def schedule(self, callback: Callable[..., Any], *, delay: Optional[float] = None, run_at: Optional[float] = None):
        if hasattr(self.scheduler, "schedule"):
            return self.scheduler.schedule(callback, delay=delay, run_at=run_at)
        if delay is not None and hasattr(self.scheduler, "call_later"):
            return self.scheduler.call_later(delay, callback)
        if run_at is not None and hasattr(self.scheduler, "call_at"):
            return self.scheduler.call_at(run_at, callback)
        raise AttributeError("Scheduler does not support schedule/call_later/call_at")

    def cancel(self, task: Any) -> Any:
        for method_name in ("cancel", "remove"):
            if hasattr(self.scheduler, method_name):
                return getattr(self.scheduler, method_name)(task)
        raise AttributeError("Scheduler does not support cancel/remove")

    def run_due(self) -> None:
        for method_name in ("run_due", "run_pending", "dispatch_due", "tick", "run_once"):
            if hasattr(self.scheduler, method_name):
                getattr(self.scheduler, method_name)()
                return
        if hasattr(self.scheduler, "run"):
            self.scheduler.run(block=False)
            return
        raise AttributeError("Scheduler does not support run_due/run_pending/tick/run")


@pytest.fixture()
def scheduler_with_clock(monkeypatch):
    clock = FakeClock()
    scheduler_path = os.environ.get("SCHEDULER_CLASS", DEFAULT_SCHEDULER_CLASS)
    module_name, class_name = scheduler_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"Unable to import scheduler module {module_name!r}: {exc}")

    scheduler_cls = getattr(module, class_name, None)
    if scheduler_cls is None:
        pytest.skip(f"Scheduler class {class_name!r} not found in {module_name!r}")

    init_params = inspect.signature(scheduler_cls).parameters
    kwargs = {}
    for name in ("time_fn", "time", "now", "clock"):
        if name in init_params:
            kwargs[name] = clock.time
            break
    for name in ("monotonic_fn", "monotonic"):
        if name in init_params:
            kwargs[name] = clock.monotonic
            break
    if "sleep_fn" in init_params:
        kwargs["sleep_fn"] = clock.sleep

    scheduler = scheduler_cls(**kwargs)

    # Patch module-level time usage if the scheduler doesn't accept injection.
    if "time" in module.__dict__:
        if hasattr(module.time, "time"):
            monkeypatch.setattr(module.time, "time", clock.time)
        if hasattr(module.time, "monotonic"):
            monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
        if hasattr(module.time, "sleep"):
            monkeypatch.setattr(module.time, "sleep", clock.sleep)

    adapter = SchedulerAdapter(scheduler)
    adapter.start()

    yield adapter, clock

    adapter.shutdown()


def _advance_and_run(adapter: SchedulerAdapter, clock: FakeClock, seconds: float) -> None:
    clock.advance(seconds)
    adapter.run_due()


def test_delayed_execution_accuracy(scheduler_with_clock):
    adapter, clock = scheduler_with_clock
    callback = mock.Mock()

    adapter.schedule(callback, delay=5)

    _advance_and_run(adapter, clock, 4.999)
    callback.assert_not_called()

    _advance_and_run(adapter, clock, 0.002)
    callback.assert_called_once()


def test_invalid_scheduling_times_raise(scheduler_with_clock):
    adapter, clock = scheduler_with_clock
    callback = mock.Mock()

    with pytest.raises(ValueError):
        adapter.schedule(callback, delay=-1)

    past_time = clock.time() - 10
    with pytest.raises(ValueError):
        adapter.schedule(callback, run_at=past_time)


def test_concurrent_dispatch_thread_safety(scheduler_with_clock):
    adapter, clock = scheduler_with_clock
    callback = mock.Mock()
    barrier = threading.Barrier(8)

    def schedule_task():
        barrier.wait()
        adapter.schedule(callback, delay=1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in range(8):
            executor.submit(schedule_task)

    _advance_and_run(adapter, clock, 1)

    assert callback.call_count == 8


def test_task_cancellation_and_edge_cases(scheduler_with_clock):
    adapter, clock = scheduler_with_clock
    callback = mock.Mock()

    task = adapter.schedule(callback, delay=2)
    cancelled = adapter.cancel(task)

    _advance_and_run(adapter, clock, 2)
    callback.assert_not_called()

    # Cancelling twice should be safe and deterministic.
    second_cancel = adapter.cancel(task)
    assert cancelled in (True, False, None)
    assert second_cancel in (True, False, None)

    # Cancelling after execution should not raise.
    task2 = adapter.schedule(callback, delay=1)
    _advance_and_run(adapter, clock, 1)
    adapter.cancel(task2)
