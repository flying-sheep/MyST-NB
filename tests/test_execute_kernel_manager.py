"""Tests for passing an externally-owned ``kernel_manager`` to `create_client`.

Unlike the rest of this test suite, these don't go through a full sphinx
build: `kernel_manager=` is only reachable by code that instantiates the
parser (or `create_client`) directly, since a normal sphinx build always
constructs the registered parser class with no arguments.
"""

from __future__ import annotations

import gc
import warnings

import nbformat
import pytest
from jupyter_client import AsyncKernelManager

from myst_nb.core.config import NbParserConfig
from myst_nb.core.execute import create_client
from nbclient.util import run_sync


class _FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _notebook():
    return nbformat.v4.new_notebook(
        metadata=nbformat.NotebookNode(
            kernelspec=dict(name="python3", display_name="", language="python")
        ),
        cells=[nbformat.v4.new_code_cell("1 + 1")],
    )


@pytest.fixture()
def km():
    manager = AsyncKernelManager(kernel_name="python3")
    yield manager
    if manager.has_kernel:
        run_sync(manager.shutdown_kernel)(now=True)


@pytest.mark.parametrize("execution_mode", ["force", "cache", "inline"])
def test_external_kernel_manager_client_is_closed(km, tmp_path, execution_mode):
    """A kernel client created from an externally supplied kernel_manager
    must have its channels closed once execution finishes.

    Regression test: previously, only kernel managers *created* internally
    by the execution client (i.e. `kernel_manager=None`) had their kernel
    client cleaned up - one supplied by the caller was left with open zmq
    channels, since neither the caller (who never sees the client) nor
    nbclient (which doesn't own the manager) closed it.
    """
    nb_path = tmp_path / "nb.ipynb"
    nbformat.write(_notebook(), nb_path)
    nb_config = NbParserConfig(
        execution_mode=execution_mode,
        execution_in_temp=True,
        execution_cache_path=str(tmp_path / ".jupyter_cache"),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with create_client(
            _notebook(), str(nb_path), nb_config, _FakeLogger(), kernel_manager=km
        ) as client:
            if execution_mode == "inline":
                # inline mode executes lazily, on request for a cell's outputs
                client.code_cell_outputs(0)
        assert client.exec_metadata["succeeded"]
        gc.collect()

    unclosed = [
        w
        for w in caught
        if issubclass(w.category, ResourceWarning) and "Unclosed" in str(w.message)
    ]
    assert not unclosed, [str(w.message) for w in unclosed]

    # the kernel itself must stay alive - the caller owns the manager and
    # may want to reuse the (warm) kernel for later executions
    assert km.has_kernel
