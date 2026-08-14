"""
communication/serial_comm.py -- UART serial link to the STM32 Blue Pill.

Sends ASCII motor commands over UART and reads optional responses.
The protocol is newline-terminated text::

    TX -> "FWD:150\\n"
    RX <- "OK\\n"

Features
--------
* Thread-safe: all I/O is guarded by a ``threading.Lock``.
* Auto-reconnect: ``send()`` transparently attempts to re-open the port
  when the connection is lost (cable unplug, STM32 reset, etc.).
* Reads ``SERIAL_PORT``, ``SERIAL_BAUDRATE`` and ``SERIAL_TIMEOUT``
  from ``config.py``.
* Compatible with Raspberry Pi 4 (``/dev/ttyUSB0`` or ``/dev/ttyAMA0``)
  and STM32 Blue Pill (USB-CDC or UART bridge).

Standalone test::

    python -m communication.serial_comm
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import config
from utils.logger import get_logger

log = get_logger(__name__)

# Maximum number of consecutive reconnect attempts before giving up
_MAX_RECONNECT_ATTEMPTS = 3
# Delay (seconds) between reconnect attempts
_RECONNECT_DELAY = 0.5


class SerialComm:
    """Thread-safe UART communication with the STM32 Blue Pill.

    Parameters
    ----------
    port : str or None
        Serial port path (e.g. ``/dev/ttyUSB0``).  Defaults to
        ``config.SERIAL_PORT``.
    baudrate : int or None
        Defaults to ``config.SERIAL_BAUDRATE``.
    timeout : float or None
        Read timeout in seconds.  Defaults to ``config.SERIAL_TIMEOUT``.
    auto_reconnect : bool
        If *True* (the default), ``send()`` will automatically attempt
        to re-open the port when a write fails due to a disconnection.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: Optional[int] = None,
        timeout: Optional[float] = None,
        auto_reconnect: bool = True,
    ) -> None:
        self._port: str = port or config.SERIAL_PORT
        self._baudrate: int = baudrate or config.SERIAL_BAUDRATE
        self._timeout: float = timeout if timeout is not None else config.SERIAL_TIMEOUT
        self._terminator: str = config.COMMAND_TERMINATOR
        self._auto_reconnect: bool = auto_reconnect

        self._serial = None  # lazily imported pyserial.Serial instance
        self._lock = threading.Lock()

        log.info(
            "SerialComm created  [port=%s  baud=%d  timeout=%.1fs  "
            "auto_reconnect=%s]",
            self._port, self._baudrate, self._timeout, self._auto_reconnect,
        )

    # ── lifecycle ────────────────────────────────

    def open(self) -> None:
        """Open the serial port.

        Raises
        ------
        RuntimeError
            If ``pyserial`` is not installed.
        serial.SerialException
            If the port cannot be opened (e.g. device not found).
        """
        serial_mod = self._import_serial()

        with self._lock:
            # Close any stale handle before opening a fresh one
            self._close_unlocked()

            try:
                self._serial = serial_mod.Serial(
                    port=self._port,
                    baudrate=self._baudrate,
                    timeout=self._timeout,
                )
                log.info(
                    "Serial opened  [port=%s  baud=%d]",
                    self._port, self._baudrate,
                )
            except serial_mod.SerialException as exc:
                log.error("Failed to open serial port %s: %s", self._port, exc)
                self._serial = None
                raise

    def close(self) -> None:
        """Close the serial port (safe to call multiple times)."""
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        """Internal close without acquiring the lock (caller must hold it)."""
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
                    log.info("Serial closed  [port=%s]", self._port)
            except Exception as exc:
                log.warning("Error closing serial port: %s", exc)
            finally:
                self._serial = None

    # ── connection status ────────────────────────

    def is_connected(self) -> bool:
        """Return *True* if the serial port is open and ready.

        This is a method (not a property) as requested, but the legacy
        ``is_open`` property is also preserved for backward compatibility
        with ``main.py``.
        """
        with self._lock:
            return self._is_connected_unlocked()

    def _is_connected_unlocked(self) -> bool:
        """Internal check without acquiring the lock."""
        return self._serial is not None and self._serial.is_open

    @property
    def is_open(self) -> bool:
        """Legacy property kept for backward compatibility with main.py."""
        return self.is_connected()

    # ── reconnect ────────────────────────────────

    def reconnect(self) -> bool:
        """Close and re-open the serial port.

        Returns
        -------
        bool
            *True* if the reconnection succeeded.
        """
        log.info("Reconnecting serial port %s ...", self._port)
        serial_mod = self._import_serial()

        with self._lock:
            self._close_unlocked()

            for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
                try:
                    self._serial = serial_mod.Serial(
                        port=self._port,
                        baudrate=self._baudrate,
                        timeout=self._timeout,
                    )
                    log.info(
                        "Reconnected on attempt %d/%d  [port=%s]",
                        attempt, _MAX_RECONNECT_ATTEMPTS, self._port,
                    )
                    return True
                except serial_mod.SerialException as exc:
                    log.warning(
                        "Reconnect attempt %d/%d failed: %s",
                        attempt, _MAX_RECONNECT_ATTEMPTS, exc,
                    )
                    self._serial = None
                    if attempt < _MAX_RECONNECT_ATTEMPTS:
                        # Release the lock briefly so other threads aren't
                        # blocked for the entire retry backoff.
                        self._lock.release()
                        try:
                            time.sleep(_RECONNECT_DELAY)
                        finally:
                            self._lock.acquire()

        log.error(
            "Reconnect FAILED after %d attempts  [port=%s]",
            _MAX_RECONNECT_ATTEMPTS, self._port,
        )
        return False

    def _try_auto_reconnect(self) -> bool:
        """Attempt one reconnect cycle if auto_reconnect is enabled.

        Must be called **without** the lock held.

        Returns *True* if the connection is now usable.
        """
        if not self._auto_reconnect:
            return False
        log.info("Auto-reconnect triggered for %s", self._port)
        return self.reconnect()

    # ── send / receive ───────────────────────────

    def send(self, command: str) -> None:
        """Send a command string (newline is appended automatically).

        If the port is disconnected and ``auto_reconnect`` is enabled,
        a reconnect is attempted before dropping the command.

        Parameters
        ----------
        command : str
            e.g. ``"FWD:150"`` or ``"STOP:0"``.
        """
        msg = (command + self._terminator).encode("ascii")

        with self._lock:
            if self._is_connected_unlocked():
                if self._try_send_unlocked(msg, command):
                    return
                # Write failed -- fall through to reconnect path
                log.warning(
                    "Write failed on %s -- will attempt reconnect",
                    self._port,
                )
                self._close_unlocked()

        # Outside the lock: port is closed, try to reconnect
        if not self._is_connected_unlocked():
            if not self._try_auto_reconnect():
                log.warning(
                    "Serial not connected -- command dropped: %s", command,
                )
                return

        # Retry the send after reconnection
        with self._lock:
            if not self._try_send_unlocked(msg, command):
                log.error(
                    "Send failed even after reconnect -- command dropped: %s",
                    command,
                )

    def _try_send_unlocked(self, msg: bytes, command: str) -> bool:
        """Attempt to write *msg* to the serial port (caller holds lock).

        Returns *True* on success, *False* on failure.
        """
        try:
            self._serial.write(msg)
            log.debug("TX -> %s", command)
            return True
        except Exception as exc:
            log.error("Serial write error: %s", exc)
            return False

    def read_line(self) -> Optional[str]:
        """Read one newline-terminated response from the STM32.

        Returns *None* on timeout, if the port is closed, or on error.
        """
        with self._lock:
            if not self._is_connected_unlocked():
                return None
            try:
                raw = self._serial.readline()
            except Exception as exc:
                log.error("Serial read error: %s", exc)
                return None

        if raw:
            line = raw.decode("ascii", errors="replace").strip()
            log.debug("RX <- %s", line)
            return line
        return None

    # ── context manager ──────────────────────────

    def __enter__(self) -> "SerialComm":
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── helpers ──────────────────────────────────

    @staticmethod
    def _import_serial():
        """Lazily import pyserial, raising a clear error if missing."""
        try:
            import serial  # type: ignore[import-untyped]
            return serial
        except ImportError:
            raise RuntimeError(
                "pyserial is not installed.  "
                "Install it with:  pip install pyserial"
            )


# ── standalone test ──────────────────────────────

if __name__ == "__main__":
    import io

    print("=" * 60)
    print("  SerialComm -- standalone test suite")
    print("=" * 60)

    passed = 0
    failed = 0

    def _assert(cond: bool, tag: str) -> None:
        global passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {tag}")
        else:
            failed += 1
            print(f"  [FAIL] {tag}")

    # ── 1. Config defaults ───────────────────────
    print("\n[1] Constructor reads config defaults")
    comm = SerialComm()
    _assert(comm._port == config.SERIAL_PORT, f"port = {comm._port}")
    _assert(comm._baudrate == config.SERIAL_BAUDRATE, f"baudrate = {comm._baudrate}")
    _assert(comm._timeout == config.SERIAL_TIMEOUT, f"timeout = {comm._timeout}")
    _assert(comm._terminator == config.COMMAND_TERMINATOR, "terminator = newline")

    # ── 2. Custom overrides ──────────────────────
    print("\n[2] Constructor accepts custom overrides")
    comm2 = SerialComm(port="/dev/ttyAMA0", baudrate=9600, timeout=2.5)
    _assert(comm2._port == "/dev/ttyAMA0", f"port = {comm2._port}")
    _assert(comm2._baudrate == 9600, f"baudrate = {comm2._baudrate}")
    _assert(comm2._timeout == 2.5, f"timeout = {comm2._timeout}")

    # ── 3. is_connected before open ──────────────
    print("\n[3] is_connected() before open() returns False")
    comm3 = SerialComm()
    _assert(comm3.is_connected() is False, "not connected before open()")
    _assert(comm3.is_open is False, "is_open property also False")

    # ── 4. send on closed port (no auto_reconnect) ──
    print("\n[4] send() on closed port drops command gracefully")
    sent_commands: list[str] = []
    comm4 = SerialComm(auto_reconnect=False)
    # Should not crash -- just log a warning
    comm4.send("FWD:150")
    _assert(comm4.is_connected() is False, "still not connected after failed send")

    # ── 5. close() is idempotent ─────────────────
    print("\n[5] close() is idempotent (no crash on double-close)")
    comm5 = SerialComm()
    comm5.close()
    comm5.close()
    _assert(True, "double close() did not crash")

    # ── 6. Mock serial for send/read ─────────────
    print("\n[6] send() and read_line() with mock serial port")

    class _MockSerial:
        """Minimal mock matching pyserial.Serial's interface."""
        def __init__(self):
            self.is_open = True
            self.written: list[bytes] = []
            self._read_buffer = io.BytesIO()

        def write(self, data: bytes) -> int:
            self.written.append(data)
            return len(data)

        def readline(self) -> bytes:
            return self._read_buffer.readline()

        def close(self) -> None:
            self.is_open = False

        def stage_response(self, text: str) -> None:
            """Stage a response for the next readline() call."""
            self._read_buffer = io.BytesIO((text + "\n").encode("ascii"))

    comm6 = SerialComm()
    mock = _MockSerial()
    comm6._serial = mock  # inject mock

    # Send a command
    comm6.send("FWD:150")
    _assert(len(mock.written) == 1, f"one write: {len(mock.written)}")
    _assert(mock.written[0] == b"FWD:150\n", f"payload = {mock.written[0]!r}")

    # Send another
    comm6.send("LEFT:80")
    _assert(len(mock.written) == 2, f"two writes: {len(mock.written)}")
    _assert(mock.written[1] == b"LEFT:80\n", f"payload = {mock.written[1]!r}")

    # Read a response
    mock.stage_response("OK")
    resp = comm6.read_line()
    _assert(resp == "OK", f"read_line() = {resp!r}")

    # Read with no data
    mock.stage_response("")
    resp = comm6.read_line()
    # Empty string read -> stripped -> empty -> None
    _assert(resp is None or resp == "", f"empty read = {resp!r}")

    # ── 7. Newline termination ───────────────────
    print("\n[7] Commands are newline-terminated")
    comm7 = SerialComm()
    mock7 = _MockSerial()
    comm7._serial = mock7
    comm7.send("STOP:0")
    _assert(
        mock7.written[0].endswith(b"\n"),
        f"ends with newline: {mock7.written[0]!r}",
    )

    # ── 8. Thread safety ─────────────────────────
    print("\n[8] Thread-safe concurrent sends")
    comm8 = SerialComm()
    mock8 = _MockSerial()
    comm8._serial = mock8
    errors: list[str] = []

    def _thread_send(cmd: str, count: int) -> None:
        try:
            for i in range(count):
                comm8.send(f"{cmd}:{i}")
        except Exception as exc:
            errors.append(str(exc))

    threads = [
        threading.Thread(target=_thread_send, args=("FWD", 50)),
        threading.Thread(target=_thread_send, args=("LEFT", 50)),
        threading.Thread(target=_thread_send, args=("RIGHT", 50)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    _assert(len(errors) == 0, f"no thread errors: {errors}")
    _assert(
        len(mock8.written) == 150,
        f"150 commands sent from 3 threads: {len(mock8.written)}",
    )

    # ── 9. Auto-reconnect on write failure ───────
    print("\n[9] Auto-reconnect on write failure")

    class _FailOnceSerial:
        """Mock that fails the first write, then succeeds."""
        def __init__(self):
            self.is_open = True
            self.written: list[bytes] = []
            self._fail_next = True

        def write(self, data: bytes) -> int:
            if self._fail_next:
                self._fail_next = False
                raise OSError("Simulated disconnect")
            self.written.append(data)
            return len(data)

        def readline(self) -> bytes:
            return b""

        def close(self) -> None:
            self.is_open = False

    comm9 = SerialComm(auto_reconnect=True)
    fail_mock = _FailOnceSerial()
    comm9._serial = fail_mock

    # Monkey-patch reconnect to simulate a successful re-open.
    # Use a mutable container to track the call from inside the closure.
    _reconnect_state = {"called": False}

    def _fake_reconnect() -> bool:
        _reconnect_state["called"] = True
        # Simulate a successful reconnect: install a fresh mock
        fresh_mock = _MockSerial()
        comm9._serial = fresh_mock
        return True

    comm9.reconnect = _fake_reconnect  # type: ignore[assignment]

    comm9.send("FWD:200")
    _assert(_reconnect_state["called"], "reconnect() was triggered by write failure")

    # ── 10. is_connected reflects state ──────────
    print("\n[10] is_connected() reflects port state")
    comm10 = SerialComm()
    _assert(comm10.is_connected() is False, "False when no serial")
    mock10 = _MockSerial()
    comm10._serial = mock10
    _assert(comm10.is_connected() is True, "True when mock is open")
    mock10.is_open = False
    _assert(comm10.is_connected() is False, "False when mock.is_open = False")

    # ── 11. Context manager ──────────────────────
    print("\n[11] Context manager calls close()")
    comm11 = SerialComm()
    mock11 = _MockSerial()
    comm11._serial = mock11
    # Simulate __exit__
    comm11.__exit__(None, None, None)
    _assert(comm11._serial is None, "serial set to None after __exit__")
    _assert(mock11.is_open is False, "mock closed by __exit__")

    # ── 12. read_line on closed port ─────────────
    print("\n[12] read_line() on closed port returns None")
    comm12 = SerialComm()
    resp = comm12.read_line()
    _assert(resp is None, f"read_line on closed port = {resp!r}")

    # ── summary ──────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  -- all clear!")
    print("=" * 60)
