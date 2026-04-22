"""TCP socket bridge to Node.js Mineflayer bot.

Sends JSON commands over a TCP socket and receives game state responses.
Uses newline-delimited JSON framing for reliable message boundaries.
"""

from __future__ import annotations

import json
import socket
import time


class BridgeError(Exception):
    """Raised when the bridge encounters a communication error."""


class MinecraftBridge:
    """TCP client that communicates with the Mineflayer bot.

    Protocol (newline-delimited JSON):
        Python -> Node.js:  {"cmd": "get_state"}
        Node.js -> Python:  {"type": "state", "health": 20, ...}

        Python -> Node.js:  {"cmd": "action", "id": 5}
        Node.js -> Python:  {"type": "state", ...}  (state after action)

        Python -> Node.js:  {"cmd": "respawn"}
        Node.js -> Python:  {"type": "state", ...}

        Python -> Node.js:  {"cmd": "disconnect"}
        Node.js -> Python:  {"type": "ack"}

    Args:
        host: Address of the Node.js bridge server.
        port: Port of the Node.js bridge server.
        timeout: Socket timeout in seconds.
        max_retries: Connection retry attempts.
        retry_delay: Seconds between retries.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        timeout: float = 10.0,
        max_retries: int = 10,
        retry_delay: float = 2.0,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._sock: socket.socket | None = None
        self._buffer = ""

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        """Connect to the Node.js bridge, retrying on failure."""
        for attempt in range(1, self._max_retries + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.connect((self._host, self._port))
                self._sock = sock
                self._buffer = ""
                return
            except (ConnectionRefusedError, OSError) as exc:
                if attempt == self._max_retries:
                    raise BridgeError(
                        f"Failed to connect to bridge at "
                        f"{self._host}:{self._port} after {self._max_retries} "
                        f"attempts: {exc}"
                    ) from exc
                time.sleep(self._retry_delay)

    def close(self) -> None:
        """Send disconnect command and close the socket."""
        if self._sock is None:
            return
        try:
            self._send({"cmd": "disconnect"})
        except (OSError, BridgeError):
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None
        self._buffer = ""

    def get_state(self) -> dict:
        """Request current game state from the bot.

        Returns:
            Dict with keys: health, food, xp_level, xp_points, position,
            yaw, pitch, on_ground, is_in_water, is_raining, time_of_day,
            light_level, altitude, block_composition, entities, inventory,
            alive.
        """
        self._send({"cmd": "get_state"})
        response = self._recv()
        if response.get("type") == "error":
            raise BridgeError(f"Bot error: {response.get('message')}")
        return response

    def send_action(self, action_id: int) -> dict:
        """Send an action to execute and return the resulting state.

        The bot executes the action, waits for the action duration ticks,
        then returns the new game state. Retries once on timeout before
        giving up, since the bot may be slow during combat/chunk loading.

        Args:
            action_id: Discrete action index (see actions.py).

        Returns:
            Game state dict after the action completes.
        """
        self._send({"cmd": "action", "id": action_id})
        try:
            response = self._recv()
        except BridgeError as exc:
            if "timed out" in str(exc):
                # Bot may be slow (combat/respawn/chunk loading). Try get_state
                # as a fallback — the action may have already executed.
                try:
                    self._send({"cmd": "get_state"})
                    response = self._recv()
                except BridgeError:
                    raise exc  # Both failed, raise original
            else:
                raise
        if response.get("type") == "error":
            raise BridgeError(f"Action error: {response.get('message')}")
        return response

    def send_composite_action(
        self, movement: int, look: int, combat: int,
    ) -> dict:
        """Send a factored composite action and return the resulting state.

        The bot executes movement, look, and combat axes in parallel within
        one tick window. Uses the same timeout/retry logic as send_action().

        Args:
            movement: Movement axis (0-6).
            look: Look axis (0-5).
            combat: Combat axis (0-3).

        Returns:
            Game state dict after the action completes.
        """
        self._send({
            "cmd": "action",
            "movement": movement,
            "look": look,
            "combat": combat,
        })
        try:
            response = self._recv()
        except BridgeError as exc:
            if "timed out" in str(exc):
                try:
                    self._send({"cmd": "get_state"})
                    response = self._recv()
                except BridgeError:
                    raise exc
            else:
                raise
        if response.get("type") == "error":
            raise BridgeError(f"Action error: {response.get('message')}")
        return response

    def respawn(self) -> dict:
        """Request the bot to respawn after death.

        Returns:
            Game state dict after respawn.
        """
        self._send({"cmd": "respawn"})
        response = self._recv()
        if response.get("type") == "error":
            raise BridgeError(f"Respawn error: {response.get('message')}")
        return response

    def send_bot_control(self, enabled: bool) -> dict:
        """Toggle FPI agent control of the character.

        When enabled, the relay suppresses the real client's movement/attack
        packets and the FPI agent's actions are injected instead. When
        disabled, the real client resumes normal play and actions sent via
        the bridge are ignored.

        Args:
            enabled: True to let the FPI agent drive, False for user control.

        Returns:
            Ack response dict with ``bot_control_active`` key.
        """
        self._send({"cmd": "bot_control", "enabled": enabled})
        response = self._recv()
        if response.get("type") == "error":
            raise BridgeError(f"Bot control error: {response.get('message')}")
        return response

    def _send(self, msg: dict) -> None:
        """Send a JSON message followed by newline."""
        if self._sock is None:
            raise BridgeError("Not connected")
        try:
            data = json.dumps(msg) + "\n"
            self._sock.sendall(data.encode("utf-8"))
        except OSError as exc:
            self._sock = None
            raise BridgeError(f"Send failed: {exc}") from exc

    def _recv(self) -> dict:
        """Receive a newline-delimited JSON message."""
        if self._sock is None:
            raise BridgeError("Not connected")

        while "\n" not in self._buffer:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout as exc:
                raise BridgeError("Receive timed out") from exc
            except OSError as exc:
                self._sock = None
                raise BridgeError(f"Receive failed: {exc}") from exc
            if not chunk:
                self._sock = None
                raise BridgeError("Connection closed by bot")
            self._buffer += chunk.decode("utf-8")

        line, self._buffer = self._buffer.split("\n", 1)
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"Invalid JSON from bot: {line[:200]}") from exc
