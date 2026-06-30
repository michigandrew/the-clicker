"""Playback device control.

The detection side of this project (capture + logo detection) is device-
agnostic, it only looks at the HDMI signal. The ONLY platform specific piece
is how you *act* on a detected commercial: mute the audio and (for delayed/DVR
viewing) skip forward. That behavior lives behind the `PlaybackController`
interface, so supporting a device other than an Apple TV is a matter of adding
one class, see `PlaybackController` and `make_controller`.

The included backend is `AppleTVController` (pyatv).
"""

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pyatv

import config

logger = logging.getLogger(__name__)


class PlaybackMode(Enum):
    LIVE = "live"
    DELAYED = "delayed"


def _slugify(text: str) -> str:
    """Turn a freeform string into a filesystem safe lowercase slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass
class PlaybackStatus:
    """Snapshot of current playback state."""
    title: str
    position: float       # seconds
    total_time: float     # seconds
    device_state: str

    @property
    def progress_ratio(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.position / self.total_time

    @property
    def mode(self) -> PlaybackMode:
        if self.progress_ratio > config.LIVE_THRESHOLD:
            return PlaybackMode.LIVE
        return PlaybackMode.DELAYED

    @property
    def channel_slug(self) -> str | None:
        """Best effort channel name derived from the title metadata.

        Live TV titles often look like "NBC", "ESPN", "CBS Sports", etc.
        Returns a slug suitable for use as a profile filename, or None if
        no title is available.
        """
        if not self.title:
            return None
        return _slugify(self.title)


class PlaybackController(ABC):
    """Interface for controlling whatever device plays the video.

    Implement this to support a device other than an Apple TV, a CEC controlled
    TV/soundbar, a Roku, a Fire TV, an IR blaster, etc., then register it in
    `make_controller` and point `config.CONTROLLER_BACKEND` at it.

    All methods are async. Notes for non Apple TV backends:
    - `get_playback_status` powers the live vs delayed decision via the
      returned `PlaybackStatus.mode`. If your device can't report playback
      position, return zeros and set `PLAYBACK_MODE_OVERRIDE="live"` so the
      engine only ever mutes (never tries to skip).
    - If your device can't skip, implement `skip_forward`/`seek_forward` as
      no ops and run in live (mute only) mode as above.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish the connection to the device."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the connection."""

    @abstractmethod
    async def get_playback_status(self) -> "PlaybackStatus":
        """Return a snapshot of current playback (drives live vs delayed)."""

    @abstractmethod
    async def mute(self) -> None:
        """Silence the audio."""

    @abstractmethod
    async def unmute(self) -> None:
        """Restore the audio."""

    @abstractmethod
    async def skip_forward(self, count: int = 1) -> None:
        """Jump forward (delayed/DVR viewing). No op if unsupported."""

    @abstractmethod
    async def seek_forward(self, taps: int = 5) -> None:
        """Finer grained forward seek. No op if unsupported."""


class AppleTVController(PlaybackController):
    """Async wrapper around pyatv for controlling the Apple TV."""

    def __init__(self, device_id: str = config.APPLE_TV_ID):
        self._device_id = device_id
        self._atv = None
        self._remote = None
        self._audio = None

    async def connect(self):
        """Scan for and connect to the Apple TV."""
        logger.info("Scanning for Apple TV (id=%s)...", self._device_id)
        loop = asyncio.get_event_loop()
        atvs = await pyatv.scan(loop, identifier=self._device_id, timeout=5)
        if not atvs:
            raise ConnectionError(
                f"Apple TV with id {self._device_id} not found on network"
            )

        conf = atvs[0]

        # Load saved credentials
        creds_file = os.path.join(os.path.dirname(__file__), "credentials.json")
        if os.path.exists(creds_file):
            with open(creds_file) as f:
                creds = json.load(f)
            for service in conf.services:
                key = str(service.protocol)
                if key in creds:
                    service.credentials = creds[key]
                    logger.info("Loaded credentials for %s", key)

        logger.info("Connecting to %s (%s)...", conf.name, conf.address)
        self._atv = await pyatv.connect(conf, loop)
        self._remote = self._atv.remote_control
        self._audio = self._atv.audio
        logger.info("Connected to Apple TV.")

    async def disconnect(self):
        """Close the connection."""
        if self._atv:
            self._atv.close()
            self._atv = None
            self._remote = None
            self._audio = None
            logger.info("Disconnected from Apple TV.")

    # ------------------------------------------------------------------
    # Playback status
    # ------------------------------------------------------------------

    async def get_playback_status(self) -> PlaybackStatus:
        """Query current playback info."""
        playing = self._atv.metadata
        status = await playing.playing()
        result = PlaybackStatus(
            title=status.title or "",
            position=status.position or 0,
            total_time=status.total_time or 0,
            device_state=str(status.device_state),
        )
        logger.debug(
            "Playback: title=%r position=%.1fs total=%.1fs ratio=%.3f mode=%s",
            result.title,
            result.position,
            result.total_time,
            result.progress_ratio,
            result.mode.value,
        )
        return result

    # ------------------------------------------------------------------
    # Volume control
    # ------------------------------------------------------------------

    async def _volume_press(self, direction: str, count: int):
        """Press volume_up or volume_down `count` times."""
        fn = getattr(self._remote, direction)
        for i in range(count):
            await fn()
            logger.debug("%s (%d/%d)", direction, i + 1, count)
            if i < count - 1:
                await asyncio.sleep(config.VOLUME_COMMAND_DELAY)

    async def mute(self):
        """Mute by spamming volume_down."""
        logger.info("Muting: %d x volume_down", config.MUTE_STEPS)
        await self._volume_press("volume_down", config.MUTE_STEPS)
        logger.info("Muted.")

    async def unmute(self):
        """Restore volume by spamming volume_up."""
        logger.info("Unmuting: %d x volume_up", config.UNMUTE_STEPS)
        await self._volume_press("volume_up", config.UNMUTE_STEPS)
        logger.info("Unmuted.")

    # ------------------------------------------------------------------
    # Skip forward
    # ------------------------------------------------------------------

    async def skip_forward(self, count: int = 1):
        """Press skip_forward `count` times (~15s each)."""
        logger.info("Skipping forward: %d x skip_forward", count)
        for i in range(count):
            await self._remote.skip_forward()
            logger.debug("skip_forward (%d/%d)", i + 1, count)
            if i < count - 1:
                await asyncio.sleep(config.SKIP_COMMAND_DELAY)
        logger.info("Skip complete.")

    async def seek_forward(self, taps: int = 5):
        """Seek forward using trackpad right + select (for sponsored overlays).

        Simulates pressing the right side of the Siri Remote trackpad
        to move the seek bar, then select to confirm.
        """
        logger.info("Seek forward: %d x right + select", taps)
        for i in range(taps):
            await self._remote.right()
            await asyncio.sleep(0.15)
        await self._remote.select()
        logger.info("Seek complete.")


# ----------------------------------------------------------------------
# Backend selection
# ----------------------------------------------------------------------

def make_controller(backend: str | None = None) -> PlaybackController:
    """Construct the configured playback controller.

    `backend` defaults to `config.CONTROLLER_BACKEND`. To add support for a new
    device, implement `PlaybackController` above and add a branch here.
    """
    backend = (backend or config.CONTROLLER_BACKEND).lower()
    if backend == "appletv":
        return AppleTVController()
    raise ValueError(
        f"Unknown CONTROLLER_BACKEND {backend!r}. Built in backends: 'appletv'. "
        "Add your own by subclassing PlaybackController and registering it in "
        "make_controller()."
    )


# ----------------------------------------------------------------------
# Convenience: run a quick test from the command line
# ----------------------------------------------------------------------

async def _test():
    ctrl = make_controller()
    await ctrl.connect()
    try:
        status = await ctrl.get_playback_status()
        print(f"Now playing: {status.title}")
        print(f"Position: {status.position:.0f}s / {status.total_time:.0f}s")
        print(f"Mode: {status.mode.value}")
    finally:
        await ctrl.disconnect()


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    asyncio.run(_test())
