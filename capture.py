"""Frame capture from a USB HDMI capture card via OpenCV.

Supports both Linux V4L2 device paths and macOS device indices.
"""

import logging
import os
import pathlib
import platform

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def _find_capture_card_index(expected_name: str = config.CAPTURE_DEVICE_NAME) -> int | None:
    """Find the OpenCV index for a capture device by its AVFoundation name.

    Replicates OpenCV's cap_avfoundation_mac.mm device enumeration:
    discovers video + muxed devices, sorts by uniqueID, returns position.
    Uses AVCaptureDeviceDiscoverySession (required on modern macOS to
    find external USB devices).
    """
    try:
        from AVFoundation import (
            AVCaptureDeviceDiscoverySession,
            AVCaptureDevicePositionUnspecified,
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeExternal,
            AVMediaTypeMuxed,
            AVMediaTypeVideo,
        )

        device_types = [
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeExternal,
        ]

        video_session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            device_types, AVMediaTypeVideo, AVCaptureDevicePositionUnspecified
        )
        muxed_session = AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
            device_types, AVMediaTypeMuxed, AVCaptureDevicePositionUnspecified
        )

        seen_ids = set()
        all_devices = []
        for d in list(video_session.devices()) + list(muxed_session.devices()):
            uid = d.uniqueID()
            if uid not in seen_ids:
                seen_ids.add(uid)
                all_devices.append(d)

        all_devices.sort(key=lambda d: d.uniqueID())

        for i, device in enumerate(all_devices):
            if device.localizedName() == expected_name:
                logger.info(
                    "Found '%s' at OpenCV index %d (uniqueID=%s)",
                    expected_name, i, device.uniqueID(),
                )
                return i

        available = [(d.localizedName(), d.uniqueID()) for d in all_devices]
        logger.warning("Device '%s' not found. Available: %s", expected_name, available)

    except ImportError:
        logger.warning("pyobjc-framework-AVFoundation not installed, cannot enumerate devices")
    except Exception as e:
        logger.warning("Failed to enumerate AVFoundation devices: %s", e)

    return None


class FrameCapture:
    """Manages the OpenCV VideoCapture device."""

    def __init__(self, device=config.CAPTURE_DEVICE):
        self._device = device
        self._cap = None

    def open(self):
        """Open the capture device."""
        if platform.system() == "Darwin" and isinstance(self._device, int):
            # Find capture card by name to avoid index instability
            idx = _find_capture_card_index()
            if idx is not None:
                self._device = idx
            else:
                logger.warning(
                    "Could not find '%s', falling back to index %d",
                    config.CAPTURE_DEVICE_NAME, self._device,
                )

        logger.info("Opening capture device: %s", self._device)

        if isinstance(self._device, int):
            if platform.system() == "Darwin":
                self._cap = cv2.VideoCapture(self._device, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(self._device)
        else:
            self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open capture device: {self._device}{self._open_failure_hint()}"
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != config.CAPTURE_WIDTH or actual_h != config.CAPTURE_HEIGHT:
            raise RuntimeError(
                f"Capture device resolution mismatch: got {actual_w}x{actual_h}, "
                f"expected {config.CAPTURE_WIDTH}x{config.CAPTURE_HEIGHT}. "
                f"Check device settings or update CAPTURE_WIDTH/CAPTURE_HEIGHT in config.py."
            )
        logger.info("Capture device opened. Resolution: %dx%d", actual_w, actual_h)

    def _open_failure_hint(self) -> str:
        """Explain a likely cause when a V4L2 device path won't open.

        The bare "Failed to open" is opaque; the usual culprits on a Linux/LXC
        deploy are a missing passthrough or a permission denied node (common in
        unprivileged Proxmox containers where the host video device stays
        root:video 0660 and container root maps to an unprivileged uid).
        """
        if not isinstance(self._device, str):
            return ""
        if not os.path.exists(self._device):
            return ", device path does not exist (check USB passthrough to the container)."
        if not os.access(self._device, os.R_OK | os.W_OK):
            return (
                ", the node exists but isn't readable/writable by this user. On an "
                "unprivileged Proxmox LXC, fix the HOST permissions with a udev rule "
                '(KERNEL=="video[0-9]*", MODE="0666") or chmod the host node.'
            )
        return ", node is accessible but the capture backend couldn't open it (device busy or unsupported format?)."

    def grab_frame(self) -> np.ndarray | None:
        """Grab a single frame. Returns BGR numpy array or None on failure."""
        if self._cap is None or not self._cap.isOpened():
            logger.error("Capture device not open.")
            return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Failed to grab frame.")
            return None

        return frame

    def close(self):
        """Release the capture device."""
        if self._cap:
            self._cap.release()
            self._cap = None
            logger.info("Capture device closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


def is_sync_artifact_frame(frame: np.ndarray) -> bool:
    """Detect the saturated green frame USB capture cards emit while syncing to HDMI.

    The artifact frame has an extreme color signature: mean green channel near
    255, blue and red near 0. Real broadcast content never looks like that, 
    even the greenest court advertisement keeps nontrivial blue/red components.
    """
    if frame is None or frame.size == 0:
        return False
    # BGR ordering in OpenCV
    mean_b, mean_g, mean_r = frame.reshape(-1, 3).mean(axis=0)
    return mean_g > 180 and mean_b < 60 and mean_r < 60


def extract_corner_rois(
    frame: np.ndarray,
    roi_w: int = config.CORNER_ROI_WIDTH,
    roi_h: int = config.CORNER_ROI_HEIGHT,
) -> dict[str, np.ndarray]:
    """Extract ROIs from all four corners of a frame.

    Returns a dict keyed by corner name: "top_left", "top_right",
    "bottom_left", "bottom_right".
    """
    h, w = frame.shape[:2]
    return {
        "top_left": frame[0:roi_h, 0:roi_w],
        "top_right": frame[0:roi_h, w - roi_w : w],
        "bottom_left": frame[h - roi_h : h, 0:roi_w],
        "bottom_right": frame[h - roi_h : h, w - roi_w : w],
    }


# ----------------------------------------------------------------------
# Quick test: grab and display a frame
# ----------------------------------------------------------------------

def _test():
    # Dev-only helper: dump a frame + corner ROIs into the gitignored scratch/
    # dir so they don't litter the repo root.
    scratch = pathlib.Path(__file__).parent / "scratch"
    scratch.mkdir(exist_ok=True)
    with FrameCapture() as cap:
        print("Grabbing a test frame...")
        frame = cap.grab_frame()
        if frame is not None:
            print(f"Got frame: {frame.shape}")
            frame_path = scratch / "test_frame.png"
            cv2.imwrite(str(frame_path), frame)
            print(f"Saved to {frame_path}")

            rois = extract_corner_rois(frame)
            for name, roi in rois.items():
                roi_path = scratch / f"test_roi_{name}.png"
                cv2.imwrite(str(roi_path), roi)
                print(f"Saved ROI {name}: {roi.shape} -> {roi_path}")
        else:
            print("No frame captured.")


if __name__ == "__main__":
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )
    _test()
