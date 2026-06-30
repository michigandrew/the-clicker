"""Pair with Apple TV to get Companion protocol credentials."""

import asyncio
import sys
import json
import os

import pyatv

import config


async def pair():
    loop = asyncio.get_event_loop()
    print("Scanning for Apple TV...")
    atvs = await pyatv.scan(loop, identifier=config.APPLE_TV_ID, timeout=5)
    if not atvs:
        print(f"Apple TV with id {config.APPLE_TV_ID} not found.")
        return

    conf = atvs[0]
    print(f"Found: {conf.name} at {conf.address}")
    print(f"Available services: {[str(s.protocol) for s in conf.services]}")

    # Pair with Companion protocol (needed for remote control)
    for protocol in [pyatv.const.Protocol.Companion, pyatv.const.Protocol.AirPlay]:
        print(f"\n--- Pairing {protocol} ---")
        pairing = await pyatv.pair(conf, protocol, loop)
        await pairing.begin()

        if pairing.device_provides_pin:
            pin = input("Enter PIN shown on Apple TV: ")
            pairing.pin(int(pin))
        else:
            print(f"Enter this PIN on Apple TV: {pairing.pin_code}")
            input("Press Enter when done...")

        await pairing.finish()

        if pairing.has_paired:
            creds = pairing.service.credentials
            print(f"Paired! Credentials: {creds}")
            # Save credentials
            creds_file = os.path.join(os.path.dirname(__file__), "credentials.json")
            existing = {}
            if os.path.exists(creds_file):
                with open(creds_file) as f:
                    existing = json.load(f)
            existing[str(protocol)] = creds
            with open(creds_file, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"Saved to {creds_file}")
        else:
            print("Pairing failed!")

        await pairing.close()


if __name__ == "__main__":
    asyncio.run(pair())
