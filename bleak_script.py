import asyncio
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from bleak import BleakScanner, BleakClient


# ---------------- BLE CONFIG ---------------- #

DEVICE_ADDRESS = "E4:B3:23:B0:5F:C6"

DATA_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
CONTROL_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"

SAMPLE_SIZE = 13
NUM_CHANNELS = 6

# ---------------- SIGNAL CONFIG ---------------- #

SAMPLING_RATE = 2000

# show last 2 seconds
DISPLAY_SECONDS = 2

BUFFER_SIZE = SAMPLING_RATE * DISPLAY_SECONDS

# plotting every Nth point
DOWNSAMPLE = 4

# rolling buffers
buffers = [
    deque([0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
    for _ in range(NUM_CHANNELS)
]

packet_count = 0
start_time = time.time()


# ---------------- PACKET DECODER ---------------- #

def decode_packet(data):

    global packet_count

    packet_count += 1

    for i in range(0, len(data), SAMPLE_SIZE):

        chunk = data[i:i + SAMPLE_SIZE]

        if len(chunk) != SAMPLE_SIZE:
            continue

        for ch in range(NUM_CHANNELS):

            idx1 = 1 + ch * 2
            idx2 = 2 + ch * 2

            if idx2 >= len(chunk):
                continue

            high = chunk[idx1]
            low = chunk[idx2]

            value = (high << 8) | low

            buffers[ch].append(value)

    # print stats once/sec only
    elapsed = time.time() - start_time

    if elapsed >= 1:

        print(f"Packets/sec: {packet_count}")

        packet_count = 0

        globals()['start_time'] = time.time()


def notification_handler(sender, data):

    decode_packet(data)


# ---------------- BLE TASK ---------------- #

async def ble_task():

    devices = await BleakScanner.discover()

    target = None

    for d in devices:

        print(d.name, d.address)

        if d.address.upper() == DEVICE_ADDRESS:
            target = d

    if target is None:
        print("Device not found")
        return

    async with BleakClient(target.address) as client:

        print("Connected")

        await client.start_notify(
            DATA_UUID,
            notification_handler
        )

        await client.write_gatt_char(
            CONTROL_UUID,
            b"START"
        )

        print("Streaming at 2000 Hz...\n")

        while True:
            await asyncio.sleep(1)


def start_ble_loop():

    asyncio.run(ble_task())


# ---------------- PLOTTING ---------------- #

fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(12, 8))

lines = []

x = list(range(0, BUFFER_SIZE, DOWNSAMPLE))

for ch in range(NUM_CHANNELS):

    line, = axes[ch].plot(
        x,
        [0] * len(x)
    )

    axes[ch].set_xlim(0, BUFFER_SIZE)

    # adjust depending on ADC range
    axes[ch].set_ylim(0, 4096)

    axes[ch].set_ylabel(f"CH {ch+1}")

    lines.append(line)

axes[-1].set_xlabel("Samples")


def update(frame):

    for ch in range(NUM_CHANNELS):

        y = list(buffers[ch])[::DOWNSAMPLE]

        lines[ch].set_ydata(y)

    return lines


# ---------------- START THREAD ---------------- #

ble_thread = threading.Thread(
    target=start_ble_loop,
    daemon=True
)

ble_thread.start()

# redraw every 50 ms
ani = FuncAnimation(
    fig,
    update,
    interval=50,
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()

plt.show()