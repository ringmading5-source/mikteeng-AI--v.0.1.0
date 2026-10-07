import math, tempfile, wave
from pathlib import Path
from tbin.audio import acoustic_trajectory

def test_acoustic_trajectory_reads_pcm_wav():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"tone.wav"
        rate=8000
        samples=[int(10000*math.sin(2*math.pi*440*i/rate)) for i in range(rate//2)]
        with wave.open(str(p),"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
            import struct
            w.writeframes(b"".join(struct.pack("<h",x) for x in samples))
        trajectory=acoustic_trajectory(p)
        assert trajectory
        assert all(0 <= x <= 255 for x in trajectory)
