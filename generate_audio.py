import wave
import struct
import math

def generate_wav(filename, duration_s, frequency_hz, sample_rate=44100):
    """Generates a sine wave and saves it as a WAV file."""
    num_samples = int(duration_s * sample_rate)
    amplitude = 32767  # Max amplitude for 16-bit audio

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        for i in range(num_samples):
            # Calculate sample value
            angle = 2 * math.pi * i * frequency_hz / sample_rate
            value = int(amplitude * math.sin(angle))
            # Pack as 16-bit signed integer
            packed_value = struct.pack('<h', value)
            wav_file.writeframes(packed_value)
    print(f"Generated '{filename}'")

def generate_ir(filename, sample_rate=44100):
    """Generates a simple echo impulse response."""
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        # Main tap
        wav_file.writeframes(struct.pack('<h', 32767))
        # Silence for ~0.25 seconds
        for _ in range(int(sample_rate * 0.25) - 2):
            wav_file.writeframes(struct.pack('<h', 0))
        # Echo tap
        wav_file.writeframes(struct.pack('<h', 16000))

    print(f"Generated '{filename}'")


if __name__ == "__main__":
    # Generate a 3-second, 440 Hz (A4 note) tone
    generate_wav("input.wav", duration_s=3, frequency_hz=440)
    # Generate a simple impulse response for an echo effect
    generate_ir("ir.wav")