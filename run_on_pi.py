import numpy as np
import time
import pywt
# Use tflite_runtime on the actual Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class ECGInferenceEngine:
    """Handles TFLite model loading and inference."""
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run_inference(self, ecg_segment):
        """Runs inference on a single ECG segment."""
        input_data = np.expand_dims(ecg_segment, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        start_time = time.perf_counter()
        self.interpreter.invoke()
        latency = (time.perf_counter() - start_time) * 1000
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(output_data), latency

def generate_and_compress_saliency_map(ecg_segment):
    """Simulates saliency map generation and wavelet compression."""
    # Simulate a saliency map (e.g., highlighting areas with high variance)
    saliency_map = np.abs(ecg_segment - np.mean(ecg_segment))
    saliency_map /= np.max(saliency_map)
    # Compress using Daubechies-4 wavelet transform
    coeffs = pywt.wavedec(saliency_map, 'db4', level=3)
    compressed_size_kb = sum(c.nbytes for c in coeffs) / 1024
    return compressed_size_kb

def qrs_triggered_monitoring(inference_engine, ecg_stream_simulator):
    """Simulates QRS-triggered duty cycling for energy efficiency."""
    print("\nStarting QRS-triggered monitoring loop...")
    for segment, is_r_peak in ecg_stream_simulator:
        if is_r_peak:
            print("R-Peak Detected! Waking up for inference.")
            prediction, latency = inference_engine.run_inference(segment)
            print(f" > Prediction: Class {prediction}, Latency: {latency:.2f} ms")
            compressed_size = generate_and_compress_saliency_map(segment)
            print(f" > Saliency map compressed size: {compressed_size:.2f} KB")
        else:
            # In a real device, the CPU would be in a low-power state.
            time.sleep(0.1)

def main():
    model_path = 'feel_ecg_model.tflite'
    engine = ECGInferenceEngine(model_path)
    X = np.load('data/ecg_segments.npy')
    def ecg_stream_simulator():
        for i in range(20):
            yield X[i], True
            yield None, False
            yield None, False
    qrs_triggered_monitoring(engine, ecg_stream_simulator())

if __name__ == '__main__':
    main()