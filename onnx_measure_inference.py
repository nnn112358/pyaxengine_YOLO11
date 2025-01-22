import argparse
import numpy as np
import onnxruntime
import time

def get_input_details(session):
    """Get input details from ONNX model"""
    model_inputs = session.get_inputs()
    input_names = [input.name for input in model_inputs]
    input_shapes = [input.shape for input in model_inputs]
    return input_names, input_shapes

def generate_random_input(input_shapes):
    """Generate random input data based on model input shapes"""
    random_inputs = {}
    for name, shape in input_shapes:
        # Generate random values between 0 and 255, then normalize to [0, 1]
        random_data = np.random.randint(0, 255, size=shape).astype(np.float32) / 255.0
        random_inputs[name] = random_data
    return random_inputs

def main():
    parser = argparse.ArgumentParser(description='Run ONNX inference with random input')
    parser.add_argument('model_path', type=str, help='Path to ONNX model file')
    parser.add_argument('--num_runs', type=int, default=100, 
                        help='Number of inference runs (default: 12)')
    parser.add_argument('--warmup_runs', type=int, default=10,
                        help='Number of warmup runs (default: 2)')
    args = parser.parse_args()

    # Load ONNX model
    try:
        session = onnxruntime.InferenceSession(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get input details
    input_names, input_shapes = get_input_details(session)
    input_shapes = list(zip(input_names, input_shapes))
    
    print(f"Model input shapes: {input_shapes}")

    # Warmup runs
    print(f"Performing {args.warmup_runs} warmup runs...")
    for _ in range(args.warmup_runs):
        random_inputs = generate_random_input(input_shapes)
        session.run(None, random_inputs)

    # Measurement runs
    print(f"Performing {args.num_runs} measurement runs...")
    inference_times = []
    
    for i in range(args.num_runs):
        random_inputs = generate_random_input(input_shapes)
        
        start_time = time.perf_counter()
        outputs = session.run(None, random_inputs)
        end_time = time.perf_counter()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_times.append(inference_time)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{args.num_runs} runs")

    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)

    print("\nInference Time Statistics:")
    print(f"Mean: {mean_time:.2f} ms")
    print(f"Std Dev: {std_time:.2f} ms")
    print(f"Min: {min_time:.2f} ms")
    print(f"Max: {max_time:.2f} ms")

if __name__ == "__main__":
    main()