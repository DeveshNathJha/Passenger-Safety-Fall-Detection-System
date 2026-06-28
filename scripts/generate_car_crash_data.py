"""
Synthesizes a 6-axis IMU dataset for 'Normal Driving' and 'Car Crash' events.
Output: data/raw/synthesized_car_crash_data/
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_driving_sequence(length=500, hz=50):
    """Normal driving: 1g on Z, small bumps on X,Y. Gyro has slow turns."""
    data = np.zeros((length, 6))
    # Accel
    data[:, 0] = np.random.normal(0, 0.5, length)    # X: side bumps
    data[:, 1] = np.random.normal(0, 1.0, length)    # Y: braking/accel
    data[:, 2] = np.random.normal(9.8, 0.5, length)  # Z: gravity + road bounce
    
    # Gyro
    data[:, 3] = np.random.normal(0, 2.0, length)
    data[:, 4] = np.random.normal(0, 5.0, length)    # Y/Z turning
    data[:, 5] = np.random.normal(0, 5.0, length)
    
    # Add a curve (turning corner)
    curve_start = np.random.randint(50, length-100)
    data[curve_start:curve_start+50, 5] += np.sin(np.linspace(0, np.pi, 50)) * 30.0
    
    return data

def generate_crash_sequence(length=300, hz=50):
    """Crash: Normal driving, then a massive >20g spike, then silence with new gravity vector."""
    data = generate_driving_sequence(length, hz)
    
    crash_idx = np.random.randint(50, length-100)
    
    # The Crash Impact (lasts ~100-200ms => 5-10 samples at 50Hz)
    impact_duration = np.random.randint(5, 12)
    # Massive deceleration on Y axis (forward/back) => e.g., 20g to 50g (200-500 m/s^2)
    data[crash_idx:crash_idx+impact_duration, 1] = np.random.normal(-300, 50, impact_duration)
    # Secondary impact on X/Z
    data[crash_idx:crash_idx+impact_duration, 0] = np.random.normal(100, 30, impact_duration)
    data[crash_idx:crash_idx+impact_duration, 2] = np.random.normal(50, 20, impact_duration)
    
    # Massive Gyroscope spin (car tumbling or spinning out)
    data[crash_idx:crash_idx+impact_duration, 3:] = np.random.normal(300, 100, (impact_duration, 3))
    
    # Post-Crash Silence (car comes to rest at a random weird angle)
    post_crash = data[crash_idx+impact_duration:]
    # New gravity vector (e.g. car is on its side)
    new_z = np.random.normal(0, 1) * 9.8
    new_x = np.random.normal(0, 1) * 9.8
    new_y = np.random.normal(0, 1) * 9.8
    
    post_crash[:, 0] = np.random.normal(new_x, 0.1, len(post_crash))
    post_crash[:, 1] = np.random.normal(new_y, 0.1, len(post_crash))
    post_crash[:, 2] = np.random.normal(new_z, 0.1, len(post_crash))
    
    # Gyro is dead silent (0)
    post_crash[:, 3:] = np.random.normal(0, 0.1, (len(post_crash), 3))
    
    return data

def save_csv(data, label, filename, output_dir):
    df = pd.DataFrame(data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    # Optional: add label column, but our preprocessor can just infer from folder
    df.to_csv(os.path.join(output_dir, filename), index=False)

if __name__ == "__main__":
    out_dir = "data/raw/synthetic_car_crash"
    os.makedirs(os.path.join(out_dir, "CRASH"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "NORMAL"), exist_ok=True)
    
    print("Generating CRASH sequences...")
    for i in range(200):
        data = generate_crash_sequence()
        save_csv(data, "CRASH", f"crash_{i:03d}.csv", os.path.join(out_dir, "CRASH"))
        
    print("Generating NORMAL sequences...")
    for i in range(200):
        data = generate_driving_sequence()
        save_csv(data, "NORMAL", f"normal_{i:03d}.csv", os.path.join(out_dir, "NORMAL"))
        
    print(f"Generated 400 sequences in {out_dir}")
