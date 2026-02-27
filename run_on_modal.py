import modal
import sys
import os
import subprocess

# 1. Build the environment AND sync the local directory
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements("requirements.txt")
    # NEW WAY: Attach your local directory directly to the image
    .add_local_dir(".", remote_path="/root/workspace") 
)

# 2. Create the Modal App
app = modal.App("qa-experiment-runner")

# 3. Create a persistent Volume to store your JSONs and model checkpoints
output_volume = modal.Volume.from_name("experiment-outputs", create_if_missing=True)

# 4. Define the remote function
@app.function(
    image=image,
    gpu="A100",
    cpu=8.0,
    timeout=14400,    
    # mounts=[...] is completely removed!
    # We only attach the Volume for saving outputs
    volumes={"/root/workspace/part4/outputs": output_volume}
)
def run_experiments():
    os.chdir("/root/workspace")
    sys.path.insert(0, "/root/workspace")

    # Ensure the outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    print("🚀 Running setup_experiments.py...")
    try:
        subprocess.run(["python", "part4/setup_datasets.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in setup: {e}")
        return

    print("🚀 Running train_baseline.py...")
    try:
        subprocess.run(["python", "part4/train_baseline.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in training: {e}")
        return

    # Commit the volume at the end to ensure all files are saved to the cloud!
    output_volume.commit()
    print("✅ All scripts finished and outputs saved to the Modal Volume!")

# 5. Local entry point
@app.local_entrypoint()
def main():
    print("Deploying container to Modal...")
    run_experiments.remote()