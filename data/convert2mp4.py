import os
import glob
import subprocess

# Define source and destination directories
src_dir = "/scratch/cxm2114/thishome/Research/SALT/layered_neural_codec/data/Hollywood2/AVIClips"
dest_dir = "/scratch/cxm2114/thishome/Research/SALT/layered_neural_codec/data/"

# Create destination directory if it does not exist
os.makedirs(dest_dir, exist_ok=True)

# Get list of all .avi files in the source directory
avi_files = glob.glob(os.path.join(src_dir, "*.avi"))

# Loop over each file and convert to mp4
for avi_file in avi_files:
    # Extract base name and create output file path with .mp4 extension
    base_name = os.path.splitext(os.path.basename(avi_file))[0]
    mp4_file = os.path.join(dest_dir, base_name + ".mp4")

    # Build the ffmpeg command
    command = [
        "ffmpeg",
        "-i", avi_file,     # Input file
        "-c:v", "libx264",  # Video codec (adjust parameters as needed)
        "-crf", "23",       # Quality parameter (lower values mean higher quality)
        "-preset", "medium",# Preset (adjust speed vs. quality trade-off)
        "-c:a", "aac",      # Audio codec
        "-b:a", "192k",     # Audio bitrate
        mp4_file            # Output file
    ]

    # Execute the command
    try:
        print(f"Converting {avi_file} to {mp4_file}...")
        subprocess.run(command, check=True)
        print(f"Successfully converted {avi_file} to {mp4_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {avi_file}: {e}")

print("Conversion process completed.")

