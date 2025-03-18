import os
import ffmpeg

source_dir = "/scratch/cxm2114/thishome/Research/SALT/layered_neural_codec/data/Hollywood2/AVIClips"
dest_dir = "/scratch/cxm2114/thishome/Research/SALT/layered_neural_codec/data"

for filename in os.listdir(source_dir):
    if filename.endswith(".avi"):
        input_path = os.path.join(source_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".mp4"
        output_path = os.path.join(dest_dir, output_filename)

        try:
            ffmpeg.input(input_path).output(output_path, vcodec='libx264', acodec='aac', strict='experimental').run()
            print(f"Converted: {filename} to {output_filename}")
        except ffmpeg.Error as e:
            print(f"Failed to convert {filename}: {e.stderr.decode()}")

print("Conversion and moving complete.")
