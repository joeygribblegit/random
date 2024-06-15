from moviepy.editor import VideoFileClip
import librosa
import os

def find_audio_spikes(video_path):
    # Extract audio from video
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path)

    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=None)
    # Find spikes in audio
    energy = librosa.feature.rms(y=y)
    frames = librosa.util.peak_pick(energy[0], pre_max=1, post_max=1, pre_avg=3, post_avg=3, delta=0.1, wait=0)
    times = librosa.frames_to_time(frames, sr=sr)
    
    # Clean up
    clip.reader.close()  # Close the video file reader explicitly
    # clip.audio.reader.close()  # Close the audio file reader explicitly if necessary
    del clip  # Optionally delete the clip object to free up resources

    os.remove(audio_path)  # Remove the temporary audio file

    return times

def scan_directory_for_spikes(directory):
    results = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(subdir, file)
                spikes = find_audio_spikes(video_path)
                if spikes.size > 0:
                    print("found spike!")
                    results.append((video_path, spikes))
    return results

# Usage
directory = "/Users/jgribble/Desktop/video/20240416/00"
spikes_info = scan_directory_for_spikes(directory)
# import pdb; pdb.set_trace()
if len(spikes_info) > 0:
    for info in spikes_info:
        print(f"Spikes found in {info[0]} at times {info[1]}")
else:
    print("nothing found at all")
