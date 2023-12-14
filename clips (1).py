from moviepy.video.io.VideoFileClip import VideoFileClip

def cut_video(input_path, output_path, start_time, end_time):
    # Load the video clip
    video_clip = VideoFileClip(input_path)

    # Define the subclip based on the start and end times
    subclip = video_clip.subclip(start_time, end_time)

    # Write the subclip to the output file
    subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    # Replace 'input_video.mp4' with the path to your input video file
    input_video_path = "D:\Ksquarez\Yolonas and Deep Sort\inference\\test_1.mp4"
    
    # Replace 'output_video_cut.mp4' with the desired output video file path
    output_video_path = "D:\Ksquarez\Yolonas and Deep Sort\inference\clip.mp4"

    # Set the start and end times for the cut (in seconds)
    start_time = 0  # 8:00 min
    end_time = 10    # 8:20 min

    # Cut the video
    cut_video(input_video_path, output_video_path, start_time, end_time)
