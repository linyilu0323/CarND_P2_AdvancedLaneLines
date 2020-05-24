from LaneFinder import process_image
from moviepy.editor import VideoFileClip


# select a video to work with
clip = VideoFileClip("test_videos/project_video.mp4")
#clip = VideoFileClip("test_videos/challenge_video.mp4")
#clip = VideoFileClip("test_videos/harder_challenge_video.mp4")
process_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!

output = 'output_videos/project_video.mp4'
#output = 'output_videos/challenge_video_diag.mp4'
#output = 'output_videos/harder_challenge_video.mp4'
process_clip.write_videofile(output, audio=False)
