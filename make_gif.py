from moviepy.editor import VideoFileClip

INPUT_VIDEO = "data/output_od_matrix.mp4"
OUTPUT_GIF = "demo.gif"

print("⏳ در حال ساخت GIF... لطفا صبر کن...")

clip = VideoFileClip(INPUT_VIDEO).subclip(25, 35).resize(width=800)
clip.write_gif(OUTPUT_GIF, fps=10)

print("✅ فایل demo.gif با موفقیت ساخته شد!")