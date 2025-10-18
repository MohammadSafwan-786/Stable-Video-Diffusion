from flask import Flask, request, send_file
from diffusers import StableVideoDiffusionPipeline
import torch
import tempfile
import imageio
import os

app = Flask(__name__)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")

    banned = ["violence", "nude", "weapon", "sex", "drugs", "kill", "blood", "terror"]
    if any(word in prompt.lower() for word in banned):
        return "Unsafe prompt", 400

    video_frames = pipe(prompt, num_inference_steps=25).frames
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "output.mp4")
    imageio.mimsave(video_path, video_frames, fps=8)

    return send_file(video_path, mimetype="video/mp4")
