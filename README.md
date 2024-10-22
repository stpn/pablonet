# Pablonet

<div class="image-container" style="flex: 1; text-align: left">
    <img src="frame.png"  width="200">
</div>

Blogpost: https://mlecauchois.github.io/posts/pablonet/

## Setup

```
python3.10 -m venv .env
source .env/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m streamdiffusion.tools.install-tensorrt
pip install polygraphy==0.47.1 --extra-index-url https://pypi.ngc.nvidia.com
pip install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com
```

## Run

Server CLI:
```
python server.py --base_model_path "Lykon/DreamShaper" \
--acceleration "tensorrt" \
--prompt "" \
--num_inference_steps 30 \
--guidance_scale 1.0 \
--t_index_list "[14,18]" \
--preprocessing canny_blur_shift \
--compression 80 \
--port 6000
```

Raspberry Pi client CLI:
```
python client_pi.py --prompt "painting in the style of pablo picasso, cubism, sharp high quality painting, oil painting, mute colors red yellow orange, background of green, color explosion, abstract surrealism" \
--image_size 150 \
--url ws://YOUR_URL_HERE \
--fullscreen \
--crop_size 900 \
--crop_offset_y 40 \
--compression 60
```
