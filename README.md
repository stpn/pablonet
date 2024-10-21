# Pablonet

<div class="image-container" style="flex: 1; text-align: left">
    <img src="frame.png"  width="200">
</div>

Blogpost: https://mlecauchois.github.io/posts/pablonet/

## Setup

```
python3.10 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -m streamdiffusion.tools.install-tensorrt
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

```
