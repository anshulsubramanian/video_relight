# Video Relighting Pipeline

A real-time video relighting solution using physically-based rendering.

## Pipeline

```
Input Frame
  ├── RVM MobileNetV3 (15 MB)    → Alpha Matte
  └── Depth-Anything-V2-Small    → Depth Map
       └── Sobel Gradients        → Surface Normals (EMA smoothed)

Alpha + Normals + Frame → Cook-Torrance BRDF → Relit Output
```

## Constraints

| Property | Target | Actual |
|---|---|---|
| Speed | ≥ 10 FPS | ~20–40 FPS @ 720p on GPU |
| Model size | < 150 MB | ~115 MB total |
| Temporal consistency | Required | RVM recurrent states + normal EMA |

## Models

- **[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)** (MobileNetV3, ONNX) — 15 MB  
  Recurrent architecture propagates temporal context frame-by-frame for flicker-free alpha mattes.
- **[Depth-Anything-V2-Small](https://github.com/DepthAnything/Depth-Anything-V2)** — ~100 MB  
  Monocular depth estimation. Depth maps are analytically converted to surface normals via Sobel cross-product.

## Setup

```bash
pip install -r requirements.txt
python download_models.py
```

## Usage

### Relist a video with a point light source

```bash
python run.py \
  --input  input.mp4 \
  --output relit.mp4 \
  --light-dir  0.5 0.8 0.3 \
  --light-color 1.0 0.95 0.85 \
  --light-intensity 3.0 \
  --roughness 0.5 \
  --metallic  0.0
```

### All CLI options

```
--input            Path to input video file
--output           Path to output video file  [default: relit_<input>.mp4]
--background       Background image or video for compositing (optional)
--light-dir        Light direction as 3 floats (x y z, unnormalized)  [default: 0 1 1]
--light-color      RGB light color, each in [0,1]  [default: 1 1 1]
--light-intensity  Scalar light intensity  [default: 2.0]
--roughness        GGX roughness in [0,1]  [default: 0.5]
--metallic         Metalness in [0,1]  [default: 0.0]
--ema-alpha        EMA smoothing for normals, 0=no memory 1=full  [default: 0.85]
--downsample-ratio RVM downsample ratio (lower = faster, less detail)  [default: 0.25]
--device           cuda or cpu  [default: cuda if available]
--checkpoint-dir   Directory with model checkpoints  [default: ./checkpoints]
```

## Temporal Consistency

Three mechanisms prevent flickering:

1. **Alpha (matting)**: RVM's recurrent GRU hidden states `r1–r4` are passed from one frame to the next, so the model has memory of previous frames built-in.
2. **Normals**: A per-pixel EMA smoother blends each new normal map with the previous one (`α=0.85` by default) and then re-normalizes, damping high-frequency temporal noise.
3. **Relighting**: The BRDF is deterministic given stable normals + alpha, so temporal stability flows naturally from the above two.

## Cook-Torrance BRDF

The specular component uses the GGX microfacet model:

```
D  = GGX Normal Distribution Function
G  = Smith-Schlick Geometry term
F  = Fresnel-Schlick  (F0 = lerp(0.04, albedo, metallic))
Lo = (D·G·F / (4·NdotV·NdotL)) · NdotL · light
   + (1 - F) · albedo/π · NdotL · light   (Lambertian diffuse)
```

Final output: `alpha * Lo + (1 - alpha) * background`
