# Day 15: Image generation mechanics — a toy denoising loop

Book chapter: **2.3 Image generation mechanics**

Task: toy denoising loop showing latency grows with step count.

## What we built

`toy_diffusion_steps.py` has two parts:

1. **How denoising works, on a tiny picture.** A ring drawn in text characters
   is buried in random noise, then recovered step by step. A stand-in for a
   trained model guesses the clean picture at each step. It shows the picture
   emerging, and that more steps give a better picture.
2. **What each step costs, with a real-sized network.** A 6-layer transformer
   denoiser (21M parameters, random weights) runs the same loop on a 32x32
   latent. It times 1 to 50 steps, then bigger images at 20 steps.

It saves `toy_diffusion_steps.png`: latency vs steps, and quality vs steps.

Run it:

```
python3 toy_diffusion_steps.py
```

It takes about 10 seconds, mostly for the 128x128 case. Part 2 uses the MPS GPU
with fp16 if available (CUDA if present), and CPU with fp32 otherwise.

## What we learned

### An image model is a pipeline of three models

Unlike an LLM, an image generator chains several models:

```
prompt -> [text encoder] -> [denoising model, 30-50 steps] -> [VAE decoder] -> image
```

| Part | Job |
|---|---|
| **Text encoder** | Turns the prompt into numbers the image model understands |
| **Denoising model** | Starts from noise and refines it into an image, step by step. The expensive part, and the only part we built |
| **VAE** (variational autoencoder) | Turns the finished latent into real pixels |

Extras are often plugged in: **LoRAs** (small fine-tunes that change style) and
**ControlNets** (outlines or edges that steer the shape). Tools like ComfyUI
wire these pieces together.

### Latent space: work on a small version of the image

A 1024 x 1024 image is over a million pixels, and attention compares every
piece with every other piece (day 8), so denoising pixels directly would be far
too expensive. Everything happens in **latent space** instead: a compressed
version of the image, about 128 x 128, roughly 1% of the values. The VAE turns
it back into pixels only at the end. Our Part 2 latents (16 x 16 up to
128 x 128) stand for exactly this.

### Diffusion works backwards from noise

An image model does not write an image left to right like an LLM writes
tokens. It starts from **pure random noise** for the whole image and
**removes a little noise at each step**, until a clean image is left.

```
pure noise --step--> slightly less noisy --step--> ... --step--> clean image
```

Every step:

1. Show the denoiser the current noisy image and how noisy it is (`t`).
2. The denoiser guesses which way to move towards a clean image.
3. Move a small amount that way.

We used the simplest way to describe "how noisy": a straight line between the
picture and static,

```
x = (1 - t) * clean + t * noise      t = 1: pure noise,  t = 0: the clean picture
```

and each step moves along that line towards the denoiser's guess.

### Why more steps give a better image

A real denoiser is **unsure when the image is very noisy**: early on it can
only guess something blurry, and it gets sharper as the noise clears. Our
stand-in copies that: at high noise it guesses the average colour, at low noise
the real picture (we give it the real picture, since we never trained one).

- **1 step:** the denoiser only sees pure noise, guesses "average grey", and we
  jump straight there. The result is a flat grey image.
- **Many small steps:** each step only moves a little, and later steps get to
  use the denoiser's sharper guesses. The picture comes out clearly.

Measured error against the clean picture:

| Steps | 1 | 2 | 4 | 8 | 16 | 32 | 50 |
|---|---|---|---|---|---|---|---|
| Error | 0.859 | 0.430 | 0.215 | 0.107 | 0.054 | 0.027 | 0.017 |

Doubling the steps halves the error here. A **perfect** denoiser would need
only 1 step (the code checks this). That is the idea behind **few-step
models** (below).

### Why latency grows with steps

**Each step is one full forward pass of the network over the whole image.**
50 steps means 50 forward passes, so time grows in a straight line:

| Steps | Total | Per step |
|---|---|---|
| 1 | 0.007 s | 6.6 ms |
| 8 | 0.051 s | 6.4 ms |
| 50 | 0.316 s | 6.3 ms |

50 steps took about 45-48x as long as 1 step. The cost per step stays the
same, so **total time is about steps x time per step**.

**Quality and speed pull in opposite directions:** fewer steps is faster but
blurrier, more steps is sharper but slower. That is the main knob for image
generation latency. Most models use 30 to 50 steps.

### Guidance: two forward passes per step

Real image models run the denoiser **twice per step**: once **with** the prompt
and once **without** it. The two answers are combined using the **guidance
scale**, pushing the image away from "any image" and towards "the prompt". A
higher scale follows the prompt more strictly; a lower one is more creative.

So **a 50-step image is 100 forward passes**. Our code runs one pass per step
(it has no text encoder), so a real pipeline with guidance would take about
twice our times.

### The settings you control per request

| Setting | What it does |
|---|---|
| Prompt | What the image should show |
| Negative prompt | What it should not show (styles, objects) |
| Number of steps | Speed vs quality, usually 30 to 50 |
| Guidance scale | Prompt-following vs creativity, usually around 4 |
| Image size | Chosen from a fixed menu of sizes and aspect ratios |

These are the image version of day 11's temperature and top-p.

### The image is cut into patch tokens

The network is a **diffusion transformer**, like day 10's, but for images:

- The image (a 32x32 grid with 4 numbers per position, a "latent") is cut
  into 2x2 patches. Each patch becomes one token: 16 x 16 = **256 tokens**.
- **No causal mask:** the whole image exists at once, so every patch can look
  at every other patch. (Day 9's attention always used a causal mask, so this
  block uses SDPA from day 14 with `is_causal=False`.)
- The network is also told the noise level `t`, so it knows how unsure to be.

### Why image generation is compute-bound

Same network, 20 steps, bigger images:

| Latent | Patch tokens | Per step | TFLOPS |
|---|---|---|---|
| 16 x 16 | 64 | 2.7 ms | 0.9 |
| 32 x 32 | 256 | 6.4 ms | 1.6 |
| 64 x 64 | 1,024 | 27 ms | 1.9 |
| 128 x 128 | 4,096 | 212 ms | 1.7 |

- **Bigger images cost more than proportionally.** 4x the tokens took about 8x
  the time from 64x64 to 128x128, because attention grows with tokens x tokens
  (day 8).
- **Every step uses a large share of the GPU's compute** (1.6-1.9 TFLOPS out of
  day 12's measured peak of about 2.8), while an LLM decode step used about 3%.
  Hundreds or thousands of tokens share each fetch of the weights, so each step
  behaves like prefill, not decode (day 12's arithmetic intensity).

**From our numbers to a real image:** a 128 x 128 latent (an HD image's size)
took 212 ms per step. At 50 steps with guidance, that is 100 passes, about
21 seconds on this Mac, and our denoiser is about 1,000x smaller than Qwen
Image's 20B one.

### How image models have grown

| Part | SDXL (2023) | Qwen Image (2025) |
|---|---|---|
| Text encoder | Small CLIP model | Qwen 2.5 VL, a full 7B LLM |
| Denoiser | Under 4B parameters | 20B parameters |
| VAE | Single encoding | Dual encoding |

SDXL also had two denoisers: a **base** (noise to a rough image) and a
**refiner** (adds detail). Newer models gained readable text, realistic faces
and better prompt-following mostly by getting bigger, especially by using a
real LLM as the text encoder. A newer direction makes image models
LLM-style (autoregressive), such as HunyuanImage-3.0.

### Few-step models

Steps are the slow part, so some models are **trained to need 8 or fewer
steps**: 80 to 90% faster, but noticeably lower quality. Two ways to make one:

- **Latent consistency:** predict the final image directly, then repeat 2 to 4
  times to polish it.
- **Distillation** (more common): train a model to imitate a bigger one in fewer
  steps (chapter 1's distillation idea).

Good for latency-sensitive uses where quality matters less, like real-time
filters.

### Video generation

- The same architecture, but bigger: 3 to 5x more parameters, 10 to 100x more
  information in latent space.
- Frame by frame does not work: small errors build up with every frame.
- Modern models hold the **whole video** in latent space and denoise it all at
  once, with every frame attending to every other frame. The latent has three
  dimensions: X, Y and T (time).
- Still about 50 steps, but each step is huge: several seconds of compute per
  second of video, usually a whole 8-GPU node for one request.

## Diffusion vs an LLM

| | LLM | Diffusion image model |
|---|---|---|
| Generates | One token at a time | The whole image, refined over and over |
| Each forward pass processes | 1 new token (decode) | Every patch of the image |
| Number of passes | One per output token | Two per step with guidance (fixed, chosen up front) |
| Bottleneck | Memory (decode) | Compute (like prefill) |
| Main speed knob | Tokens per second | Number of steps, image size |
| KV cache | Yes | No: every step sees a new noisy image |

## Connection to earlier days

- Day 10: the same transformer pieces (embedding, blocks, output layer),
  with patches instead of word tokens. Random weights cost the same as trained
  ones, so they are fine for timing.
- Day 8 and 14: attention with no causal mask, via SDPA.
- Day 12: many tokens per weight fetch means compute-bound; the book said image
  and video generation are compute-bound, and this measures it.
- Day 1 and 6: latency again, now set by steps rather than tokens.
- Day 11: guidance scale and steps are per-request knobs, like temperature.
- Chapter 1: distillation, here used to make few-step models.

## Caveats

- **The stand-in denoiser is not a model.** We give it the real picture and make
  it blurry on purpose, so Part 1 shows the mechanics and the quality trend, not
  real image quality.
- **Part 2's network has random weights**, so its output is not an image. It is
  there for the cost of each step, which does not depend on the weight values.
- We only built the denoiser: no text encoder, no VAE, and **no guidance**, so
  each of our steps is one pass where a real model does two.
- Real image models are much larger and use smarter step schedules. The
  straight-line noise path is one common choice, not the only one.
- The TFLOPS figure only counts the big matrix multiplies, so it is approximate.
- Timings vary between runs.

## Things to try

1. Change the stand-in to `lambda x, t: clean` (a perfect denoiser) and see that
   1 step is enough.
2. Change `N_LAYERS` to 12 and check that time per step roughly doubles.
3. Change `PATCH` to 4: 4x fewer tokens per step. How much faster is it?
4. Work out from the steps table how long 1,000 steps would take, then run it.
