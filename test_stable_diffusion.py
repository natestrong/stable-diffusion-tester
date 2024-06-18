import requests
import torch
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def run_test():
    device = get_device()
    print(f"Using device: {device}")

    # Initialize the pipeline with the default VAE
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    )

    # Move model to device
    pipe.to(device)

    # Disable the safety checker by returning the images unmodified and an empty list of nsfw content
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    # Fetch an initial image
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))

    # Define the prompt
    prompt = "A fantasy landscape, trending on artstation"

    # Generate the image
    if device == "mps":
        with torch.no_grad():
            result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, num_inference_steps=50)
    else:
        with torch.autocast(device):
            result = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, num_inference_steps=50)

    # Save the result
    output_image = result.images[0]
    output_image.save("/tmp/stable_diffusion_test_output.png")
    print("Image saved to /tmp/stable_diffusion_test_output.png")


if __name__ == "__main__":
    run_test()
