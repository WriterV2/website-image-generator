import argparse
import glob
import re

from PIL import Image, ImageFilter, ImageOps
from diffusers import StableDiffusionPipeline

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a white pencil drawing on a black background from a text prompt using stable diffusion")
    parser.add_argument("prompt", help="text prompt to generate the image from")
    parser.add_argument("--output", help="path to the output file")
    args = parser.parse_args()

    # create stable diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)

    # generate image from prompt
    image = pipe(args.prompt).images[0]

    # convert to black and white pencil drawing
    image = image.convert("L")

    # apply pencil drawing filter
    image = image.filter(ImageFilter.CONTOUR)

    # invert colors to get white foreground on black background
    image = ImageOps.invert(image)

    # determine output file path
    if args.output:
        output_path = args.output
    else:
        # strip non-alphanumeric characters from the prompt
        prompt_str = re.sub("[^0-9a-zA-Z]", "", args.prompt)

        # make the output file name all lowercase
        prompt_str = prompt_str.lower()

        output_files = glob.glob(f"{prompt_str}-*.png")
        if output_files:
            output_files.sort()
            last_output_file = output_files[-1]
            last_output_num = int(last_output_file.split("-")[1].split(".")[0])
            output_path = f"{prompt_str}-{last_output_num + 1}.png"
        else:
            output_path = f"{prompt_str}-1.png"

    # save processed image to file
    image.save(output_path)

if __name__ == "__main__":
    main()
