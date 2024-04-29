import os
import sys
import base64
import ollama
import time
from pathlib import Path

def encode_image_to_base64(filepath):
    """Encode image to base64."""
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def query_llava_model(image_base64, prompt):
    """Query LLaVa model with base64 encoded image."""
    try:
        response = ollama.chat(model='llava:13b', messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            },
        ])
        return response['message']['content']
    except ollama.ResponseError as e:
        return f"Error: {e}"

def write_caption_to_file(image_path, caption):
    """Write the caption to a file with the same name as the image but with .llava.txt extension."""
    caption_path = image_path.with_suffix('.llava.txt')
    with open(caption_path, 'w') as file:
        file.write(caption)

def process_directory(directory_path, prompt):
    """Process all images in the directory and write captions."""
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    images = [p for p in Path(directory_path).iterdir() if p.suffix.lower() in image_extensions]
    total_images = len(images)
    start_time = time.time()

    for idx, image_path in enumerate(images):
        caption_file = image_path.with_suffix('.llava.txt')
        if caption_file.exists():
            print(f"Skipped {image_path.name}, existing caption file.")
            continue

        print(f"Processing {image_path.name} ({idx + 1}/{total_images})...")
        image_base64 = encode_image_to_base64(image_path)
        caption = query_llava_model(image_base64, prompt)
        
        if "Error" in caption:
            print(caption)
        else:
            write_caption_to_file(image_path, caption)
            print("Generated Caption:", caption)

        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / (idx + 1)
        estimated_total_time = avg_time_per_image * total_images
        estimated_time_left = estimated_total_time - elapsed_time
        print(f"Completed {100 * (idx + 1) / total_images:.2f}% in {elapsed_time:.2f}s, estimated {estimated_time_left:.2f}s left.")

def load_prompt_from_file(file_path):
    """Load the system prompt from a specified file."""
    with open(file_path, 'r') as file:
        return file.read()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory> <prompt file [optional]>")
        sys.exit(1)

    directory_path = sys.argv[1]
    # if sys.argv[2] is unset, use the default prompt file
    if len(sys.argv) == 2:
        prompt_file_path = "prompt.txt"
    else:
        prompt_file_path = sys.argv[2]

    if not os.path.isdir(directory_path):
        print("Provided path is not a directory")
        sys.exit(1)

    if not os.path.isfile(prompt_file_path):
        print("Prompt file does not exist")
        sys.exit(1)

    systemprompt = load_prompt_from_file(prompt_file_path)
    
    process_directory(directory_path, systemprompt)