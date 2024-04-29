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
    response = ollama.chat(model='llava:13b', messages=[
        {
            'role': 'user',
            'content': prompt,
            'images': [image_base64]
        },
    ])
    return response['message']['content']

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
        print(f"Processing {image_path.name} ({idx + 1}/{total_images})...")
        image_base64 = encode_image_to_base64(image_path)
        caption = query_llava_model(image_base64, prompt)
        print(f"Caption: {caption}")
        write_caption_to_file(image_path, caption)
        elapsed_time = time.time() - start_time
        avg_time_per_image = elapsed_time / (idx + 1)
        estimated_total_time = avg_time_per_image * total_images
        estimated_time_left = estimated_total_time - elapsed_time
        print(f"Completed {100 * (idx + 1) / total_images:.2f}% in {elapsed_time:.2f}s, estimated {estimated_time_left:.2f}s left.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    directory_path = sys.argv[1]
    if not os.path.isdir(directory_path):
        print("Provided path is not a directory")
        sys.exit(1)
    
    systemprompt = """SYSTEM Your job is to caption images. You will be shown an image and you must write a caption for it. 
    This caption must describe:
    - The subject of the image
    - What is happening in the image
    - The mood or tone of the image
    - Anything relevant to the style of the image
    - Include the type of image it is (e.g. photograph, painting, etc.)
    - Include any pertinent objects.
    - Include a description of the background scene.
    - Use creative judgement to expand upon the above with anything found in the image.
    Consider these images as being catalogued for a museum, where it is important to note search terms that will help people find the image.
    Reduce filler words where possible to highlight the keywords. Comma limited descriptions are perfect. Respond 
    only with a caption, no prefix or post-text is needed. Be detailed.
    Caption this image.
    """
    
    process_directory(directory_path, systemprompt)