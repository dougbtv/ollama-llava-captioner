import base64
import ollama

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

# Path to your image
image_path = '/home/doug/ai-ml/kohya_ss/dataset/guideboat/images/15_guideboat/9745ACAA-17CF-4F65-BFC1-511565219730.jpg'
image_base64 = encode_image_to_base64(image_path)

systemprompt = """SYSTEM Your job is to caption images. You will be shown an image and you must write a caption for it. 
This caption must describe:

- The subject of the image
- What is happening in the image
- The mood or tone of the image
- Anything relevant to the style of the image
- Include the type of image it is (e.g. photograph, painting, etc.)
- Include any pertitent objects.
- Include a description of the background scene.
- Use creative judgement to expand upon the above with anything found in the image.

Consider these images as being catalogued for a museum, where it is important to note search terms that will help people find the image.
Reduce filler words where possible to highlight the keywords. Comma limited descriptions are perfect. Respond 
only with a caption, no prefix or post-text is needed. Be detailed.


Caption this image.
"""

# Respond in the format of: {subject}, {action}, {mood}, {style}, {objects}, {background}, {image type}


# Query the model
caption = query_llava_model(image_base64, systemprompt)
print("Generated Caption:", caption)