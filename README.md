# LLaVa stable diffusion training image captioner

This script takes a directory of images and 

Setup:

```
pip install ollama
```

Running it...

```
python llava-caption-ollama.py /path/to/images
```

or, specify your own prompt file:

```
python llava-caption-ollama.py /path/to/images /path/to/myprompt.txt
```

If the prompt argument is empty, uses the `prompt.txt` in this directory.
