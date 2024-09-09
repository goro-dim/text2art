<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    
</head>
<body>
    <h1>🧙‍♂️ text2art: A Guide to Summoning Art, using pixel sorcery with Stable Diffusion - AI-Generated Art from Text 🧙‍♂️ </h1>
    <p>Welcome to <strong>text2art</strong>! This guide will show you how to generate beautiful images from text descriptions using Stable Diffusion on Google Colab. No technical knowledge needed—just follow these simple steps!</p>

  <h2>🚀 Getting Started</h2>

   <h3>Prerequisites</h3>
    <ol>
        <li><strong>Google Account</strong>: You’ll need this to use Google Colab.</li>
        <li><strong>Internet Browser</strong>: Works best with Chrome or Firefox.</li>
    </ol>

  <h3>Step 1: Set Up Google Colab</h3>
    <ol>
        <li>Open <a href="https://colab.research.google.com/">Google Colab</a>.</li>
        <li>Click <strong>"New Notebook"</strong>.</li>
    </ol>

   <h3>Step 2: Enable GPU for Faster Processing</h3>
    <ol>
        <li>Go to the <strong>"Runtime"</strong> menu at the top.</li>
        <li>Select <strong>"Change runtime type"</strong>.</li>
        <li>Set <strong>"Hardware accelerator"</strong> to <strong>GPU</strong> (e.g., T4).</li>
        <li>Click <strong>"Save"</strong>.</li>
    </ol>
    
  <h3>Step 3: Install Required Libraries</h3>
    <p>Copy and paste the following code into a new cell and run it by pressing <strong>Shift + Enter</strong>:</p>
    <pre><code class="language-bash">
# Install Necessary Libraries
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers transformers scipy ftfy
    </code></pre>
    <h3>Step 4: Import and Load the Stable Diffusion Model</h3>
    <p>In a new cell, copy and paste the following code to load the model:</p>
    <pre><code class="language-python">
# Import Libraries
from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display
# Load the Stable Diffusion Model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
    </code></pre>
    
  <h3>Step 5: Generate Your First Image</h3>
  <p>Now comes the fun part! In a new cell, use the following code to generate an image from your text prompt:</p>
 
<pre><code class="language-python">
# Define your text prompt
prompt = "A beautiful landscape with mountains, a river, and a sunset in the background."

# Generate the image
image = pipe(prompt).images[0]

# Display the generated image
display(image)
</code></pre>

  <h3>(Optional)Step 6: Save Your Image</h3>
    <p>To save your image directly in Google Colab, use:</p>
    <pre><code class="language-python">
# Save the image locally
image.save("generated_image.png")
    </code></pre>
    <p>To download the image, click on the <strong>Files</strong> icon on the left sidebar, find <code>generated_image.png</code>, right-click, and select <strong>Download</strong>.</p>

   <h3>(Optional) Step 7: Save the Image to Google Drive</h3>
    <p>To keep your generated images in Google Drive:</p>
    <ol>
        <li>Run this code to mount your Drive:</li>
        <pre><code class="language-python">
from google.colab import drive
drive.mount('/content/drive')
        </code></pre>
        <li>Save the image to your Drive:</li>
        <pre><code class="language-python">
image.save("/content/drive/MyDrive/generated_image.png")
        </code></pre>
    </ol>

   <h2>🖌️ Generating More Images</h2>
    <p>To create more images, simply change the text prompt in Step 5 and run the code again. Enjoy creating stunning AI-generated art with <strong>text2art</strong>!</p>

  <h2>🔗 Resources</h2>
    <ul>
        <li><a href="https://colab.research.google.com/">Google Colab</a></li>
        <li><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">Stable Diffusion on Hugging Face</a></li>
    </ul>

  <p><strong>Happy Creating! 🎨✨</strong></p>
</body>
</html>
