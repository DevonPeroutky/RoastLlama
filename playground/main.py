import torch
import requests

from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load LLM model and processor (LLama3)
llm_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Freeze both the LLM and CLIP
for param in llm_model.parameters():
    param.requires_grad = False
for param in clip_model.parameters():
    param.requires_grad = False


# Define the image
# TODO: Replace with dataloader of image + caption dataset
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Generate image embeddings
inputs = clip_processor(images=image, return_tensors="pt", padding=True)
image_embeddings = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
print(image_embeddings.shape) # torch.Size([1, 768])

# Projection Matrix
projection_w, projection_h = 512, 768
projection_matrix = torch.randn(projection_w, projection_h, requires_grad=True)

# Optimizer for the projection matrix
optimizer = optim.Adam([projection_matrix], lr=0.001)

# Inference
caption_messages = [
    {"role": "system", "content": "You are a hilarious comedian"},
    {"role": "user", "content": "Roast this person"},
]

input_ids = llm_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Batch (Image - Caption pairs)
dataloader = []

def train(dataset, epochs):
    for epoch in range(epochs):
        print("===================\nEpoch: ", epoch)
        for batch in dataloader:
            # Unpack batch
            images = batch['images']
            captions = batch['captions']
            images: List[PIL.Image] = [Image.open(image_path) for image_path in images]

            # Generate image embeddings
            inputs = clip_processor(images=images, return_tensors='pt', padding=True)
            with torch.no_grad():  # Ensure no gradients for CLIP
                image_embeddings = clip_model.get_image_features(**inputs)

            projected_embeddings = torch.matmul(image_embeddings, projection_matrix)

            # Tokenize Labels
            messages += { 'role': 'assistant', 'content': captions}
            label_ids = llm_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(llm_model.device).input_ids

            # Forward Pass
            outputs = llm_model(input_embeds=projected_embeddings, labels=label_ids)
            print("Loss: ", outputs.loss)

            # Zero gradients
            optimizer.zero_grad()

            # Backward pass to calculate gradients
            loss.backward()

            # Have Adam update the weights with the gradients and learning rate
            optimizer.step()

