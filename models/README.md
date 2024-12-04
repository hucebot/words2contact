# Using Local LLMs / VLMs

Since we use llama.cpp for the local LLMs, you can download any gguf model from hugging face for example and use it with the following command, assuming you are in the root directory of the repository:

```bash
cd models
git curl https://huggingface.co/MaziyarPanahi/Calme-7B-Instruct-v0.9-GGUF/resolve/main/Calme-7B-Instruct-v0.9.Q6_K.gguf?download=true -o Calme-7B-Instruct-v0.9.Q6_K.gguf
cd ..
```

Then you can adapt the words2contact.py file so that you can include your models.