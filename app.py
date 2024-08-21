import streamlit as st
import logging
import time
import csv
import io
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from cryptography.fernet import Fernet
from typing import Optional, Union

# Streamlit app title
st.title("Encryption & Decryption with AI Q&A Generator")

# Function to generate model output
def generate(
    prompt: str,
    model: Union[str, AutoModelForCausalLM],
    hf_access_token: str = None,
    tokenizer: Union[str, AutoTokenizer] = 'meta-llama/Llama-2-7b-hf',
    device: Optional[str] = None,
    max_length: int = 1024,
    assistant_model: Optional[Union[str, AutoModelForCausalLM]] = None,
    generate_kwargs: Optional[dict] = None,
) -> str:
    """Generates output given a prompt."""
    if not device:
        if torch.cuda.is_available() and torch.cuda.device_count():
            device = "cuda:0"
            logging.warning('Inference device is not set, using cuda:0, %s', torch.cuda.get_device_name(0))
        else:
            device = 'cpu'
            logging.warning('No CUDA device detected, using cpu, expect slower speeds.')

    if 'cuda' in device and not torch.cuda.is_available():
        raise ValueError('CUDA device requested but no CUDA device detected.')

    if not hf_access_token:
        raise ValueError((
            'Hugging face access token needs to be specified. '
            'Please refer to https://huggingface.co/docs/hub/security-tokens'
            ' to obtain one.'
            )
        )

    if isinstance(model, str):
        checkpoint_path = model
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )
    model.to(device).eval()
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            token=hf_access_token,
        )

    draft_model = None
    if assistant_model:
        draft_model = assistant_model
        if isinstance(assistant_model, str):
            draft_model = AutoModelForCausalLM.from_pretrained(
                assistant_model,
                trust_remote_code=True
            )
        draft_model.to(device).eval()

    tokenized_prompt = tokenizer(prompt)
    tokenized_prompt = torch.tensor(
        tokenized_prompt['input_ids'],
        device=device
    )
    tokenized_prompt = tokenized_prompt.unsqueeze(0)

    stime = time.time()
    output_ids = model.generate(
        tokenized_prompt,
        max_length=max_length,
        pad_token_id=0,
        assistant_model=draft_model,
        **(generate_kwargs if generate_kwargs else {}),
    )
    generation_time = time.time() - stime

    output_text = tokenizer.decode(
        output_ids[0].tolist(),
        skip_special_tokens=True
    )

    return output_text, generation_time

def extract_qa(text):
    """Extract question and answer pairs from generated text."""
    qas = []
    parts = text.split("\n")

    for i in range(0, len(parts)-1, 2):
        question = parts[i].strip()
        answer = parts[i+1].strip()

        if question and answer:
            qas.append((question, answer))

    return qas

# Encryption section
st.header("Enter text to be encrypted")

user_input = st.text_area("Enter the text you want to encrypt:")
encrypt_button = st.button("Encrypt Text")

if encrypt_button and user_input:
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    encrypted_text = cipher_suite.encrypt(user_input.encode('utf-8'))

    st.write(f"Encrypted Text: {encrypted_text.decode('utf-8')}")
    st.download_button("Download Key", key, file_name="encryption_key.key")
    st.download_button("Download Encrypted Text", encrypted_text, file_name="encrypted_text.txt")

# Decryption section
st.header("Enter encrypted text and key to decrypt")
encrypted_input = st.text_area("Enter the encrypted text:")
key_input = st.text_area("Enter the encryption key:")

decrypt_button = st.button("Decrypt Text")

if decrypt_button and encrypted_input and key_input:
    try:
        cipher = Fernet(key_input.encode('utf-8'))
        decrypted_text = cipher.decrypt(encrypted_input.encode('utf-8')).decode('utf-8')
        st.write(f"Decrypted Text: {decrypted_text}")
    except Exception as e:
        st.error("An error occurred during decryption. Please check your key and encrypted text.")

# Q&A generation section
st.header("Generate AI Q&A")
prompts = st.text_area("Enter prompts for AI Q&A (one per line):").splitlines()
generate_button = st.button("Generate Q&A")

if generate_button and prompts:
    args = {
        'model': 'apple/OpenELM-270M',
        'hf_access_token': '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',
        'device': None,
        'max_length': 350,
        'assistant_model': None,
        'generate_kwargs': {'repetition_penalty': 2.0},
    }

    # Prepare in-memory CSV
    output_stream = io.StringIO()
    csvwriter = csv.writer(output_stream)
    csvwriter.writerow(['Generated Question', 'Generated Answer'])

    for prompt in prompts:
        output_text, generation_time = generate(
            prompt=prompt,
            model=args['model'],
            device=args['device'],
            max_length=args['max_length'],
            assistant_model=args['assistant_model'],
            generate_kwargs=args['generate_kwargs'],
            hf_access_token=args['hf_access_token'],
        )

        qas = extract_qa(output_text)
        for question, answer in qas:
            csvwriter.writerow([question, answer])

        st.write(f"Generated Q&A for prompt: {prompt}")
        st.write(f"Generation time: {round(generation_time, 2)} seconds")
        st.write("---")

    # Encrypt the CSV data
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    csv_data = output_stream.getvalue().encode('utf-8')
    encrypted_data = cipher_suite.encrypt(csv_data)

    # Display encrypted CSV data
    st.write(f"Encrypted CSV Data: {encrypted_data.decode('utf-8')}")
    st.download_button("Download Encrypted CSV Data", encrypted_data, file_name="encrypted_ai_qa_dataset.csv")
    st.download_button("Download Key", key, file_name="encryption_key.key")