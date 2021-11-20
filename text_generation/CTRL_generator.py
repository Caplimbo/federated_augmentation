import numpy as np
import torch
from collections import defaultdict

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
)

# MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
#
# DEVICE = "cuda"
#
# print("Loading CTRL Model...")
# tokenizer = CTRLTokenizer.from_pretrained("ctrl", cache_dir="models")
# model = CTRLLMHeadModel.from_pretrained("ctrl", cache_dir="models")
# if DEVICE == "cuda":
#     model.half()
# model.to(DEVICE)
# print("CTRL Model Loaded!")


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def generate_by_ratings(ratings=None, length=30, num=1):
    if ratings is None:
        ratings = [1, 2, 3, 4, 5]
    # print(f"Augmenting, rating={ratings}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = adjust_length_to_model(length=length, max_sequence_length=model.config.max_position_embeddings)

    prompt_texts = [f"Reviews Rating: {score}.0" for score in ratings]

    encoded_prompt = torch.tensor([tokenizer.encode(
        prompt_text, add_special_tokens=False
    ) for prompt_text in prompt_texts])
    encoded_prompt = encoded_prompt.to(DEVICE)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt
    # print("input ids: ", input_ids)
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=length + len(encoded_prompt[0]),
        temperature=0.2,
        top_k=2,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=num,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        # print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        raw_text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) + 5:]
        generated_sequences.append(raw_text)

    return generated_sequences

SAMPLE_DICT = np.load("text_generation/generated_samples.npy", allow_pickle=True).item()

def sample_by_ratings(ratings):
    # print("Sampling!")
    counts = [ratings.count(i) for i in range(1, 6)]
    sampled = []
    for i, count in enumerate(counts):
        full = SAMPLE_DICT[i+1]
        sample = np.random.choice(full, count, replace=False)
        sampled.extend(sample)
    return sampled

# def generate_and_save(path):
#     dct = {}
#     target_length = [25] * 100 + [30] * 2000 + [35] * 900
#     for length in target_length:
#         for label in range(1, 6):
#             print(f"generating length {length}, label {label}")
#             generated = generate_by_ratings([label]*32, length=length, num=1)
#             # print(generated)
#             if label not in dct:
#                 dct[label] = generated
#             else:
#                 dct[label].extend(generated)
#     np.save(path, dct)


if __name__ == "__main__":
    # generate_and_save("generated_samples.npy")
    print(sample_by_ratings([1, 2, 3, 4, 5]))
