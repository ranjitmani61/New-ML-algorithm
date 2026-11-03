"""Build a tiny random Qwen2-architecture model plus a Qwen-style tokenizer for local smoke tests.

    python3 make_tiny_model.py OUT_DIR

Not a real model: it only lets run_llm.py execute its exact code path on CPU (chat template,
answer-only loss masking, padding, generation, parsing, saving). Digits are split one per token
and the chat format and end-of-turn token follow Qwen2.5-Instruct.
"""

import sys

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

import problems as P

out = sys.argv[1]
words = set()
for w in (P.NAMES_TRAIN + P.NAMES_HELDOUT + P.ITEMS_TRAIN + P.ITEMS_HELDOUT + P.OTHER_THINGS
          + [P.singular(x) for x in P.ITEMS_TRAIN + P.ITEMS_HELDOUT + P.OTHER_THINGS]):
    words.add(w)
for t in P.GAIN + P.LOSS + P.INACTIVE + P.IRRELEVANT + [P.DOUBLE, P.HALVE, "{name} has {n} {item}.",
                                                          "How many {item} does {name} have now?", P.SYSTEM if hasattr(P, "SYSTEM") else ""]:
    for w in t.replace("{", " ").replace("}", " ").replace("'", " ' ").replace(".", " . ").replace("?", " ? ").split():
        words.add(w)
words |= set("Solve the word problem . Answer with the final number only system user assistant".split())
special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]"]
vocab = {t: i for i, t in enumerate(special + [str(d) for d in range(10)] + sorted(words) + ["\n", "-", "'", "s"])}
tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                              unk_token="[UNK]", additional_special_tokens=["<|im_start|>"])
tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                     "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  eos_token_id=vocab["<|im_end|>"], pad_token_id=vocab["<|endoftext|>"], tie_word_embeddings=True)
Qwen2ForCausalLM(cfg).save_pretrained(out)
tok.save_pretrained(out)
print("tiny model and tokenizer saved to", out, "vocab", len(vocab))
