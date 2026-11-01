"""Builds notebooks/finetune_unsloth_colab.ipynb (run: py notebooks/build_notebook.py).

The notebook is generated so its JSON is always valid; edit the cells here, not the .ipynb by hand.
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# --------------------------------------------------------------------------------------------------------
md(r"""
# Fine-tune an insurance-claims extractor with Unsloth (Google Colab)

Teaches a small open model to do what the pipeline in this repository does: **read one insurance claim and answer with the JSON record** (header, policy, insured objects, incident).

**What you need** (both come from the repo's `data/finetune/` folder - upload only these two files, *not* `data/review/`):

| File | What it is |
|---|---|
| `train.jsonl` | training examples, one chat per line: `system` instruction + `user` claim text + `assistant` JSON answer |
| `validation.jsonl` | held-out examples used to measure the model before and after training |

Both files are **de-identified** (names, addresses, VINs and ids are consistent fakes).

**Runtime:** `Runtime > Change runtime type > T4 GPU` (free tier is enough).

## The process, step by step
0. Check the GPU
1. Install Unsloth
2. Settings (model, sequence length, epochs ... in one place)
3. Load `train.jsonl` / `validation.jsonl`
4. Load the base model in 4-bit (QLoRA)
5. Attach LoRA adapters (the only weights that get trained)
6. Apply the chat template and check every example fits the context window
7. **Baseline**: score the *untrained* model on the validation set
8. Build the trainer (loss only on the assistant's JSON answer)
9. Train
10. **Evaluate again** and compare with the baseline
11. Save the LoRA adapter and **download it**
12. (optional) Drive / Hugging Face / merged model / GGUF, and how to reload the adapter

> **Be realistic about the data.** The repository export has about **21 training and 5 validation examples**. That is enough to check that the whole recipe works and to teach the output *format*, but far too little to build a reliable model - and 5 validation examples make every score noisy (one example = 20 points). Treat the results as a smoke test, then add many more claims.
""")

md(r"""
## Which model? (chosen from this project's numbers)

| What the project needs | Why it matters for the model choice |
|---|---|
| Output is a **fixed JSON schema** with up to 26 fields, enums and id formats | Needs a model that follows instructions well - use an **Instruct** model, not a base model |
| Each example is **~1,100 tokens on average, up to ~1,900** (instruction ~2,300 characters + claim up to ~2,700 + JSON answer up to ~800) | Needs at least a 3,072-token window to avoid cutting the answer off; small models handle this fine |
| Only **~21 examples** | We teach *format and vocabulary*, not new knowledge -> an already instruction-tuned model + a small LoRA; a bigger model starts from a better understanding of dates and claim wording |
| **Free Colab T4** (16 GB) | Use **4-bit QLoRA**; keep the model at 8B or below |

| Choice | Model id (Unsloth) | When to use it |
|---|---|---|
| **Recommended start** | `unsloth/Llama-3.2-3B-Instruct` | Same model as Unsloth's own Llama 3.2 notebook; fast (minutes), fits the T4 easily, follows JSON formats well after a little fine-tuning |
| Quick smoke test | `unsloth/Llama-3.2-1B-Instruct` | Fastest; use it to check the notebook runs end to end - expect weaker accuracy |
| Stronger, slower | `unsloth/Llama-3.1-8B-Instruct-bnb-4bit` | If the 3B score on dates / incident type is not good enough. Fits a T4 in 4-bit with batch size 1, but is several times slower - watch the memory print |

All three use the same chat template (`llama-3.1`), so switching is one setting in step 2. Other families (Qwen, Gemma, Phi ...) work with Unsloth too but need a different `CHAT_TEMPLATE` - copy the matching Unsloth notebook if you want one. **Check the model's license** (Llama Community License) before using the result commercially.
""")

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 0 - Check the GPU
Fine-tuning needs a CUDA GPU. If this cell fails, use `Runtime > Change runtime type > T4 GPU` and run it again.
""")
code(r'''
import torch

assert torch.cuda.is_available(), "No GPU found: Runtime > Change runtime type > T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))
!nvidia-smi
''')

md(r"""
## Step 1 - Install Unsloth
This cell is copied from Unsloth's official Llama 3.2 Colab notebook. On Colab it installs Unsloth and the exact versions of the supporting libraries it was tested with (`transformers`, `trl`, `datasets`, `torchao`). It takes a few minutes and prints nothing (`%%capture`).

> Unsloth updates often. If this cell ever fails, open the latest "Llama 3.2 (3B) conversational" notebook from <https://unsloth.ai/docs/get-started/unsloth-notebooks> and copy **its first cell** here.
""")
code(r'''
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth  # local machine
else:
    # Colab: install the pinned set Unsloth tests against
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
    !unsloth install-kernels
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.57.6
!pip install --no-deps trl==0.22.2
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 2 - Settings
Everything you might want to change is here. Defaults are chosen for ~21 short examples on a T4.

* `MAX_SEQ_LENGTH` - longest example (in tokens) the model sees. Step 6 checks that no example is longer, because a too-short window would silently cut off the JSON answer.
* `EPOCHS` - passes over the data. With so few examples, 3-5 is plenty; more just memorises.
* `LORA_R` - size of the LoRA adapters (8-64). Bigger = more capacity, more memory, easier to overfit.
* `SMOKE_TEST` - set to `True` to train for only 5 steps and prove the notebook runs.
""")
code(r'''
MODEL_CHOICES = {
    "3B (recommended)":      "unsloth/Llama-3.2-3B-Instruct",
    "1B (smoke test)":       "unsloth/Llama-3.2-1B-Instruct",
    "8B (stronger, slower)": "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
}
MODEL_KEY      = "3B (recommended)"      # <- pick a key from MODEL_CHOICES
MODEL_NAME     = MODEL_CHOICES[MODEL_KEY]
CHAT_TEMPLATE  = "llama-3.1"             # same template for all three models above

MAX_SEQ_LENGTH = 3072                    # examples are ~900-1,900 tokens; leave head-room
LOAD_IN_4BIT   = True                    # QLoRA: base weights in 4-bit, only LoRA weights are trained

LORA_R         = 16
LORA_ALPHA     = 16
EPOCHS         = 4
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 1                       # per step; long examples + T4 memory
GRAD_ACCUM     = 4                       # effective batch = BATCH_SIZE * GRAD_ACCUM = 4
SEED           = 3407
SMOKE_TEST     = False                   # True = 5 training steps only

print("Model:", MODEL_NAME)
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 3 - Load the data
Pick **one** way to get the two files into Colab:

* `"upload"` - a file picker opens; choose `train.jsonl` and `validation.jsonl` (from your repo's `data/finetune/`).
* `"drive"` - mount Google Drive and read them from a folder (set `DRIVE_FOLDER`).

Each line is one chat: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`. The `assistant` message is the JSON answer the model must learn to produce.
""")
code(r'''
import os

DATA_SOURCE  = "upload"                              # "upload" or "drive"
DRIVE_FOLDER = "/content/drive/MyDrive/insurance_finetune"

if DATA_SOURCE == "upload":
    from google.colab import files
    print("Select train.jsonl and validation.jsonl ...")
    files.upload()
    TRAIN_PATH, VAL_PATH = "train.jsonl", "validation.jsonl"
else:
    from google.colab import drive
    drive.mount("/content/drive")
    TRAIN_PATH, VAL_PATH = f"{DRIVE_FOLDER}/train.jsonl", f"{DRIVE_FOLDER}/validation.jsonl"

for p in (TRAIN_PATH, VAL_PATH):
    assert os.path.exists(p), f"{p} not found - upload exactly train.jsonl and validation.jsonl (not 'train (1).jsonl')"
''')
code(r'''
from datasets import load_dataset

raw = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": VAL_PATH})
print(raw)

# Sanity checks on what we are about to train on
for split in ("train", "validation"):
    for row in raw[split]:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"], roles
print("OK: every example is system / user / assistant")
print("\n--- one assistant answer (the target the model must learn) ---")
print(raw["train"][0]["messages"][2]["content"][:600], "...")
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 4 - Load the base model in 4-bit
`FastLanguageModel.from_pretrained` downloads the model (Unsloth's mirror - no Hugging Face login needed for these ids) and loads it quantised to 4-bit, which is what lets a 3B-8B model train on a free T4 (QLoRA). `dtype=None` lets Unsloth pick the right precision for the GPU.
""")
code(r'''
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = None,             # auto: float16 on T4, bfloat16 on newer GPUs
    load_in_4bit   = LOAD_IN_4BIT,
    # token = "hf_...",                # only needed for gated models you load from their original repo
)
''')

md(r"""
## Step 5 - Attach LoRA adapters
Instead of changing all the model's weights, LoRA adds small trainable matrices next to the attention and MLP layers (`q/k/v/o_proj`, `gate/up/down_proj`). Only these are trained - typically 1-2% of the parameters - so training is fast, memory-light, and the result is a **small adapter file** instead of a whole new model.

`use_gradient_checkpointing="unsloth"` trades a little speed for much lower memory, which matters with ~1,500-token examples.
""")
code(r'''
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_R,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0,                       # 0 is the optimised setting
    bias = "none",                          # "none" is the optimised setting
    use_gradient_checkpointing = "unsloth", # ~30% less VRAM for long examples
    random_state = SEED,
    use_rslora = False,
    loftq_config = None,
)
model.print_trainable_parameters()
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 6 - Chat template, and a length check
The model was trained with a specific chat format (special tokens marking system / user / assistant turns). `get_chat_template` makes the tokenizer use it, and `apply_chat_template` turns each `messages` list into the single text string the trainer learns from.

Then we **count tokens** for every example. If one exceeds `MAX_SEQ_LENGTH` the end of its JSON answer would be cut off and the model would learn broken JSON - so we stop if that happens.
""")
code(r'''
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template = CHAT_TEMPLATE)

def to_text(batch):
    # messages -> one string, WITH the assistant answer (this is what is trained on)
    return {"text": [tokenizer.apply_chat_template(m, tokenize = False, add_generation_prompt = False)
                     for m in batch["messages"]]}

train_ds = raw["train"].map(to_text, batched = True)
val_ds   = raw["validation"].map(to_text, batched = True)

lengths = [len(tokenizer(t, add_special_tokens = False).input_ids) for t in list(train_ds["text"]) + list(val_ds["text"])]
print(f"{len(lengths)} examples | tokens: min {min(lengths)}  avg {sum(lengths)//len(lengths)}  max {max(lengths)}  (limit {MAX_SEQ_LENGTH})")
assert max(lengths) <= MAX_SEQ_LENGTH, "An example is longer than MAX_SEQ_LENGTH: raise it in Step 2 (needs more memory)."
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 7 - Baseline: how good is the model *before* training?
We measure the untrained model on the validation claims, so the improvement is visible rather than assumed. (Right after `get_peft_model` the LoRA weights are zero, so this *is* the base model.)

For each validation claim we generate an answer greedily (`do_sample=False`, so it is repeatable) and compare it with the expected JSON:

* **valid JSON** - did it produce parseable JSON at all?
* **field accuracy** - share of the fields (11 to 26 per claim) whose value matches exactly (a missing or invented field counts as wrong)
* **record exact match** - share of claims where *every* field is right
* **per-field accuracy** - which fields are hard (usually incident type, dates and addresses)
""")
code(r'''
import json, re, collections

def extract_json(text):
    """Parse the model's answer; tolerate a code fence or text around the JSON. None if it cannot be parsed."""
    text = text.strip()
    for candidate in (text, *re.findall(r"\{.*\}", text, flags = re.S)):
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except ValueError:
            continue
    return None

def flatten(obj, prefix = ""):
    """{'a': {'b': 1}, 'c': [{'d': 2}]} -> {'a.b': 1, 'c[0].d': 2}"""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

def generic(path):
    return re.sub(r"\[\d+\]", "", path)       # insured_objects[1].year -> insured_objects.year

def score(expected_list, predicted_list):
    """expected_list/predicted_list: parallel lists of dicts (predicted may be None when unparseable)."""
    valid = exact = 0
    right, total = collections.Counter(), collections.Counter()
    for exp, pred in zip(expected_list, predicted_list):
        fe = flatten(exp)
        fp = flatten(pred) if pred is not None else {}
        valid += pred is not None
        all_ok = pred is not None
        for path, value in fe.items():
            ok = path in fp and fp[path] == value
            right[generic(path)] += ok
            total[generic(path)] += 1
            all_ok &= ok
        all_ok &= set(fp) <= set(fe)           # invented extra fields also count as a mistake
        exact += all_ok
    n = len(expected_list)
    fields_ok, fields_all = sum(right.values()), sum(total.values())
    return {"n": n, "valid_json": valid / n, "record_exact": exact / n, "field_accuracy": fields_ok / fields_all,
            "per_field": {k: right[k] / total[k] for k in total}}

def generate_answer(messages, max_new_tokens = 900):
    """messages = [system, user]; returns the model's text answer."""
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True,
                                           return_tensors = "pt").to("cuda")
    out = model.generate(input_ids = inputs, attention_mask = torch.ones_like(inputs),   # one unpadded sequence
                         max_new_tokens = max_new_tokens, use_cache = True, do_sample = False)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens = True)

def evaluate(dataset, name):
    FastLanguageModel.for_inference(model)      # Unsloth: switch to the fast inference mode
    expected, predicted, raw_answers = [], [], []
    for row in dataset:
        msgs = row["messages"]
        raw_answers.append(generate_answer(msgs[:2]))
        predicted.append(extract_json(raw_answers[-1]))
        expected.append(json.loads(msgs[2]["content"]))
    result = score(expected, predicted)
    result["raw_answers"] = raw_answers
    print(f"\n=== {name} ({result['n']} validation claims) ===")
    print(f"valid JSON:       {result['valid_json']:.0%}")
    print(f"field accuracy:   {result['field_accuracy']:.1%}")
    print(f"record exact:     {result['record_exact']:.0%}")
    worst = sorted(result["per_field"].items(), key = lambda kv: kv[1])[:8]
    print("hardest fields:   " + ", ".join(f"{k} {v:.0%}" for k, v in worst))
    return result
''')
code(r'''
baseline = evaluate(val_ds, "BASELINE (before fine-tuning)")
print("\n--- what the untrained model answered for the first claim ---")
print(baseline["raw_answers"][0][:700])
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 8 - Build the trainer
`SFTTrainer` (supervised fine-tuning) feeds the formatted `text` to the model. Settings worth knowing:

* `packing=False` - one claim per sequence (our claims are long; packing helps short ones).
* `eval_strategy="epoch"` - measures the validation loss after every epoch so you can see over-fitting (train loss falling while validation loss rises).
* `optim="adamw_8bit"` - a memory-saving optimiser.

**`train_on_responses_only`** is important: it masks everything except the assistant's JSON answer, so the loss (and the learning) is only about *producing the answer*, not about re-learning the long instruction or the claim text. The cell afterwards prints the part that is still trained - it should be only the JSON.
""")
code(r'''
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

FastLanguageModel.for_training(model)           # back to training mode after the baseline run

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_steps = 2,
        num_train_epochs = EPOCHS,
        max_steps = 5 if SMOKE_TEST else -1,    # -1 = use num_train_epochs
        learning_rate = LEARNING_RATE,
        logging_steps = 1,
        eval_strategy = "epoch",
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = SEED,
        output_dir = "outputs",
        report_to = "none",
    ),
)

from unsloth.chat_templates import train_on_responses_only
try:
    trainer = train_on_responses_only(trainer)   # newer Unsloth detects the answer markers itself
except TypeError:
    trainer = train_on_responses_only(           # older Unsloth: name the Llama-3 markers explicitly
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
''')
code(r'''
# Verify the masking: this prints ONLY the tokens that count towards the loss. It must be the JSON answer alone.
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[0]["labels"]]))
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 9 - Train
`logging_steps=1` prints the training loss every step; the validation loss appears after each epoch. With ~21 examples this takes a few minutes on a T4. A healthy run shows the loss dropping quickly in the first epoch and then flattening.
""")
code(r'''
gpu = torch.cuda.get_device_properties(0)
start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"GPU = {gpu.name}. Max memory = {round(gpu.total_memory / 1024**3, 3)} GB. {start_mem} GB reserved before training.")

trainer_stats = trainer.train()

used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
print(f"\nTraining took {round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes. Peak GPU memory: {used} GB.")
''')
code(r'''
# Loss per epoch: training loss should fall; if validation loss starts rising, you are over-fitting (use fewer epochs).
for entry in trainer.state.log_history:
    if "eval_loss" in entry:
        print(f"epoch {entry['epoch']:.0f}: validation loss {entry['eval_loss']:.4f}")
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 10 - Evaluate again and compare
Same validation claims, same greedy decoding, same scoring as the baseline - now with the LoRA adapter trained.

How to read it with 5 validation claims: **one claim = 20 percentage points** on "record exact", so look at *field accuracy* and *valid JSON* first, and at the hardest fields. A big jump in valid JSON and field accuracy means the model learned the format; whether it generalises needs far more validation data.
""")
code(r'''
tuned = evaluate(val_ds, "FINE-TUNED")

# If Step 7 was skipped (or the session restarted) there is no baseline in memory. We can still get one:
# switching the LoRA adapter off gives back the untrained base model.
if globals().get("baseline") is None:
    try:
        with model.disable_adapter():
            baseline = evaluate(val_ds, "BASELINE (LoRA adapter switched off)")
    except Exception as e:
        baseline = None
        print("Could not compute a baseline now:", e)

print("\n=== BEFORE vs AFTER ===")
print(f"{'':<18}{'baseline':>10}{'fine-tuned':>12}")
for key, label in (("valid_json", "valid JSON"), ("field_accuracy", "field accuracy"), ("record_exact", "record exact")):
    before = f"{baseline[key]:.1%}" if baseline else "n/a"
    print(f"{label:<18}{before:>10}{tuned[key]:>12.1%}")

print("\n--- fine-tuned answer for the first validation claim ---")
print(tuned["raw_answers"][0][:700])
''')
code(r'''
# Where does the fine-tuned model still go wrong? (per-field accuracy, worst first)
for path, acc in sorted(tuned["per_field"].items(), key = lambda kv: kv[1]):
    print(f"{acc:>5.0%}  {path}")
''')

code(r"""
# Every wrong field, claim by claim: what was expected and what the model said.
# Read this before deciding what to change - it shows WHAT kind of mistakes the model makes.
for n, (row, answer) in enumerate(zip(val_ds, tuned["raw_answers"])):
    expected = flatten(json.loads(row["messages"][2]["content"]))
    parsed = extract_json(answer)
    got = flatten(parsed) if parsed else {}
    wrong = [(p, expected[p], got.get(p, "<missing>")) for p in expected if got.get(p, "<missing>") != expected[p]]
    print(f"\nclaim {n}: {len(expected) - len(wrong)}/{len(expected)} fields right")
    for path, want, said in wrong:
        print(f"   {path}\n      expected: {want!r}\n      model:    {said!r}")
    for path in got:
        if path not in expected:
            print(f"   (invented field) {path} = {got[path]!r}")
""")

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 11 - Save the LoRA adapter and download it
`save_pretrained` writes **only the adapter** (the small trained LoRA weights, typically tens of MB) plus the tokenizer - not the whole base model. We also write a little `training_info.json` so you remember which base model, settings and scores produced it.

The last cell zips the folder and **downloads it to your computer**. (Colab's disk is wiped when the session ends, so download it - or copy it to Drive - before you close the tab.)
""")
code(r'''
import json, os

ADAPTER_DIR = "lora_adapter"
model.save_pretrained(ADAPTER_DIR)          # LoRA weights + adapter_config.json
tokenizer.save_pretrained(ADAPTER_DIR)

info = {
    "base_model": MODEL_NAME, "chat_template": CHAT_TEMPLATE, "max_seq_length": MAX_SEQ_LENGTH,
    "load_in_4bit": LOAD_IN_4BIT, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA, "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "seed": SEED,
    "train_examples": len(train_ds), "validation_examples": len(val_ds),
    "baseline": ({k: baseline[k] for k in ("valid_json", "field_accuracy", "record_exact")}
                 if globals().get("baseline") else None),
    "fine_tuned": {k: tuned[k] for k in ("valid_json", "field_accuracy", "record_exact")},
}
with open(f"{ADAPTER_DIR}/training_info.json", "w") as f:
    json.dump(info, f, indent = 2)

print("Saved:", sorted(os.listdir(ADAPTER_DIR)))
''')
code(r'''
# >>> DOWNLOAD THE ADAPTER (zip the folder, then download to your computer) <<<
import shutil
from google.colab import files

shutil.make_archive("lora_adapter", "zip", "lora_adapter")
files.download("lora_adapter.zip")
''')

# --------------------------------------------------------------------------------------------------------
md(r"""
## Step 12 (optional) - Keep it somewhere safer, or use it
Everything below is switched off (`if False:`) - change it to `True` for the one you want.

* **Google Drive copy** - survives the session ending.
* **Hugging Face Hub** - needs a token from <https://huggingface.co/settings/tokens> (write access). Keep the repo **private**: the adapter was trained on de-identified claims, but it is still derived from your data.
* **Merged model / GGUF** - turn "base + adapter" into one standalone model (for serving or for llama.cpp / Ollama). Much larger files than the adapter.
* **Reload the adapter later** - in a fresh Colab session, install Unsloth (Step 1), upload and unzip `lora_adapter.zip`, then run the reload cell.
""")
code(r'''
# --- copy the adapter to Google Drive ---
if False:
    from google.colab import drive
    drive.mount("/content/drive")
    !cp -r lora_adapter "/content/drive/MyDrive/insurance_lora_adapter"

# --- push the adapter to the Hugging Face Hub (private) ---
if False:
    model.push_to_hub("YOUR_HF_USERNAME/insurance-claims-lora", private = True, token = "hf_...")
    tokenizer.push_to_hub("YOUR_HF_USERNAME/insurance-claims-lora", private = True, token = "hf_...")

# --- one standalone model with the adapter merged in (16-bit) ---
if False:
    model.save_pretrained_merged("insurance_merged_16bit", tokenizer, save_method = "merged_16bit")

# --- GGUF for llama.cpp / Ollama (q4_k_m is a good size/quality trade-off) ---
if False:
    model.save_pretrained_gguf("insurance_gguf", tokenizer, quantization_method = "q4_k_m")
''')
code(r'''
# --- reload the saved adapter in a new session and run it on one claim ---
if False:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_adapter",            # the unzipped folder (it remembers its base model)
        max_seq_length = 3072,
        dtype = None,
        load_in_4bit = True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")
    FastLanguageModel.for_inference(model)

    system_prompt = raw["train"][0]["messages"][0]["content"]       # the same instruction used in training
    claim_text = "PASTE ONE (DE-IDENTIFIED) CLAIM HERE"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": claim_text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True,
                                           return_tensors = "pt").to("cuda")
    out = model.generate(input_ids = inputs, max_new_tokens = 900, use_cache = True, do_sample = False)
    print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens = True))
''')

md(r"""
## What to do next
1. **More data is the biggest lever.** Add many more claims to the repository pipeline (`py -m pipeline`, then `py -m pipeline.finetune`) and re-run this notebook on the new `train.jsonl` / `validation.jsonl`.
2. **Fill the held-back records.** Decide the conflicting records in `data/review/decisions.jsonl` to recover them (24 / 6 instead of 21 / 5).
3. **Compare models** - run Step 2 with the 8B model and compare the field accuracy table.
4. **Keep a test set once you have more data.** The export deliberately has only train + validation; with a larger set, hold out claims the model never sees during tuning.
5. **Never judge quality on 5 claims.** Use this notebook's numbers to catch broken runs, not to promise accuracy.
""")

# --------------------------------------------------------------------------------------------------------
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4", "name": "finetune_unsloth_colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
for i, c in enumerate(cells):
    c["id"] = f"cell{i:02d}"      # nbformat 4.5 wants a unique id per cell

out = Path(__file__).resolve().parent / "finetune_unsloth_colab.ipynb"
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"wrote {out} with {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code)")
