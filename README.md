# SMELL: Multi-file Python reproduction

This repository is an engineering reproduction of the method described in the supplied manuscript **SMELL: Jointly Optimized Multimodal Representation and Curriculum Scheduling for Social Media-Based English Language Learning**.

## What is reproduced

- ResNeXt-50 32x4d visual backbone.
- Squeeze-and-Excitation recalibration.
- Six-layer, 512-dimensional, eight-head text Transformer.
- Bidirectional image-region/text-token cross-attention.
- 256-dimensional fused pedagogical embedding.
- Five-state proficiency HMM with neural emissions for incorrect/correct/skipped outcomes.
- Posterior-derived target difficulty and nearest-difficulty curriculum selection.
- AFOA with quadratic inertia decay, personal/global best terms, Levy-flight perturbation, diversity trigger, and finite-difference local refinement.
- AdamW training and cosine learning-rate decay.

## Important reproducibility caveats

The manuscript does **not** publish an exact loss for all jointly optimized components, the exact formula of the `SimulatedProficiencyGain` fitness simulator, preprocessing details sufficient to reconstruct all benchmark labels, or downloadable code/checkpoints. Therefore:

1. `src/curriculum.py::simulated_proficiency_gain` is explicitly an **engineering proxy** so the AFOA loop can be executed. It is not presented as the paper's hidden original simulator.
2. The classification head uses cross-entropy for CEFR/difficulty labels because a complete end-to-end training objective is not specified in the manuscript.
3. The paper mentions ImageNet-21K initialization. torchvision does not provide that exact ResNeXt-50 32x4d checkpoint, so the built-in `visual_pretrained: true` option uses torchvision's default ImageNet pretrained weights unless you load an external checkpoint yourself.
4. BLEU, RCP and SPG require generated exercises or offline sequence simulation; they cannot be reconstructed from difficulty-class logits alone.

## Project structure

```text
smell_reproduction/
├── configs/default.yaml
├── requirements.txt
├── src/
│   ├── curriculum.py
│   ├── train_utils.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── vocab.py
│   ├── models/
│   │   ├── se.py
│   │   ├── visual_encoder.py
│   │   ├── text_encoder.py
│   │   ├── cross_modal.py
│   │   ├── proficiency_hmm.py
│   │   └── smell.py
│   ├── optim/afoa.py
│   └── utils/config.py
├── scripts/
│   ├── make_synthetic_data.py
│   ├── train_encoder.py
│   ├── extract_embeddings.py
│   ├── fit_hmm.py
│   ├── optimize_curriculum.py
│   ├── evaluate.py
│   ├── evaluate_model.py
│   ├── plot_reported_results.py
│   └── demo_pipeline.py
└── tests/test_afoa.py
```

## Expected dataset format

`train.csv`:

```csv
image_path,text,label,sequence_id,timestep,outcome
001.jpg,"A short social-media caption",A2,learner_001,0,correct
002.jpg,"Another caption",B1,learner_001,1,incorrect
```

Required columns: `image_path,text,label`.

For HMM fitting also provide `outcome`, encoded as `incorrect/correct/skipped` or `0/1/2`. `sequence_id` and `timestep` are recommended for multiple offline sequences.

## Installation

```bash
cd smell_reproduction
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Fast pipeline test

```bash
python scripts/demo_pipeline.py
```

This avoids the full ResNeXt training run and checks cross-attention, HMM filtering, curriculum selection and AFOA.

## Fully runnable synthetic example

Generate toy images and sequences:

```bash
python scripts/make_synthetic_data.py --out data --n 120
```

Train the multimodal encoder:

```bash
python scripts/train_encoder.py --config configs/default.yaml
```

For a quick CPU test, edit `configs/default.yaml`: set `epochs: 1`, `batch_size: 2`, `device: cpu`.

Extract item embeddings and learned difficulty scores:

```bash
mkdir -p outputs
python scripts/extract_embeddings.py --checkpoint outputs/checkpoints/smell_best.pt
```

Fit the neural-emission HMM and perform a Baum-Welch-style transition update:

```bash
python scripts/fit_hmm.py --embeddings outputs/embeddings.npz
```

Optimize curriculum settings with AFOA:

```bash
python scripts/optimize_curriculum.py --embeddings outputs/embeddings.npz
```

## Mapping from paper equations to code

- ResNeXt aggregated visual mapping / SE recalibration: `src/models/visual_encoder.py`, `src/models/se.py`.
- Bidirectional cross-attention and fusion: `src/models/cross_modal.py`.
- Fused representation z and embedding-derived difficulty: `src/models/smell.py`.
- Neural emissions, HMM posterior and target difficulty: `src/models/proficiency_hmm.py`.
- AFOA position update, Levy perturbation, diversity and local finite differences: `src/optim/afoa.py`.
- Top-K difficulty-matched delivery: `src/curriculum.py`.

## Suggested real-data workflow

1. Convert the benchmark to the CSV format above and place images under `data/images/`.
2. Build train/validation/test splits externally to avoid leakage across social-media source or learner sequence.
3. Train the multimodal encoder on difficulty labels.
4. Extract frozen post embeddings.
5. Fit the HMM on chronological offline interaction sequences.
6. Define the exact SPG/RCP simulator used by your experiment and replace the proxy function.
7. Run AFOA on training/validation histories only; evaluate its final curriculum on held-out sequences.
8. Repeat with three random seeds, matching the manuscript's reporting protocol.


## Plot the manuscript-reported main results

`data_paper_reported_results.csv` contains only the values explicitly reported in the manuscript tables. Generate a comparison figure with:

```bash
python scripts/plot_reported_results.py
```

This plot is a visualization of **reported numbers**, not a claim that the current reproduction has independently recovered them.
