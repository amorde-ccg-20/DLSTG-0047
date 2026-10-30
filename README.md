# Q2

Run these commands from this directory in a Linux terminal. The `.venv-ready` folder contains the ready-to-use GPU-capable Python environment on this machine. `data/` is for the downloaded dataset and is not committed to Git; `outputs/` contains the results.

## Environment

The `.venv-ready` environment is set up on this machine using packages from the existing `ai623-dvlm` Conda environment. It is ready to use; do not run `pip install` or recreate it. Activate it with:

```bash
source .venv-ready/bin/activate
```

Check that PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The current environment has CUDA-enabled PyTorch and OpenCV installed. The Part 1 optical-flow script uses the CPU: the assignment specifies OpenCV's `calcOpticalFlowFarneback` method. Later PyTorch models must explicitly move their models and tensors to `cuda` to use the GPU.

Only if setting up an independent CUDA environment on another machine with a compatible NVIDIA driver and Python 3.10 or newer:

```bash
python3 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-gpu.txt
```

For a CPU-only environment instead:

```bash
python3 -m venv .venv-cpu
source .venv-cpu/bin/activate
pip install -r requirements-cpu.txt
```

The independent environments require a substantial download. On this machine, use `source .venv-ready/bin/activate` when opening a new terminal.

## Step 1: Download and inspect

```bash
python 1_data.py
```

This downloads approximately 781 MiB into `data/MovingMNIST/`, splits videos by index, converts frames to float32 in [0, 1], and prints the split sizes and the first video's shape and range. Loading the complete dataset requires substantial memory.

## Step 2: Check transformations and windows

```bash
python 2_checks.py
```

Read the checks printed in the terminal and saved in `outputs/table1.txt`. Checks (a)–(c) print the maximum difference. Compare the results with the expected values in the handout. Frame indices in the marker checks are zero-based; the window frame numbers are one-based.

`common.py` contains the mirror, rotation and window functions used by these checks; use these same functions in later experiments. `window(x, t, k)` takes the one-based start frame `t` and speed `k`, returning up to five input and five available target frames.

## Step 3: Inspect motion in the training set

```bash
python 3_flow.py
```

The script processes all 19 consecutive frame pairs in the first 50 training videos and saves `outputs/figure1.pdf`, `outputs/figure1.png`, and `outputs/flow.txt`. The figure shows fractions of kept pixels: direction uses 18 bins, and the speed panel marks the median. Use the median to calculate the corresponding 2× and 3× speeds. Consider whether overlapping digits could affect optical flow when answering Q2.1.

## Report

Review Table 1, Figure 1, and the answers to Q2.1 and Q2.2 in `report/main.tex`. Figure 1 is included automatically once generated. Add the remaining question parts when completing the full assignment.

To compile with a LaTeX installation containing `pdflatex`:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
cd ..
```

The report is `report/main.pdf`. Running the commands from the `report` directory makes the figure path resolve correctly.

## Part 2: Baseline ConvLSTM

Train model A with both seeds. This uses the GPU when available and saves the final weights and mean training time in `outputs/`:

```bash
python 4_train.py
```

Evaluate all 1,000 test videos and make Figure 2 with three chosen test video indices. Inspect the videos and choose at least one where the digits overlap; replace the example indices below with your selections. Indices are the original dataset indices, from 9000 to 9999.

```bash
python 5_evaluate.py --videos 9000 9001 9002
```

The results are saved as `outputs/table2.txt` and `outputs/figure2.pdf` (also PNG).

## Part 3: Testing the baseline

Run the mirror and rotation tests using both trained seeds:

```bash
python 6_symmetry.py
```

This saves Table 3, Figure 4, and the four Figure 3 panels and GIFs in `outputs/`. The printed test-video indices identify the two largest consistency gaps for each transformation. Include the corresponding GIF paths in the Figure 3 caption.

Run the speed tests, choosing a test-video index between 9000 and 9999:

```bash
python 7_speed.py --video 9007
```

This saves Table 4 and Figure 5 in `outputs/`. Inspect Figure 5 and choose another index if the digits overlap in any of the displayed true frames. The speed tests evaluate all 1,000 test videos regardless of the selected figure index.

## Part 4: What Can Data Fix?

Train models B, C and D with seeds 0 and 1:

```bash
python 8_train.py
```

B uses symmetry augmentation at 1×; C uses speed augmentation with model A's architecture; D uses speed augmentation and dilated convolutions. D has 480,784 parameters and a 16-pixel one-step reach. The script saves the final weights and mean training times in `outputs/`.

After training, calculate Table 5 on all 1,000 test videos:

```bash
python 9_table5.py
```

The table is saved as `outputs/table5.csv`, and the hardware name as `outputs/table5_hardware.txt`. Re-running training replaces the saved B–D weights and timings.

## Part 5: The final model

After Part 4 training, run:

```bash
python 10_final.py
```

This wraps models B and D with all eight square symmetries and evaluates both seeds. The final model is symmetrised D. `outputs/table6.csv` contains the results and `outputs/table6_hardware.txt` records the hardware. Inference timing includes all eight predictions and their transformations.

Figure 6 uses the Part 3 video selections stored in `outputs/figure3_videos.txt` and `outputs/figure5_video.txt`. The four mirror and rotation panels and GIFs are saved as `outputs/figure6_*`; the speed comparison is `outputs/figure6_speed.pdf` (also PNG). If you change the Part 3 video selection, rerun its script before generating Figure 6.
