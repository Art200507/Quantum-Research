# Ten experiments: can phase-inspired memory earn an efficiency advantage?

Date: 26 October 2026. This is an exploratory local research sprint, not a paper.

## Answer in easy words

We tried to put many relationships into one compact mathematical memory. That
worked when there were few relationships, but their signals interfered as we
added more. Dividing memory into banks helped until slightly damaged queries
went to the wrong bank. A simple trained classical layer did much better.

The useful result is a tiny classical memory whose weights can be stored in
about 4 KB. It recalled 256 stored relationships accurately in the tested mild
conditions and can be updated carefully. It failed more often when overloaded
or given badly corrupted queries. We did not invent quantization or associative
memory, and nothing here demonstrates a quantum advantage.

## What was actually tested

Each stored key is a random vector, associated with one of 16 labels. The model
must return its label when shown the original key or a perturbed version. Random
labels make this a memorization/retrieval task: there is no meaningful way to
predict the independent label of an unseen key. No language data, conversational
reasoning or full transformer was tested. A ridge-fitted linear layer is the
trained neural baseline; it is not a deep model or a QNN.

Phase memory uses complex64 unit phases: m=sum_i k_i*v(label_i). Retrieval
unbinds with conjugate(q) and chooses the class prototype with greatest real
inner product. This is classical holographic-style binding. A real-vector
version and direct nearest-key retrieval provide controls. Noise means independent
Gaussian perturbations of each key phase, measured in radians, not hardware noise.
All classes are compared each query. Chance accuracy is 6.25%.

Direct linear memory forms W=K^H Y/d with one-hot class labels Y. It retrieves
Re(QW). Ridge regression fits a linear neural layer from real/imaginary key
features to Y with fixed regularization .1. Training labels are stored
associations; perturbed queries test robustness of recall. This is supervised
weight fitting, with no backpropagated key encoder. No meaningful claim about
an unseen semantic task follows.

## The ten cycles

Each result was inspected and its decision entered in JOURNAL.md before running
the next experiment. Rows and settings are in results/01.json through 10.json.

| # | Question/refinement | Actual result | Decision |
|---|---|---|---|
| 1 | Can phase binding retrieve a light load? | 98.96% at 16 pairs/d128; exact 100% | Test load; include the large decoder codebook |
| 2 | Does fixed-size memory scale? | At 256 pairs, only 26.04%; exact 100% | Investigate representation rather than claim capacity |
| 3 | Are phases better at equal bytes? | No consistent win over real binding | Reject a complex-number advantage claim |
| 4 | Does increasing dimension rescue it? | 64-pair recall rises 41.67 to 97.92%, with 8x array storage | Add stronger baselines |
| 5 | Does a trained classical layer do better? | At 256 pairs: binding 26.04%, linear 96.875%, ridge 100% | Pursue simpler task-matched models |
| 6 | Can routing handle corrupted keys? | 16-bank recall drops 97.36 to 71.81% at mild noise; ridge 100% | Reject fragile routing as primary model |
| 7 | Can we reduce precision? | Ridge/linear state 16,384 to 4,100 bytes with unchanged mild-noise accuracy | Test dynamic writes |
| 8 | Can memory replace stale relationships? | Appending 46.88%, corrected 100%, corrected int8 99.93% after overwrites | Count previous-label bookkeeping |
| 9 | Does qubit encoding preserve the computation? | Circuit probabilities match math; squared-overlap readout loses signed information and accuracy | No hardware run justified |
| 10 | Does the candidate survive fresh seeds and a compressed exact baseline? | 100% at 256 pairs/mild noise with 4,100-byte state; fails under heavier load/noise | Keep a narrow classical baseline, not a transformer claim |

Experiments 1–8 use 12 random memories per condition, mostly seeds 100–111.
Some conditions intentionally repeat to make comparisons and identities explicit;
these are ten adaptive experiment cycles, not ten independent discoveries.
Experiment 9 has 12 memories per load and four classes. Experiment 10 uses 24
fresh seeds 1000–1023, with frozen models. Conditions within a seed are dependent;
uncertainty is summarized across independent memory seeds, not by pretending all
queries are independent experiments. The fresh-seed test is held out from model
choice; it is not a test on new semantic tasks.

## Fresh-seed result that is worth retaining

256 pairs, d128 complex inputs, sigma=.3 radians, 16 output classes, 24 memories:

| Method | Mean recall | Persistent array payload |
|---|---:|---:|
| Single-vector phase binding, including decoder | 24.43% | 17,408 B |
| Direct linear memory, float | 96.13% | 16,384 B |
| Direct linear memory, int8 | 96.14% | 4,100 B |
| Trained ridge layer, float32 | 100% | 16,384 B |
| Trained ridge layer, int8 | 100% | 4,100 B |
| Nearest-key retrieval, complex64 keys | 100% | 263,168 B |
| Nearest-key retrieval, int8 real/imaginary keys | 100% | 66,564 B |

The quantized ridge layer has 16.24x smaller array payload than the compressed
nearest-key store. This is a task-specific storage result, not a universal
compression ratio or an advantage over the best possible classical implementation.
A stable symbolic-key dictionary could be far more appropriate for exact lookup;
if slot IDs are supplied at query time, labels alone can answer it. Our static
queries supply continuous keys, not slot IDs. We do not claim superiority over
hash tables or approximate-nearest-neighbor libraries.

Across-seed mean±SE at n256/sigma.3 is 96.14%±0.225 percentage points for linear-int8
and 24.43%±0.478 points for binding. All ridge seeds succeeded in this finite
sample; zero observed between-seed variation is not a certainty interval for
future tasks. At n512/sigma.3 ridge-int8 mean recall falls to 93.51%; at
n256/sigma.8 it is 80.99%. Nearest-key baselines remain 100% on these sampled
conditions. The storage result trades away robustness/capacity.

## Why the quantum test matters

Experiment 9 is an actual Qiskit implementation on local ideal statevectors:
prepare normalized memory on 3 qubits, unbind query phases, inverse-prepare a
prototype, and measure all-zero probability. Across 672 circuits, agreement with
classical squared overlaps is within 3.11e-15. Compilation uses 14 CX and depth
27 per candidate on an all-to-all U/CX basis, including preparation; this is not
an IBM-native hardware estimate. Four candidates need four fresh preparations.
The finite-shot tests use independent binomial sampling from verified ideal
probabilities: 128/2048 shots per candidate, 512/8192 per query.

At four associations, signed classical readout gives 91.67% versus 77.08% ideal
quantum-overlap and 72.92% at 128 shots/candidate. Mapping a classical vector to
a quantum state changes accessible information: overlap probabilities square the
magnitude, losing the signed score. More shots approach that different readout,
not the original function. This is a failed encoding/readout approach, not a
proof that every QNN must fail. No variational QNN was trained, no QRAM assumed,
no hardware/energy advantage measured. State-preparation instructions and the
classical source data are part of the workload, not free quantum storage.

## Cost accounting and boundaries

Payload counts include the model's memory and output codebook, or all stored
keys and 32-bit labels for nearest retrieval. Int8 counts include a float32 scale.
Decoder tables dominate the vector-memory method. Experiment 8 additionally
charges a 512-byte previous-label table and explicitly assumes updates supply
stable slot IDs, full keys and new labels. Inputs/codebooks are generated from
seeds for reproducibility, but seed compression is not used as an algorithmic
advantage. The input generator is a shared external interface, not a learned
encoder whose cost has been evaluated.

In experiments 7–8, packed bytes are a serialization-equivalent ledger while
working arrays are dequantized. Experiment 10 retains actual packed payloads
in the prediction closures and reconstructs query workspace as needed.
We do not measure total process RAM, temporary workspace, Python object overhead,
key encoding, encoder parameters or training-set storage as persistent inference
state. Builds temporarily hold data/normal equations and can cost much more than
the final layer. Exact batched retrieval uses an N-by-N similarity workspace;
chunking can reduce that. Query batches can obscure single-query costs. Int8
weights are dequantized for arithmetic: no int8 kernel or low-power claim.

Timing is a diagnostic on one machine, one fresh seed per condition, 25 warmed
full-batch repetitions, with BLAS thread limits requested through environment
variables. At n256/mild noise, median query batch times are 69.67 microseconds for
ridge-int8 versus 280.75 for compressed nearest retrieval, but simple linear
memory is faster and ridge has extra build cost. Build cost and timing spread
are retained in 10.json. This is not an end-to-end benchmark or measured energy
saving. No 100x computation or power improvement was demonstrated.

## What would make the next stage useful?

Study a **quantized updateable associative-memory module**, treating quantum-inspired
phases as an optional representation rather than the source of a presumed gain.
Compare float/int8 linear and ridge baselines, robust routing, and actual
attention-based recall on structured sequences. Test absent keys, conflicting
keys, mixed retrieval/update workloads and distribution shifts. Charge previous
labels, embeddings, rebuilding, workspace, encoding and memory movement. Define
accuracy/latency/storage targets before choosing a model.

Before adding an LLM or making a publication proposal, reproduce a small published
associative-recall benchmark and compare against modern classical methods. Only
then decide whether adaptive precision, update handling or robust compressed
recall offers an unresolved question. Our current results justify a baseline and
a better scoped next experiment; they do not establish novelty.
