# Adaptive experiment journal — 26 October 2026

All experiments run locally. Phase arithmetic is classical, inspired by holographic
memory; it is not quantum hardware or a demonstrated quantum speedup. The task is
retrieving random key-to-class associations, not language modeling. Sixteen output
classes; chance accuracy 6.25%. Twelve independent generated memories per condition
unless stated. Seeds 100–111 are exploratory; final validation will use fresh seeds.

## 1 — Does phase binding retrieve anything?

Ran 16 stored associations with 128 complex components. Mean retrieval accuracy
98.96%, versus exact nearest-key 100%. Three algebraic checks passed. Importantly,
the 1,024-byte memory needs a 16,384-byte output codebook: total 17,408 bytes,
slightly more than exact storage's 16,448 bytes. Counting just the memory vector
would create an illusory compression win. This is a mechanism check, not learning.

Decision before experiment 2: keep dimension fixed and increase stored pairs.
Test whether the small-state representation retains accuracy under interference.

## 2 — Load breaks the single-vector memory

At 16/64/128/256 associations, phase accuracy is 98.96/57.42/39.58/26.04%.
Exact retrieval stays 100%. Compression grows, but its accuracy cost is severe.
The fixed vector is not lossless storage; superposed bindings interfere.

Decision before experiment 3: determine whether complex numbers offer an advantage
at equal bytes over a real-valued binding memory, rather than comparing unequal
numbers of floating-point scalars.

## 3 — Complex phases are not a clear equal-byte win

At matching 8,704-byte persistent arrays, phase versus real accuracies are
55.99 vs 63.54% (32 pairs), 26.56 vs 24.54% (128), and 19.08 vs 17.58% (256).
Differences depend on load and include random-data uncertainty. The encodings
are different, so this is not a paired identical-input comparison. There is no
consistent dramatic benefit and no quantum claim. A real-feature identity check
also reproduces complex inner products, reminding us that these computations
have direct classical representations.

Decision before experiment 4: increase dimension and charge the decoder; measure
whether capacity can be recovered economically rather than declaring phases bad
or good from one chosen dimension.

## 4 — Increasing capacity costs real storage

For 64 pairs, accuracy rises 41.67 -> 57.42 -> 84.64 -> 97.92% as dimension
64 -> 128 -> 256 -> 512. Persistent arrays rise 8,704 -> 69,632 bytes. At 256
pairs even d512 yields only 58.72%. More dimensions help, but do not remove the
capacity/accuracy trade-off. The experiment is not evidence of exponential memory.

Decision before experiment 5: compare direct key-to-class linear accumulation
and a trained linear neural layer (ridge fit) on identical keys/labels. The task
has discrete class outputs; avoid making all baselines pay for unnecessary
random value-vector decoding.

## 5 — A stronger classical baseline changes the conclusion

At 256 pairs and d128, binding achieves 26.04%; direct linear key-to-class
accumulation achieves 96.875% with 16,384 bytes versus binding's 17,408 bytes.
Ridge fitting achieves 100% with 32,768 bytes. Ridge uses float64 in this
implementation because normalization promotes precision; recorded bytes reflect
actual arrays. Exact nearest-key stays 100% but uses 263,168 bytes. Since labels
are random, these are stored-association recall scores, not generalization to new
random labels. Most compression comes from recognizing the task has only 16
output classes, not from quantum physics.

Decision before experiment 6: test corrupted queries and add routed memory banks.
Banks can reduce interference but may introduce routing failures; compare all
methods on the same corrupted inputs rather than perfect queries alone.

## 6 — Routing improves clean recall but is fragile

At 256 pairs/d128, 16 banks increase clean accuracy to 97.36%, but cost 32,768
bytes and fall to 71.81% under sigma=.3-radian query perturbations. Linear memory
uses 16,384 bytes and retains 95.74%; ridge retains 100% at that noise level.
At sigma=.8, linear/ridge reach 84.96/80.86%, exact nearest-key 100%, 16-bank
memory only 39.94%. Content routing is not free robustness. We reject banks as
the primary candidate for this task, rather than fixing retrieval using oracle IDs.

Decision before experiment 7: seek an implementable efficiency improvement in the
stronger classical model. Test float32 ridge and int8 persistent-state quantization.
Keep decoded workspace separate from stored payload; no claim of int8 execution.

## 7 — Quantization is a practical storage result

Float32 ridge matches earlier float64 accuracy with half the state bytes. Int8
linear/ridge payloads need 4,100 bytes versus 16,384 bytes; clean/mild-noise
accuracies are unchanged (linear 96.875/95.74%, ridge 100/100%). At sigma=.8,
quantized ridge is 81.09% versus 80.86% float32, a small incidental change, not
claimed improvement. Binding saves little overall because its decoder dominates.
Quantized arrays are dequantized for computation, so this is stored-weight
compression only. Query workspace, Python overhead and energy are not measured.

Decision before experiment 8: test dynamic updates. A reusable memory needs to
replace outdated relationships without indefinitely accumulating contradictory
ones. Compare naive appending, global decay, and subtract-old/add-new with its
required previous-label metadata charged.

## 8 — Dynamic memory needs deletion, not just more writes

After 256 overwrites of 128 associations, naive append recalls 46.88%, global
.98 decay 55.01%, correct subtract-old/add-new 100%. Requantizing corrected
weights after every update yields 99.93% recall, using 4,612 bytes including a
512-byte old-label table. Corrected float state matches a full rebuild within
1.88e-8. This result depends on an explicit stable-ID/full-key update interface;
we do not hide the need to know the previous label. It is a standard classical
update rule and not a new quantum architecture.

Decision before experiment 9: implement a genuinely quantum-circuit version of a
tiny phase memory locally. Test amplitude preparation, query phase unbinding and
prototype-overlap measurement, verify against classical Born probabilities, and
count state-preparation gates and shots. Do not equate complex NumPy with qubits.

## 9 — Encoding the memory in qubits does not preserve the same readout

Built and checked 672 three-qubit Qiskit circuits with amplitude-state preparation,
query-dependent diagonal phases and inverse prototype preparation. Circuit outcome
probabilities match classical squared overlaps within 3.11e-15. Compilation to
U/CX uses 14 CX gates and depth 27 per candidate, including preparation/unbinding/
prototype comparison but excluding physical routing and measurement. Four
candidates require 512 or 8,192 shots/query at the tested budgets.

At 4 pairs, signed classical readout accuracy is 91.67%, ideal squared-overlap
quantum readout 77.08%, and 128-shot/candidate readout 72.92%. At 8 pairs those
are 64.58/52.08/48.96%. This encoding changes the function (absolute squared overlap
loses sign information); the gap is not hardware noise. The finite-shot results
are synthetic Born sampling. No benefit justifies IBM jobs. This is not a trained
QNN or a no-go result for other quantum encodings.

Decision before experiment 10: freeze model choices and test fresh seeds 1000–1023,
new load 512, and mild/heavier query noise. Include compressed nearest-key storage,
not only uncompressed keys. Separate build time, batch query time, persistent
payload and query workspace. Success criterion for a narrow stepping stone:
quantized trained head retains >=95% mean recall at n256/sigma.3 with less stored
array payload than compressed-key nearest retrieval. Failure under larger loads
or corruption must be reported. No transformer/energy claim follows either way.

## 10 — Fresh-seed validation and a stronger compressed baseline

Completed 24 fresh random memories per condition, seeds 1000–1023, loads
128/256/512, query noise 0/.3/.8 radians, seven frozen methods (1,512 rows).
At n256/sigma.3, phase binding 24.43%, linear float/int8 96.13/96.14%, ridge
float32/int8 100/100%, exact/quantized-key nearest retrieval 100/100%.
Quantized ridge state is 4,100 bytes; compressed nearest-key arrays 66,564 bytes
(16.24x larger), not just the 263,168-byte uncompressed baseline. The narrow
predeclared >=95% recall/storage criterion passes. All-success seed results do
not guarantee zero errors on future memories; the sample is small and synthetic.

At n512/sigma.3 ridge-int8 falls to 93.51%; at n256/sigma.8 it falls to 80.99%.
At n512/sigma.8 it is 58.78%, below simple linear-int8's 61.15%. Exact and
quantized-key nearest retrieval remain 100% in these tested regimes. Higher
capacity/robustness has not been achieved at the same compressed size.

One warmed timing diagnostic (seed1000,n256,sigma.3,25 batch repeats) gives
median full-batch query times: binding 52.08 us, linear 15.63 us, linear-int8
24.88 us, ridge-float32 65.00 us, ridge-int8 69.67 us, exact 195.87 us,
compressed exact 280.75 us. These include dequantization but exclude build cost.
Ridge-int8 is ~4x faster than compressed exact in this single microbenchmark,
not 100x; it is also slower than unquantized ridge and simple linear memory.
Build costs, warm caches, batch sizes and BLAS threads strongly affect results.
No process-peak-memory or energy measurement was performed.

Final decision: do not propose quantum advantage, a superior general neural net,
or a transformer replacement. Retain a reproducible benchmark and quantized
classical updateable-memory baseline as a stepping stone. If continued, evaluate
held-out structured tasks, ambiguity, absent keys and learned representations,
then actual end-to-end costs including encoder and updates. Novelty must be
reviewed against associative memories, linear attention and quantized models.
