# arXiv Submission Notes

Recommended metadata:

- Title: `Spherical Chamfer Generative Networks: One-Step Continuous Text-Conditional Generation by Joint Hypersphere Matching`
- Authors: `Ivan Shivalov`
- Primary category: `cs.LG`
- Cross-list: `cs.CV`
- Comments: `Preliminary technical report. Code: https://github.com/intexcor/SCGN`
- Code: `https://github.com/intexcor/SCGN`

Abstract:

We introduce Spherical Chamfer Generative Networks (SCGN), a simple one-step conditional image generation prototype trained without adversarial losses, diffusion sampling, paired reconstruction, or explicit optimal-transport solvers. The method maps random noise and a continuous conditioning vector directly to an image in a single forward pass. Its core constraint is geometric: generated and real images are projected onto a fixed-radius hypersphere, and training minimizes a bidirectional Chamfer objective in a joint space formed by image direction and condition direction. The hyperspherical projection prevents the trivial low-variance mean solution that appears under naive pixel matching, while the joint image-condition nearest-neighbor loss encourages conditional routing without class-classification losses. We report preliminary qualitative results on MNIST, Fashion-MNIST, CIFAR-10, and a synthetic text-encoder mode in which each image receives a unique prompt embedding. This technical report is intended to document the method and its initial empirical behavior; large-scale evaluation and comparisons are left for future work.

Account notes:

- Create an arXiv account before submission.
- First-time submitters may need endorsement for `cs.LG` or `cs.CV`.
- If endorsement is requested, start the submission, copy the endorsement request link from arXiv, and send it to one relevant established arXiv author.
- Upload `scgn_arxiv_source.tar.gz` if the compiled PDF preview matches `main.pdf`.

File upload checklist:

- Upload `paper/scgn_arxiv_source.tar.gz`, not `paper/main.pdf`.
- Select `PDFLaTeX` if arXiv does not auto-detect it.
- Top-level TeX file: `main.tex`.
- Keep `main.bbl`; arXiv needs it for the bibliography.
- Keep all files under `figures/`; the paper includes them directly.
- All file names currently use arXiv-safe characters only: `a-z A-Z 0-9 _ + - . , =`.
- After arXiv compiles the source, open the generated preview PDF and compare it with `paper/main.pdf`.
