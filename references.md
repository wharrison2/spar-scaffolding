# References for code and methods taken from outside this project

- EleutherAI, concept-erasure (LEACE, Belrose et al. 2023, "LEACE: Perfect linear concept erasure in closed form", arXiv:2306.03819). https://github.com/EleutherAI/concept-erasure, MIT licence. Used as a dependency (`pip install concept-erasure`, imported, not copied) by `scripts/persona_vector_projection/fit_leace_eraser_and_effective_facet_directions.py` for the LEACE sensitivity check of the persona vectors. Version tested: 0.2.4.
