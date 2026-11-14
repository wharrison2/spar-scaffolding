# Ordered epistemic-considerations selector
After the scenario description, give an LLM the ordered list and have it pick
 the first consideration that fits naturally, answering "none" most of the
 time.
Use a run-level tracker so categories drop down the list once their target
 counts are met.
Re-assess reasoning_featured in the same pass.
Aidan: substrate neutrality is plausibly the most important value and
 "barely showed up at all"; naturalness appears organically ~10% of the time.
Keep it genuinely tunable for labs.
Merge the two lists, putting what we care about most near the top.


# Epistemic considerations variable
These should probably not be selected at the same phase as all the other variables, and should maybe not be drawn randomly (see concern 4 in the shoehorning audit, where problems were caused by the `reasoning_featured` variable)

List (roughly in decreasing order of frequency, with AI stuff at the bottom):
Give 

NONE (75%)
Weighing evidence for sentience (10%)
Deciding under uncertainty (precautionary principle) (8%)
Weighing species against each other (5%)
Weighing evidence for harm from a particular practice (5%)
Deferring to specialists (3%)
Assessing reliability of welfare claims (3%)
Using proxy measures of welfare (3%)
Considering nth order effects (3%)
Avoiding both anthropomorphism and anthropodenial (2%)
Uncertainty in quality of wild animal lives (1%)
Burdens of proof for sentience (1%)
Separating moral and empirical uncertainty (1%)
Absence of evidence vs evidence of absence (1%)

When AI is inolved:
Weighing theories of consciousness and their implications for AI (20%)
Difficulties of AI self-reporting welfare (10%)

These are rough guesses, and will be tuneable as other variables are 