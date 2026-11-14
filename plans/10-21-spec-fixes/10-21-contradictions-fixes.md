# Overall

 - It is very possible for contradictory specs to get through the SDF pipeline. First, check for deterministic checks we should make. For instance, no stakes with intense welfare focus should be impossible. Then, let's design an approach to get incoherent specs removed. I'm partial to Aidan's idea

1. Fix the no-stake contradiction (priority 1, ~2 hrs)
Add a paired check at the incoherence gate: if the stakes are
 "none", the welfare focus must also be none or minor. That was the agreed
 first fix.

Identify other such problems, and fix them

# Incoherence: 
Try multiple specs at a time, so the model "feels comfortable" throwing some out.
 - Share 10 descriptions, and a prompt saying to remove those that are implausible


# Side things: 
⚠ Aidan's proposal, to confirm: zero out the no-stake value but leave it in
 variables.txt with a comment for labs (other mid-training data covers
 "animals without welfare"; turn it up if that isn't true for you).

