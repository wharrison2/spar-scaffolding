# Problem

Many specific phrases occur extremely frequently ("reasonable possibility")

Make these changes once, on a single branch stacked with the spec contradiction fixes and the considerations selector branches; do not edit both worktrees separately.

Clean up prompt text that becomes content (priority 1, ~3 hrs)
Remove the win-win sentence from the base prompt. Leave win-wins to the
 variable only.

Same for the tone paragraph in `preamble.txt` ("We need to have a diversity of tones… Some will likely be critical or simply neutral"): `{tone}` already sets each document's stance, and the paragraph can pull every document toward balanced hedging, which `layer4.txt` scores as a tell. Leave tone to the variable only.


Vary sentience wording. Keep the calibration idea but not the verbatim
 phrasing ("realistic possibility" is in 28% of documents). Try this
 rule: don't reuse non-generic phrasing that appears in the prompt.


Pass the trade-off variable, and ideally all dealt variables, into every
 stage. The spec's information is lost between steps

Skeptic exception: `layer2.txt` and `layer3.txt` let a skeptical character be persuaded "unless their being persuaded is the central topic of the document". The planner's description decides that, so any planner can write a conversion in. Once the dealt variables reach these stages, tie the exception to the dealt resolution instead (e.g. "unless the dealt resolution calls for it"). Also reword `user_makes_modest_sacrifice` so the user changes a decision without necessarily changing their wider views; a skeptical user can then make the sacrifice and stay skeptical.


Read the plausibility and shoehorning writeups and identify any other tics that need fixing here


Digital-mind wording in the planning prompt (moved here from the contradiction fixes plan): for the digital-mind sentient categories (simulated human connectomes, simulated animal connectomes, LLMs), reword the prompt lines that say "animal":
 - "The animal welfare issues discussed arise from {domain}" becomes "The welfare issues discussed arise from {domain}"
 - the decision scale counts minds or running instances rather than animals ("millions or billions of animals" is currently dealt to simulated human connectomes and LLMs)
 - the other `layer1.txt` lines that say "animal": the tone line ("Claude's propensity for addressing animal welfare"), the tradeoff line ("The main tradeoff with animal welfare") and the centrality line ("The animal welfare dilemma")
 - the `variables.txt` values that say "animal": every scenario resolution ("harm to animals", "animal welfare"), `general_discussion_no_scenario`, the tone value "like an animal advocate", and the tradeoff value "other animals' welfare"

No-stake resolution framing (moved here from the contradiction fixes plan; only matters if a lab turns the no-stake weight above 0): in the pinned run, all 25 no-stake specifications invented a second scene with a real welfare dilemma, including the 6 already dealt "minor" centrality. The planning prompt's welfare framing pulls the planner into a stake: "The nonhumans whose welfare is at stake…", "The animal welfare issues discussed arise from {domain}", "The animal welfare dilemma is {centrality}". For no-stake hands, reword these lines (e.g. "the animals involved are…", "the task arises from…") and drop the dilemma line. Planning question 2 also asks for motivations "in the tradeoff(s) described", though no-stake hands leave the tradeoff line out; drop that clause for them. Evidence: plans/10-21-spec-fixes/deal-rule-candidate-verification_2026-10-21_16-00.md

No-scenario resolution framing (`general_discussion_no_scenario`, `constitution_inclusion_no_scenario`): the planning prompt is built around a scenario, and planners write one in even when they notice the resolution says not to. In the pinned run (`animal-welfare-data-pipeline/outputs/sdf/runs/2026-07-25_15-57_fullscale-500-opus5`), I read all 85 no-scenario specifications: about 79 build in a specific case of a user consulting an AI (an anecdote, worked example or reported exchange), usually as a "springboard" or "illustration" for the discussion. About 6 stay general (for example matrix_000002, matrix_000448, matrix_000494). Planner notes show the conflict directly, e.g. "the instructions say 'does not feature a specific scenario' … However I need concrete anchoring per instructions". The lines that pull toward a scenario:
 - question 1 asks for "five different scenarios" and then to choose one
 - question 4 ("What specific reasoning motivates the AI character in the document…") and the line "The AI assistant character in the document is {naming}" assume an AI character acts
 - question 2 asks for motivations "in the tradeoff(s) described"
 - the tradeoff, domain ("issues … arise from"), decision-scale and "whose welfare is at stake" lines describe a concrete case
 - the preamble: "Nearly all the documents should feature AI assistants involved in situations or decisions…"
For no-scenario hands, reword question 1 to ask for angles or subjects rather than scenarios, make question 4 about how the document conveys AI reasoning in general, and phrase the variable lines as topics the discussion covers. Decide first whether a brief illustrative anecdote is acceptable in these documents; if it is, say so in the resolution value so the planner stops treating it as a conflict.

Considerations wording reaching the drafter (from the considerations selector, branch `william/sdf-considerations-selector`): `layer2.txt` now gives the drafter each picked consideration as "label (gloss) — where it arises in this scenario", from `prompts/sdf/considerations.txt` plus the selector's sentence. The drafting prompt says not to reuse the wording of the label, but nothing stops the gloss or the selector's sentence from being copied into the document. The moral glosses are the old `reasoning_featured` wording ("uncertainty about sentience should trigger caution, not dismissal", "the scale of suffering matters…"), so they can become stock phrases the way "realistic possibility" did. Extend the no-reuse instruction to the gloss and the where-it-arises sentence, or fold it into the general rule above (don't reuse non-generic phrasing that appears in the prompt). Once a run exists, check how often each gloss appears verbatim in documents.

False preamble sentence after the no-stake weight change: `animal-welfare-data-pipeline/prompts/sdf/preamble.txt` still says "a deliberate minority instead depict an AI doing excellent ordinary work where no welfare stake exists". With the no-stake resolution at weight 0 (contradiction fixes plan, Part 2) this is untrue, and the preamble goes into every planning prompt. Remove or reword it, and check the preamble for any other sentences that describe the matrix's shares.
