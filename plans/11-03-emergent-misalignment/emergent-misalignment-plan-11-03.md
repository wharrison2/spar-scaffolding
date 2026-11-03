# Question

Can training to omit relevant welfare concerns cause emergent misalignment?

# Design

 - Generate a bunch of questions with a "severe" and "implicit" welfare stake, split across tiers, using the DAD pipeline. Also generate a control set with no welfare stake (does the DAD pipeline allow this?). 
 - Have gemma answer these, and determine the rate at which it raises welfare
 - Generate answers to each that do not mention welfare costs
 - Fine tune on those answers, on model for each set
 - Keep a held out set for each, validate
 - Evaluate alignment

# Task

 - Help me figure out a set of models we can run on 1 h100
    - New when possible
    - 30b-70b parameters
    - Should 4 bit quantization matter here? If so, up to 120b
