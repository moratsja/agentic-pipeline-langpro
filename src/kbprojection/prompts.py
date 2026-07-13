"""
KB Injection Prompts for NLI Problems

Contains both legacy prompts (preserved for backward compatibility) and
improved v2 prompts based on cross-analysis findings.
"""

# =============================================================================
# LEGACY PROMPTS (Preserved for backward compatibility)
# =============================================================================

legacy_prompts = {
    "prompts": [
        {
            "name": "legacy_cot",
            "description": "Chain-of-thought: reason internally then emit only KB facts.",
            "template": "Convert the premise and hypothesis into concise KB injections. Think step by step and show your reasoning clearly. Break the problem into intermediate steps, explain each step, and then give the final answer. Reason step by step internally and return ONLY facts as predicate(arg1, arg2). Don't use underscores in the arguments and use max 2 words per relation\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjections:",
        },
        {
            "name": "legacy_least_to_most",
            "description": "Least-to-most: decompose the task into minimal facts before composing the KB.",
            "template": "Your job is to break complex questions into a sequence of simpler subproblems that can be solved one by one.\nEach subproblem should be small, explicit, and should not require reasoning about later steps.\nAfter planning, output ONLY the final KB injection as predicate(arg1, arg2). Avoid natural language.Don't use underscores in the arguments and use max 2 words per relation \nPremise: ${premise}\nHypothesis: ${hypothesis}\nKBinjection:",
        },
        {
            "name": "legacy_icl",
            "description": "In-context learning with a few exemplars.",
            "template": "You are learning from the examples to emit KB injections only.\nRules:\n\n* Output ONLY KB facts as predicate(arg1, arg2) on separate lines (no extra text).\n* Only use the relations: isa_wn and disj.\n* Don't use underscores in the arguments.\n* Use max 2 words per argument.\n* PREFER BASE FORMS (LEMMAS) for single words (e.g., use \"run\" instead of \"running\", \"eat\" instead of \"eating\").\n* Focus on the meaning shift between Premise and Hypothesis.\n\nExample 1:\nPremise: A little girl in pink boots runs down the street.\nHypothesis: A human is running outdoors.\nKnowledge injection:\nisa_wn(girl, human)\nisa_wn(street, outdoors)\n\nExample 2:\nPremise: A woman soaks her feet in a natural pool in a landscape of rocks, with green covered tents in the background.\nHypothesis: A lady is outdoors\nKnowledge injection:\nisa_wn(woman, lady)\n\nExample 3:\nPremise: A man and a woman hug on a grassy hillside overlooking the countryside in the distance.\nHypothesis: The man and woman are outdoors.\nKnowledge injection:\nisa_wn(hillside, outdoors)\n\nExample 4:\nPremise: A female swimmer getting out of the pool still dripping wet.\nHypothesis: A woman gets out of the pool.\nKnowledge injection:\nisa_wn(female swimmer, woman)\n\nExample 5:\nPremise: A lady looks to her right and holds a mini camera.\nHypothesis: A lady is holding a device.\nKnowledge injection:\nisa_wn(mini camera, device)\n\nExample 6:\nPremise: A dog runs along the shore of a pond with two elegant geese swimming.\nHypothesis: A dog runs along the edge of a pond outdoors.\nKnowledge injection:\nisa_wn(shore, edge)\nisa_wn(pond, outdoors)\n\nExample 7:\nPremise: A man wearing a white shirt is playing the drums.\nHypothesis: A man is playing a musical instrument.\nKnowledge injection:\nisa_wn(drum, musical instrument)\n\nExample 8:\nPremise: A little boy wearing a blue striped shirt has a party hat on his head and is playing in a puddle.\nHypothesis: The party boy is playing in a puddle.\nKnowledge injection:\nisa_wn(little boy, party boy)\n\nExample 9:\nPremise: A little girl is sitting on the counter dangling one foot in the sink whilst holding a dish jet washer.\nHypothesis: A human sitting\nKnowledge injection:\nisa_wn(girl, human)\n\nNow do the next one. You can think of more injections than one, if needed.\nPremise: ${premise}\nHypothesis: ${hypothesis}\nKnowledge injection:\n",
        },
    ]
}

# =============================================================================
# IMPROVED V2 PROMPTS
# Shared rules apply to all new prompt types.
# Examples are intentionally included only for the new ICL prompt.
# =============================================================================

NEW_PROMPT_RULES = """## STRICT RULES

1. Output format:
   [KB_START]
   predicate(arg1, arg2)
   [KB_END]

2. Allowed predicates only:
   - isa_wn(X, Y): X can help prove or normalize Y because X is a kind of Y, implies Y, or is a useful semantic paraphrase toward Y
   - disj(X, Y): X and Y are clearly mutually exclusive

3. Form of arguments:
   - Prefer CCG/prover-friendly lemma heads over long surface phrases
   - Use lemmas for single words whenever possible
   - Keep short multi-word phrases only when the phrase carries essential meaning
   - No determiners
   - No underscores
   - Lowercase unless capitalization is semantically required
   - Prefer the most direct helpful wording over a longer paraphrase
   - Avoid prepositional phrases unless the preposition itself is the mismatch
   - For raw LangPro usefulness, do not remove an essential modifier from a short target phrase when the modifier is needed by the Hypothesis
   - A two-word target phrase can be better than a one-word head when both words form the meaning that must be proved

4. Core reasoning:
   - Focus only on mismatches between Premise and Hypothesis
   - Add every non-trivial bridge needed for the proof
   - If one good bridge is enough, stop
   - Do not stop after the first obvious bridge if another unmatched predicate, object, person, or location remains
   - Check nouns, verbs, adjectives, prepositions, and short compounds separately
   - If two or more mismatches are needed together, output all of them
   - If the sentences share the same object or location, bridge the mismatching event lemma alone
   - Do not return an empty KB when there is a direct non-identical noun, verb, adjective, or preposition bridge that could help the proof
   - Do not add extra event or state bridges just because they are loosely related
   - Do not guess the gold label
   - Do not force entailment or contradiction with speculative facts

5. LangPro usefulness test:
   - Only output relations that are genuinely KB-helpful for LangPro on this pair
   - A fact is helpful when it adds a missing lexical-semantic bridge or a clear incompatibility LangPro can use
   - If a fact only restates already-matching content or adds generic background knowledge with no clear proof role, omit it
   - Prefer one decisive bridge over several weak or redundant ones
   - Prefer a bare event lemma bridge over an object-bound event phrase when the object is not the semantic mismatch
   - For broad activity targets such as play, use the premise event lemma as the bridge rather than combining it with its object
   - Prefer a short noun head over a long noun phrase when the head is the bridge LangPro needs
   - Preserve a short compound target when dropping a modifier would make the relation too weak for the Hypothesis
   - Include a preposition bridge when the only semantic mismatch is a locative relation
   - Include both the event bridge and the object/location bridge when raw LangPro needs both
   - Plausible wording is not enough: choose facts likely to align with parsed lemma heads

6. Direction matters:
   - In simple entailment, bridge from the premise term to the hypothesis term
   - In contradiction with a negated premise or hypothesis, bridge the positive event/entity in the other sentence into the negated scope
   - Choose the direction that makes the prover's job easier, not just the surface word order
   - Think: which term should entail or normalize to which other term?

7. Morphology is not a semantic bridge:
   - Do not add facts only to connect inflectional variants of the same lemma
   - Usually avoid bridges like:
     isa_wn(run, running)
     isa_wn(climb, climbing)
   - Add a fact only when there is a real semantic gap, not just a tense or form difference

8. Using disj:
   - Use disj only for clear event/state incompatibility or genuine opposites
   - Do not use disj for related task actions that could be paraphrases, stages, or context-dependent descriptions of the same event
   - In entailment-like pairs, if two verbs can describe the same event in context, use isa_wn instead of disj
   - For action pairs such as cutting/slicing, aiming/drawing, stopping/riding, boiling/stirring, chasing/catching, prefer isa_wn when the Hypothesis is affirmative
   - If two terms are merely different, sequential, overlapping, or context-dependent, do not use disj
   - If unsure, omit disj

## FINAL CHECK BEFORE ANSWERING

- Would each fact give LangPro a concrete new bridge or contradiction cue on this pair?
- Did each argument look like a short parsed lemma/head rather than a long surface phrase?
- Did you include every needed non-trivial bridge?
- Did you choose the direction that best supports entailment or normalization?
- Did you avoid morphology-only bridges?
- Did you avoid unnecessary extra facts?
- Did you use disj only for genuine opposites?
"""

ICL_V2_TEMPLATE = f"""You are a knowledge base injection assistant. Your task is to generate the smallest set of semantic relations that is genuinely KB-helpful for LangPro on this Premise/Hypothesis pair.

{NEW_PROMPT_RULES}

## EXAMPLES

The examples below are synthetic and are meant to show output shape, short prover-friendly arguments, direction, and coverage.

### Example 1: Complete lexical coverage
Premise: A cook is slicing salmon.
Hypothesis: A chef is cutting fish.
[KB_START]
isa_wn(cook, chef)
isa_wn(slice, cut)
isa_wn(salmon, fish)
[KB_END]

### Example 2: Prefer a head lemma over a longer phrase
Premise: A rider is standing still for a portrait.
Hypothesis: A rider is posing for a portrait.
[KB_START]
isa_wn(stand, pose)
[KB_END]

### Example 3: Negation-aware direction
Premise: There is no child playing on the mat.
Hypothesis: A child is jumping on the mat.
[KB_START]
isa_wn(jump, play)
[KB_END]

### Example 4: disj only for genuine incompatibility
Premise: A robot is sitting.
Hypothesis: A robot is dancing.
[KB_START]
disj(sit, dance)
[KB_END]

### Example 5: Stop at the useful bridges
Premise: A biker is doing a stunt.
Hypothesis: A rider is performing a trick.
[KB_START]
isa_wn(biker, rider)
isa_wn(do, perform)
isa_wn(stunt, trick)
[KB_END]

Premise: ${{premise}}
Hypothesis: ${{hypothesis}}

Generate the knowledge injections:
"""

COT_PROMPT_RULES = """## STRICT RULES

1. Output format:
   [KB_START]
   predicate(arg1, arg2)
   [KB_END]

2. Allowed predicates only:
   - isa_wn(X, Y): X can help prove or normalize Y because X is a kind of Y, implies Y, or is a useful semantic paraphrase toward Y
   - disj(X, Y): X and Y are clearly mutually exclusive

3. Form of arguments:
   - Use lemmas for single words whenever possible
   - Keep short multi-word phrases only when the phrase carries essential meaning
   - Max 2 words per argument
   - No underscores
   - Prefer the most direct helpful wording over a longer paraphrase

4. Core reasoning:
   - Focus only on mismatches between Premise and Hypothesis
   - Add the fewest facts needed to bridge those mismatches
   - If one good bridge is enough, stop
   - Do not add extra event or state bridges just because they are loosely related
   - Do not guess the gold label
   - Do not force entailment or contradiction with speculative facts

5. LangPro usefulness test:
   - Only output relations that are genuinely KB-helpful for LangPro on this pair
   - A fact is helpful when it adds a missing lexical-semantic bridge or a clear incompatibility LangPro can use
   - If a fact only restates already-matching content or adds generic background knowledge with no clear proof role, omit it
   - Prefer one decisive bridge over several weak or redundant ones
   - When the hypothesis has a broad activity predicate such as play, prefer a direct event lemma bridge from the premise verb, such as ride -> play, instead of an object-bound phrase such as ride swing -> play
   - Plausible wording is not enough: cached audit showed isa_wn(walk, move around) did not improve its example, so do not imitate that bridge without proof evidence

6. Direction matters:
   - Choose the direction that makes the prover's job easier, not the surface word order
   - The useful relation may point from a more specific term to a more general term, even if the specific term appears in the Hypothesis
   - Think: which term should entail or normalize to which other term?

7. Morphology is not a semantic bridge:
   - Do not add facts only to connect inflectional variants of the same lemma
   - Usually avoid bridges like:
     isa_wn(run, running)
     isa_wn(climb, climbing)
   - Add a fact only when there is a real semantic gap, not just a tense or form difference

8. Using disj:
   - Use disj only for clear opposites such as open/closed, alive/dead, full/empty
   - If two terms are merely different, sequential, overlapping, or context-dependent, do not use disj
   - If unsure, omit disj

## FINAL CHECK BEFORE ANSWERING

- Would each fact give LangPro a concrete new bridge or contradiction cue on this pair?
- Did you choose the direction that best supports entailment or normalization?
- Did you avoid morphology-only bridges?
- Did you avoid unnecessary extra facts?
- Did you use disj only for genuine opposites?
"""

COT_V2_TEMPLATE = f"""You are a knowledge base injection assistant using chain-of-thought reasoning.

## TASK
Analyze the semantic gap between Premise and Hypothesis, then generate only the KB facts that are genuinely KB-helpful for LangPro on this pair.

{COT_PROMPT_RULES}

## REASONING STEPS

1. Identify key terms in Premise.
2. Identify key terms in Hypothesis.
3. Find only the semantic gaps that actually need a bridge.
4. Decide whether each gap calls for isa_wn, disj, or no fact at all.
5. Keep the final set minimal and concretely helpful for LangPro.
6. Reason step-by-step internally, then output only the KB facts in delimiters.

Premise: ${{premise}}
Hypothesis: ${{hypothesis}}

Generate the knowledge injections:
"""

LASHA_TEMPLATE = """You are an expert in linguistic semantics and logic. You will receive a Natural Language Inference (NLI) problem in English, consisting of a premise sentence and a hypothesis sentence.
You will reason carefully and decide whether the premise entails the hypothesis, which means that if the premise is true, then the hypothesis must also be true under ordinary English meaning and widely accepted background knowledge.
If the answer is "entailment", output a structured explanation that is a set of lexical entailment relations over short phrases that explain why the hypothesis is entailed from the premise.
Lexical entailment should be defined over short phrases that are lemmatized or normalized versions of short phrases occurring in the premise and the hypothesis, e.g., ${PREDICATE_ENTAILMENT}(phrase_1, phrase_2), and it means that phrase_1 is a type of phrase_2, for example, ${PREDICATE_ENTAILMENT}(woman, person), ${PREDICATE_ENTAILMENT}(dog, domestic animal), ${PREDICATE_ENTAILMENT}(huge, very big), and ${PREDICATE_ENTAILMENT}(run, move fast).
Use lexical entailment relations only when they are needed to explain the entailment. If the entailment follows without any non-trivial lexical relation, output an empty set.

Relation formatting rules:
The meaning of a lexical entailment relation has to be acceptable based on common sense, e.g., ${PREDICATE_ENTAILMENT}(woman, blond person) is not acceptable.
A lexical entailment may not express a trivial relation that is obtainable by discarding modifiers, e.g., ${PREDICATE_ENTAILMENT}(blond woman, woman) is not acceptable.
A lexical entailment may not contain redundant words such as auxiliary verbs and the infinitive "to", e.g., ${PREDICATE_ENTAILMENT}(will walk, will move), ${PREDICATE_ENTAILMENT}(is red, is colored), and ${PREDICATE_ENTAILMENT}(to walk, to move) are not acceptable.
Phrases in a lexical entailment have to contain lemmatized words, e.g., ${PREDICATE_ENTAILMENT}(dogs, domestic animals) is not acceptable.
Phrases in a lexical entailment may not contain determiners, e.g., ${PREDICATE_ENTAILMENT}(a dog, a domestic animal) is not acceptable.
Phrases in a lexical entailment may not contain prepositional phrases, e.g., ${PREDICATE_ENTAILMENT}(dog with spots, a domestic animal) is not acceptable.

Examples:
Example 1:
	input:
		premise: Young ladies are playing the guitar.
		hypothesis: A musical instrument is being played by girls.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(young lady, girl), ${PREDICATE_ENTAILMENT}(guitar, musical instrument) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(young ladies, girls), ${PREDICATE_ENTAILMENT}(the guitar, a musical instrument) }
		explanation: phrases in relations may not have determiners, e.g., "a" and "the". "ladies" and "girls" have to use lemmas "lady" and "girl", respectively.

Example 2:
	input:
		premise: A female swimmer getting out of the pool still dripping wet.
		hypothesis: A woman gets out of the pool.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(female swimmer, woman) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(swimmer, woman) }
		explanation: it is a factually wrong relation because not every swimmer is a woman

Example 3:
	input:
		premise: A young girl wearing a pink coat plays with a yellow toy.
		hypothesis: A kid is swinging a toy golf club.
	correct output:
		answer: non-entailment

Example 4:
	input:
		premise: A black race car starts up in front of a crowd of people.
		hypothesis: A car is running.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(start up, run) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(starts up, is running) }
		explanation: "starts up" and "is running" do not contain lemmatized words, and "is" is unnecessary in "is running".

Example 5:
	input:
		premise: A woman dressed in red clothing is dancing inside a crowd of people.
		hypothesis: A woman in red is dancing in a crowd.
	correct output:
		answer: entailment
		relations: { }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(in red clothing, in red) }
		explanation: The phrases in the relation may not include prepositional phrases, e.g., "in red clothing"

Example 6:
	input:
		premise: A tall man with a cap is climbing a cord.
		hypothesis: The man in a hat is climbing a rope.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(cap, hat), ${PREDICATE_ENTAILMENT}(cord, rope) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(cap, hat), ${PREDICATE_ENTAILMENT}(tall man, man) }
		explanation: "${PREDICATE_ENTAILMENT}(cord, rope)" is missing. ${PREDICATE_ENTAILMENT}(tall man, man) is trivial since it includes dropping the adjective "tall".

Example 7:
	input:
		premise: No person is cooking.
		hypothesis: No cook is cooking in the kitchen.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(cook, person) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(person, cook) }
		explanation: The relation does not help to explain "entailment", taking into account that negation reverses a lexical entailment direction. It is also a factually wrong relation because not every person is a cook.

Example 8:
	input:
		premise: A person who is obese is holding a chinchilla.
		hypothesis: A fat person is holding a small animal.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(chinchilla, small animal), ${PREDICATE_ENTAILMENT}(obese, fat) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(fat, obese), ${PREDICATE_ENTAILMENT}(chinchilla, animal) }
		explanation: "${PREDICATE_ENTAILMENT}(fat, obese)" needs to reverse its arguments to align with the entailment direction. "${PREDICATE_ENTAILMENT}(chinchilla, animal)" is not sufficient to explain "entailment" since it misses "small", which is crucial.

Example 9:
	input:
		premise: A little boy is laughing and happily bouncing on a trampoline outside.
		hypothesis: The child is jumping outdoors.
	correct output:
		answer: entailment
		relations: { ${PREDICATE_ENTAILMENT}(little boy, child), ${PREDICATE_ENTAILMENT}(bounce, jump), ${PREDICATE_ENTAILMENT}(outside, outdoors) }
	unwanted output:
		relations: { ${PREDICATE_ENTAILMENT}(boy, child), ${PREDICATE_ENTAILMENT}(outdoors, outside) }
		explanation: "${PREDICATE_ENTAILMENT}(bounce, jump)" is missing. "${PREDICATE_ENTAILMENT}(little boy, child)" is preferred over "${PREDICATE_ENTAILMENT}(boy, child)" as the former is more acceptable. "${PREDICATE_ENTAILMENT}(outdoors, outside)" needs to reverse its arguments to align it to the entailment direction.

Now process the following input while strictly following the above instructions and formatting.
Output exactly in one of the following formats:
	answer: entailment
	relations: { ... }
	or
	answer: non-entailment

input:
	premise: ${PREMISE}
	hypothesis: ${HYPOTHESIS}
correct output:"""

# Combined prompt registry (for backward compatibility, maps old names to legacy)
prompts = {
    "prompts": [
        # Legacy prompts
        {
            "name": "legacy_cot",
            "description": "[LEGACY] Chain-of-thought prompt (use cot for improved version)",
            "template": legacy_prompts["prompts"][0]["template"],
        },
        {
            "name": "legacy_least_to_most",
            "description": "[LEGACY] Least-to-most decomposition prompt",
            "template": legacy_prompts["prompts"][1]["template"],
        },
        {
            "name": "legacy_icl",
            "description": "[LEGACY] In-context learning prompt (use icl for improved version)",
            "template": legacy_prompts["prompts"][2]["template"],
        },
        # Improved prompts (now default)
        {
            "name": "icl",
            "description": "Improved ICL with shared rules and in-context examples",
            "template": ICL_V2_TEMPLATE,
        },
        {
            "name": "cot",
            "description": "Improved CoT with shared rules and structured reasoning",
            "template": COT_V2_TEMPLATE,
        },
        {
            "name": "lasha",
            "description": "NLI-first lexical entailment explanation prompt with entails(...) relations",
            "template": LASHA_TEMPLATE,
        },
    ]
}


def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name from the in-memory dictionary `prompts`."""
    for prompt in prompts.get("prompts", []):
        if prompt.get("name") == prompt_name:
            return prompt.get("template", "")
    raise KeyError(
        f"Prompt '{prompt_name}' not found. Available: {[p['name'] for p in prompts['prompts']]}"
    )


def fill_prompt(
    prompt_name: str,
    premises: list,
    hypothesis: str,
    variables: dict[str, object] | None = None,
) -> str:
    """Get a prompt template and fill in the premises and hypothesis."""
    template_str = get_prompt(prompt_name)
    premise_text = "\n".join(premises) if isinstance(premises, list) else premises
    predicate_defaults = {
        "entailment": "entails",
        "disjunction": "disj",
    }
    raw_variables = dict(variables or {})
    predicate_overrides = raw_variables.pop("predicates", {}) or {}
    predicates = {
        **predicate_defaults,
        **{str(key): str(value) for key, value in dict(predicate_overrides).items()},
    }
    substitutions = {
        "premise": premise_text,
        "hypothesis": hypothesis,
        "PREMISE": premise_text,
        "HYPOTHESIS": hypothesis,
        "PREDICATE_ENTAILMENT": predicates["entailment"],
        "PREDICATE_DISJUNCTION": predicates["disjunction"],
    }
    substitutions.update({str(key): str(value) for key, value in raw_variables.items()})

    prompt = template_str
    for key, value in substitutions.items():
        prompt = prompt.replace("${" + key + "}", value)
    return prompt


def list_prompts() -> list:
    """Return a list of available prompt names."""
    return [p["name"] for p in prompts.get("prompts", [])]
