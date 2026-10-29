"""Pinned fixtures for the routing-agreement smoke (Experiment B).

Everything here is frozen before the first run: model revisions, the probe set,
the quantization contract (4-bit symmetric RTN, block 32 along the last dim),
the variant order, and the pre-registered expectation bands. Thresholds are
never re-tuned after seeing results; a wrong band is recorded as a miss.

Probe prompts are recorded verbatim so the probe set can be reconstructed
without this file changing later. Every prompt is tokenized without special
tokens and truncated/padded to ``SEQ_LEN`` ids (see ``smoke.encode_prompt``).
"""

from __future__ import annotations

SEED = 20261029

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

PRIMARY = {
    "label": "tinymixtral",
    "repo_id": "Isotonic/TinyMixtral-4x248M-MoE",
    "revision": "1e3516176a6279ce93923060fcad848dc043af79",
    "dtype": "float32",
    "n_layers": 12,
    "n_experts": 4,
    "top_k": 2,
}

SECONDARY = {
    "label": "olmoe",
    "repo_id": "allenai/OLMoE-1B-7B-0924",
    "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5",
    "dtype": "bfloat16",
    "n_layers": 16,
    "n_experts": 64,
    "top_k": 8,
}

# Secondary model is attempted only if these budgets hold (spec Experiment B).
OLMOE_RAM_BUDGET_GB = 20.0
OLMOE_TIME_BUDGET_S = 600.0

# ---------------------------------------------------------------------------
# Probe set — 24 prompts, 6 per domain, fixed order, 96 tokens each
# ---------------------------------------------------------------------------

SEQ_LEN = 96
DOMAINS = ("code", "math", "factual", "narrative")

_PROMPTS: dict[str, list[str]] = {
    "code": [
        "Write a Python class that implements a least-recently-used cache with a fixed "
        "capacity. The class should expose get(key) and put(key, value) methods, both "
        "running in constant amortized time. Internally keep a dictionary from keys to "
        "nodes plus a doubly linked list that tracks recency, moving a node to the front "
        "on every access and evicting the tail node when the capacity is exceeded. Add a "
        "docstring, type hints, and a short example that inserts three keys, reads one of "
        "them back, and shows which key is evicted next.",
        "Given a table of orders with columns customer_id, order_date, and total_cents, "
        "write a SQL query that returns each customer's third most recent order. Use a "
        "window function that partitions by customer, orders by order_date descending, "
        "and filters on the row number rather than on a self join. Handle customers with "
        "fewer than three orders by excluding them, and explain why the filter has to sit "
        "in an outer query or in a common table expression instead of the WHERE clause of "
        "the same SELECT that computes the row number.",
        "Explain how Rust's borrow checker treats a function that takes a mutable slice of "
        "a vector and also needs to read the vector's length. Write a function that "
        "increments every element of the slice by one, then another that returns the "
        "index of the first negative value. Point out where a second mutable borrow would "
        "be rejected at compile time, how to restructure the code with split_at_mut, and "
        "why returning an index instead of a reference sidesteps the lifetime problem.",
        "Write a JavaScript function that fetches a JSON document with exponential "
        "backoff, retrying up to five times on network errors and on server responses in "
        "the 500 range. Each delay should double starting from a quarter second, and the "
        "function should reject with the last error once the budget is exhausted. Use "
        "async and await, an AbortController for cancellation, and avoid retrying on "
        "client errors in the 400 range. Finish with a small usage example that logs the "
        "parsed field named items.",
        "Show a shell pipeline that finds every Python file under a source directory, "
        "skips virtual environments and caches, and runs a formatter over the files in "
        "batches. Use find with a null byte separator so that spaces in file names cannot "
        "split an argument, pipe the list through xargs with a maximum argument count, and "
        "print a summary line at the end. Explain why the naive version that pipes find "
        "output into a while read loop can drop the exit status of the formatter and how "
        "to preserve it with the pipefail option.",
        "Implement a singly linked list in C with push, pop, and reverse operations. "
        "Define the node struct with an integer value and a next pointer, allocate nodes "
        "with malloc, and free them in a destroy function that walks the chain. The "
        "reverse function should relink the pointers iteratively without allocating new "
        "nodes. Discuss the three-pointer technique, what happens on an empty list, and "
        "why a recursive reverse risks a stack overflow for long lists. Include a main "
        "function that builds a list of five values and prints it forwards.",
    ],
    "math": [
        "Compute the derivative of f of x equals the natural logarithm of one plus x "
        "squared, raised to the third power. State the chain rule step by step: first "
        "differentiate the outer cube, then the logarithm, then the inner polynomial, and "
        "multiply the three factors. Simplify the result into a single fraction, identify "
        "the domain where the expression is differentiable, and evaluate the derivative at "
        "x equal to one. Then find the second derivative at the same point and comment on "
        "the sign of the first derivative on either side of zero.",
        "A committee of four people must be chosen from a group of seven women and five "
        "men. Count the number of committees that contain exactly two women, and then "
        "count the committees that contain at least one man. Explain why the second count "
        "is easier to obtain by subtracting the all-women committees from the total. Then "
        "compute the probability that a uniformly random four-person committee contains "
        "exactly two women, and verify that the probabilities of zero, one, two, three, "
        "and four women sum to one.",
        "Let A be a real three by three matrix whose rows are linearly independent. Show "
        "that the columns are also linearly independent, and explain the role of the "
        "determinant in that argument. Suppose one row is replaced by the sum of the other "
        "two rows: describe how the rank and the determinant change, and whether the "
        "column space shrinks. Finish by stating the dimension theorem for linear maps "
        "between finite dimensional spaces and applying it to the map defined by A acting "
        "on three dimensional vectors.",
        "A medical test detects a disease in ninety-nine percent of infected people and "
        "wrongly flags one percent of healthy people. The disease affects one person in "
        "ten thousand. Compute the probability that a person who tests positive actually "
        "has the disease, writing out Bayes' rule, the prior, the likelihood, and the "
        "evidence term. Then recompute the same probability when the prevalence rises to "
        "one percent, and explain in plain language why the posterior changes so sharply "
        "with prevalence even though the test itself did not change.",
        "Determine whether the series whose terms are n factorial divided by n to the "
        "power n converges. Apply the ratio test, simplify the ratio into a product that "
        "tends to one over e, and conclude. Then compare with the series of one over n "
        "squared and explain why one converges while the harmonic series diverges. State "
        "the integral test and use it to justify the convergence of the p-series for p "
        "strictly greater than one, and finally bound the tail of the p-series with an "
        "integral.",
        "Consider minimizing the quadratic function given by one half x transpose A x "
        "minus b transpose x, where A is symmetric positive definite. Derive the gradient "
        "descent update with step size alpha, determine the largest step size that keeps "
        "the iteration stable in terms of the largest eigenvalue of A, and identify the "
        "condition number that governs the convergence rate. Then write the update for a "
        "two by two diagonal A with entries four and one, choose a step size, and explain "
        "why the slow direction dominates the error.",
    ],
    "factual": [
        "Describe the formation of the Andes mountain range. Explain how the subduction of "
        "an oceanic plate beneath the South American plate produces crustal thickening, "
        "volcanism, and a high plateau, and why the range is narrow in some latitudes and "
        "broad in others. Mention the role of the Nazca plate, the age of the descending "
        "lithosphere, and the sediment supply from the Amazon basin to the trench. Then "
        "contrast this ocean-continent setting with the collision of two continental "
        "plates that built the Himalayas.",
        "Explain how movable type printing changed the circulation of texts in fifteenth "
        "century Europe. Cover the earlier use of woodblock printing, the innovation of "
        "reusable metal casts and oil-based ink, and the economics of setting a page. "
        "Describe how the reduced cost per copy shifted the scale of publishing from "
        "scribal workshops to commercial presses, and how the resulting abundance of "
        "pamphlets interacted with religious controversy and with the growth of vernacular "
        "literacy. Note one important limit on the speed of that change.",
        "Summarize the periodic trends in electronegativity across a row and down a group "
        "of the periodic table. Explain the underlying causes in terms of effective "
        "nuclear charge and atomic radius, and connect the trend to the polarity of bonds "
        "and to the acidity of binary hydrogen compounds. Give two examples that follow "
        "the trend and one apparent exception, and explain the exception with atomic size "
        "and shielding rather than with a single number on a scale. Then explain why the "
        "noble gases are usually left out of the trend entirely, and why the scale is "
        "defined by a convention rather than measured directly.",
        "Explain how stars on the main sequence produce energy through hydrogen fusion. "
        "Describe the proton-proton chain and the carbon-nitrogen-oxygen cycle, and say "
        "which dominates at which temperature. Discuss why the luminosity of a main "
        "sequence star scales steeply with its mass, how the star remains in hydrostatic "
        "equilibrium, and what happens to the core once hydrogen is exhausted. Then "
        "explain how observations of a cluster's turnoff point yield its age.",
        "Describe the role of mitochondria in cellular energy metabolism. Explain how the "
        "electron transport chain builds a proton gradient across the inner membrane, how "
        "ATP synthase converts that gradient into chemical energy, and why oxygen is the "
        "terminal electron acceptor in aerobic respiration. Contrast the yield of ATP from "
        "glycolysis alone with the yield from complete oxidation, and mention the "
        "endosymbiotic origin of mitochondria and the evidence from their own genome.",
        "Explain the principle of comparative advantage and how it predicts gains from "
        "trade between two countries even when one is more productive in every industry. "
        "State the assumptions about factor mobility, transport costs, and constant "
        "returns, then describe how a tariff affects consumer prices, producer surplus, "
        "and total welfare. Distinguish the static welfare loss from the dynamic effects "
        "on specialization, and give one reason why the empirical estimates of the losses "
        "are often small relative to the political attention the policy receives.",
    ],
    "narrative": [
        "The lighthouse keeper had counted the same nine ships that week, so the tenth "
        "one, arriving without lights in the fog, made him set down his tea and reach for "
        "the logbook. He wrote the time, the bearing, and the fact that the vessel did not "
        "answer the signal lamp. Outside, the sea threw itself against the rocks in long "
        "grey ropes, and the gull that lived on the north railing refused to move. He "
        "climbed the stairs twice to be sure of what he had seen, and both times the shape "
        "was still there, closer, patient, running dark.",
        "At the train station they stood on opposite sides of a pillar because neither "
        "wanted to be the first to speak. The departure board clicked through its "
        "permutations above them, and a woman with a cello case argued quietly with a "
        "porter. You could have called, he said at last, and she said she had, twice, and "
        "that the second time she had let it ring until the recorded voice asked her to "
        "leave a message and she had left one. The train for the coast was announced. He "
        "picked up her bag before she could, and they walked together toward the platform.",
        "By the time the market opened, the street had already been awake for hours. Men "
        "unloaded crates of oranges still cold from the night, a woman arranged bundles of "
        "parsley into a green wall, and somewhere behind the stalls a radio played a song "
        "about a river. Steam rose from the tea stall and hung in the cold air, and the "
        "first customers drifted in with their collars up, testing the peaches with their "
        "thumbs. The light came slowly, sliding down the rooftops, until the whole street "
        "was gold and everyone began to talk at once.",
        "The generation ship had been travelling for four hundred years when the child "
        "asked what a sky was. The teacher showed her the archive footage again, and the "
        "children watched the clouds move and argued about whether the water in the "
        "recording could really fall from above. In the corridor outside, the hydroponics "
        "fans turned with their endless patience, and the hull carried its low note through "
        "the floor plates. She decided that a sky must be something like the ceiling of the "
        "atrium, only farther, and she wrote that down in her notebook before the bell.",
        "It had been raining since the funeral, which the detective took as a personal "
        "insult. He stood under the awning of the closed pharmacy and watched the house "
        "across the street, where a single lamp burned behind a curtain that had not moved "
        "in an hour. The case was three days old and already bent out of shape: a stolen "
        "car that nobody reported, a key that fit no lock in the building, and a witness "
        "who kept changing the colour of the coat. He lit a cigarette, put it out, and "
        "crossed the road.",
        "Every summer we went back to the river, and every summer my grandmother insisted "
        "that the water was colder than the year before. She would sit on the flat stone "
        "with her skirt tucked under her and tell me which rocks to avoid, though I never "
        "once saw her swim. The current carried leaves past in a slow procession, and the "
        "bridge upstream hummed whenever a truck crossed it. Years later I understood that "
        "she had been keeping watch, the way you keep watch over something you are not "
        "ready to give up.",
    ],
}

PROMPTS = [{"domain": domain, "text": text} for domain in DOMAINS for text in _PROMPTS[domain]]

# ---------------------------------------------------------------------------
# Quantization contract (frozen) and variants
# ---------------------------------------------------------------------------

QUANT_BITS = 4
QUANT_BLOCK = 32

# Applied in this order. The order lets every variant be reached from the loaded
# weights with only the router (a few hundred KB) backed up:
#   base            -> nothing
#   router_shuffled -> permute router rows, then restore
#   experts_q4      -> QDQ experts in place and leave them QDQ'd
#   all_q4          -> QDQ the router on top of the already-QDQ'd experts
#                      (= base experts QDQ'd once + router QDQ'd once)
VARIANTS = ("base", "experts_q4", "all_q4", "router_shuffled")

# Actual capture order. The shuffle runs on pristine weights; expert QDQ then
# stays applied so all_q4 only adds the router, which keeps peak memory at one
# model copy (needed for the 14 GB OLMoE repeat).
EXECUTION_ORDER = ("base", "router_shuffled", "experts_q4", "all_q4")

VARIANT_NOTES = {
    "base": "fp32/bf16 reference routing",
    "experts_q4": "all expert tensors quantize-dequantized, router untouched",
    "all_q4": "expert tensors + router gate weight quantize-dequantized",
    "router_shuffled": "per-layer seed permutation of router rows, experts left in place (control)",
}

# ---------------------------------------------------------------------------
# Pre-registered expectations (written before the first run)
# ---------------------------------------------------------------------------
# Each entry: (id, model label, variant, text, kind, params)
# kind "min": observed >= param; "lt_variant": observed < other variant;
# "near_chance": |observed - chance| <= param (absolute tolerance).

EXPECTATIONS = [
    {
        "id": "E1",
        "model": "tinymixtral",
        "variant": "experts_q4",
        "kind": "min",
        "param": 0.95,
        "text": "experts_q4 vs base overall set agreement >= 0.95",
    },
    {
        "id": "E2",
        "model": "tinymixtral",
        "variant": "all_q4",
        "kind": "lt_variant",
        "param": "experts_q4",
        "text": "all_q4 vs base overall set agreement < experts_q4",
    },
    {
        "id": "E3",
        "model": "tinymixtral",
        "variant": "router_shuffled",
        "kind": "near_chance",
        "param": 0.06,
        "text": "router_shuffled vs base overall set agreement within 0.06 of chance",
    },
    {
        "id": "E4",
        "model": "olmoe",
        "variant": "experts_q4",
        "kind": "min",
        "param": 0.95,
        "text": "experts_q4 vs base overall set agreement >= 0.95 (k=8, E=64)",
    },
    {
        "id": "E5",
        "model": "olmoe",
        "variant": "all_q4",
        "kind": "lt_variant",
        "param": "experts_q4",
        "text": "all_q4 vs base overall set agreement < experts_q4",
    },
    {
        "id": "E6",
        "model": "olmoe",
        "variant": "router_shuffled",
        "kind": "near_chance",
        "param": 0.01,
        "text": "router_shuffled vs base overall set agreement within 0.01 of chance",
    },
]
