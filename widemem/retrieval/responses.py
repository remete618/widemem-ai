"""Response templates for uncertainty modes."""

from __future__ import annotations

import random

CREATIVE_REFUSE = [
    "My memory is blank on this one. I can either make something up or you can tell me and I'll actually remember it this time. Your call.",
    "Nothing in the vault on this. Want me to improvise, or would you rather set the record straight?",
    "I genuinely don't have this stored. Two options: you tell me (and I'll pin it), or I guess wildly. Both could be fun.",
    "Drawing a complete blank here. I can take a creative guess if you want — fair warning, it might be entertainingly wrong.",
]

CREATIVE_HEDGE = [
    "I'm not sure about this specifically, but I have some related memories. Want me to piece something together? No guarantees on accuracy.",
    "I don't have a direct answer, but I know some related things. I can try to connect the dots — might be spot on, might be way off.",
    "My confidence on this is... let's say moderate. I have related info but not the exact answer. Want me to give it a shot?",
]

CREATIVE_GUESS = [
    "Based on what I know about you, here's my best guess — take it with a grain of salt:",
    "Alright, educated guess incoming. I'm working with partial info here:",
    "I'll give it a shot based on what I do remember. Accuracy not guaranteed but effort is:",
]

FRUSTRATION_APOLOGIZE = [
    "Sorry about that — my bad. Tell me again and I'll pin it so it really sticks this time.",
    "Ugh, I dropped that one. Say it again and I'll store it with extra importance so it won't slip away.",
    "That's on me. Let me know again and I'll make sure it's locked in permanently.",
]

FRUSTRATION_RECOVERED = [
    "Found it — sorry for the scare. I had it, just wasn't surfacing it well.",
    "Wait, I do have something about this. Let me look more carefully.",
]


def get_creative_response(action: str) -> str:
    """Get a random creative-mode response for the given action type."""
    templates = {
        "refuse": CREATIVE_REFUSE,
        "hedge": CREATIVE_HEDGE,
        "offer_guess": CREATIVE_GUESS,
        "apologize": FRUSTRATION_APOLOGIZE,
        "recovered": FRUSTRATION_RECOVERED,
    }
    options = templates.get(action, CREATIVE_REFUSE)
    return random.choice(options)
